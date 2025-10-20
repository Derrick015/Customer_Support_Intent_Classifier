import concurrent.futures
from google import genai
from google.cloud import bigquery
from google.genai.types import EmbedContentConfig
from datetime import datetime
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler
import logging
import numpy as np
from tqdm import tqdm
from google.cloud.bigquery_storage import BigQueryReadClient
import time
import ast

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

bigquery_client = bigquery.Client(project='deron-innovations')
bqstorage_client = BigQueryReadClient()
project = "deron-innovations"
location = "us-central1"
client = genai.Client(vertexai=True, project=project, location=location) 




def add_gemini_embeddings(
    df,
    text_column,
    model="gemini-embedding-001",
    task_type = 'RETRIEVAL_QUERY',
    embedding_column="embedding",
    batch_size=1,
    max_workers=8,
    client=client
):
    """
    Adds Gemini embeddings to a DataFrame column using the Google Gemini API.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the text data.
        text_column (str): The name of the column in df containing the text to embed.
        embedding_column (str, optional): The name of the new column to store embeddings. Defaults to "embedding".
        batch_size (int, optional): Number of rows to process in each batch. Defaults to 1.
        max_workers (int, optional): Number of threads for parallel processing. Defaults to 16.
        project (str, optional): Google Cloud project ID. Defaults to the provided value.
        location (str, optional): Google Cloud location. Defaults to "us-central1".

    Returns:
        pd.DataFrame: The input DataFrame with an added column containing the embeddings.
    
    Note: 
        Set batch size to 1 for gemini-embedding-001 as they only support 1 instance per request - 22-05-2025. For others you can go up to 250
    """



    def embed_batch(batch):
        response = client.models.embed_content(
            model=model,
            contents=batch,
            config=EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in response.embeddings]

    batches = [df[text_column].iloc[i:i+batch_size].tolist() for i in range(0, len(df), batch_size)]
    embeddings = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(embed_batch, batches)
        for batch_embeddings in tqdm(results, total=len(batches), desc="Embedding batches"):
            embeddings.extend(batch_embeddings)
    df[embedding_column] = embeddings
    df=df.copy()
    return df

def process_similarity_batch_multi(
    df_batch,
    collections,
    query_embedding_col='embedding',
    top_n=10
):
    """
    Compute top-N cosine similarities for a batch of queries against multiple collections.

    Args:
        df_batch (pd.DataFrame): Slice of the query DataFrame.
        collections (list[dict]): Each dict must contain:
            - 'name': str, unique name/prefix for the collection output columns
            - 'df': pd.DataFrame, the collection DataFrame
            - 'output_col': str, column containing the output label/id
            - 'embedding_col': str, column containing the embedding list/array
            - 'document_col': str, column containing the source document text
        query_embedding_col (str): Column name in df_batch for embeddings.
        top_n (int): Number of top results to return per collection.

    Returns:
        dict[name -> dict[str, list[list]]]: For each collection name returns dict with keys
            'outputs', 'scores', 'documents' mapping to per-row lists.
    """
    # Convert lists of embeddings to numpy arrays
    product_embeddings = np.vstack(df_batch[query_embedding_col].values)

    per_collection_results = {}
    for col_spec in collections:
        name = col_spec.get('name', col_spec.get('output_col', 'collection'))
        df_c = col_spec['df']
        output_col = col_spec['output_col']
        embedding_col = col_spec['embedding_col']
        document_col = col_spec['document_col']

        collection_embeddings = np.vstack(df_c[embedding_col].values)
        sim = cosine_similarity(product_embeddings, collection_embeddings)
        k = min(top_n, sim.shape[1])
        topn_idx = np.argsort(-sim, axis=1)[:, :k]
        topn_scores = np.take_along_axis(sim, topn_idx, axis=1)

        outputs = [[df_c.iloc[i][output_col] for i in idx] for idx in topn_idx]
        documents = [[df_c.iloc[i][document_col] for i in idx] for idx in topn_idx]

        per_collection_results[name] = {
            'outputs': outputs,
            'scores': [list(score_row) for score_row in topn_scores],
            'documents': documents
        }

    return per_collection_results


def normalize_scores(scores, method='minmax'):
    scores = np.array(scores)
    if method == 'minmax':
        min_s, max_s = scores.min(), scores.max()
        if max_s == min_s:
            return np.ones_like(scores)  # or zeros, if you prefer
        return (scores - min_s) / (max_s - min_s)
    elif method == 'zscore':
        mean_s, std_s = scores.mean(), scores.std()
        if std_s == 0:
            return np.zeros_like(scores)
        return (scores - mean_s) / std_s
    elif method == 'l1':
        norm = np.sum(np.abs(scores))
        if norm == 0:
            return np.zeros_like(scores)
        return scores / norm
    elif method == 'l2':
        norm = np.linalg.norm(scores)
        if norm == 0:
            return np.zeros_like(scores)
        return scores / norm
    elif method == 'sigmoid':
        return 1 / (1 + np.exp(-scores))
    else:
        raise ValueError("Unknown normalization method")



def batch_process_similarity(
    df_query,
    collections,
    query_embedding_col='embedding',
    batch_size=1000,
    top_n=10
):
    """
    Batch process similarity for multiple collections and return df_query augmented with
    per-collection top-N outputs, scores and documents. No explicit ID column needed; order is preserved.

    Args:
        df_query (pd.DataFrame): Query DataFrame with an embedding column.
        collections (list[dict]): Each dict must contain:
            - 'name': str, unique name/prefix for the collection output columns
            - 'df': pd.DataFrame, the collection DataFrame
            - 'output_col': str, column containing the output label/id
            - 'embedding_col': str, column containing the embedding list/array
            - 'document_col': str, column containing the source document text
        query_embedding_col (str): Column name in df_query for embeddings. Default 'embedding'.
        batch_size (int): Number of rows per batch.
        top_n (int): Number of top results to keep per collection.

    Returns:
        pd.DataFrame: Copy of df_query with additional columns for each collection:
            '{name}_top{top_n}_output', '{name}_top{top_n}_score', '{name}_top{top_n}_document'
    """
    num_rows = len(df_query)
    logging.info(
        f"Starting batch similarity (multi-collection). Rows: {num_rows}, Batch size: {batch_size}, Top N: {top_n}, Collections: {[c.get('name', c.get('output_col', 'collection')) for c in collections]}"
    )

    # Prepare accumulators per collection
    accumulators = {}
    for col_spec in collections:
        name = col_spec.get('name', col_spec.get('output_col', 'collection'))
        accumulators[name] = {
            'outputs': [],
            'scores': [],
            'documents': []
        }

    # Iterate batches
    for start in tqdm(range(0, num_rows, batch_size), desc="Processing batches"):
        end = min(start + batch_size, num_rows)
        df_batch = df_query.iloc[start:end]

        batch_results = process_similarity_batch_multi(
            df_batch=df_batch,
            collections=collections,
            query_embedding_col=query_embedding_col,
            top_n=top_n
        )

        for name, res in batch_results.items():
            accumulators[name]['outputs'].extend(res['outputs'])
            accumulators[name]['scores'].extend(res['scores'])
            accumulators[name]['documents'].extend(res['documents'])

    # Attach results to a copy of the query df
    df_out = df_query.copy()
    for col_spec in collections:
        name = col_spec.get('name', col_spec.get('output_col', 'collection'))
        df_out[f'{name}_top{top_n}_output'] = accumulators[name]['outputs']
        df_out[f'{name}_top{top_n}_score'] = accumulators[name]['scores']
        df_out[f'{name}_top{top_n}_document'] = accumulators[name]['documents']

    logging.info("Finished batch similarity (multi-collection). Augmented df shape: %s", df_out.shape)
    return df_out



def process_similarity_single_query(
    query_embedding,
    collections,
    top_n=10
):
    """
    Compute top-N cosine similarities for a single query embedding against multiple
    collections and return a one-row DataFrame with per-collection results.

    Args:
        query_embedding (list | np.ndarray): The query vector.
        collections (list[dict]): Each dict must contain:
            - 'name': str, unique name/prefix for the collection output columns
            - 'df': pd.DataFrame, the collection DataFrame
            - 'output_col': str, column containing the output label/id
            - 'embedding_col': str, column containing the embedding list/array
            - 'document_col': str, column containing the source document text
        top_n (int): Number of top results to return per collection.

    Returns:
        pd.DataFrame: One-row DataFrame with columns, for each collection name:
            '{name}_top{top_n}_output', '{name}_top{top_n}_score', '{name}_top{top_n}_document'
    """
    logging.info(
        "Starting single-query similarity (multi-collection). Top N: %s, Collections: %s",
        top_n,
        [c.get('name', c.get('output_col', 'collection')) for c in collections]
    )

    # Ensure numpy array shape (1, d)
    q = np.asarray(query_embedding, dtype=float)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    elif q.ndim != 2 or q.shape[0] != 1:
        # Flatten any higher dims to 1D then reshape
        q = q.reshape(1, -1)

    result_cols = {}

    for col_spec in collections:
        name = col_spec.get('name', col_spec.get('output_col', 'collection'))
        df_c = col_spec['df']
        output_col = col_spec['output_col']
        embedding_col = col_spec['embedding_col']
        document_col = col_spec['document_col']

        if len(df_c.get(embedding_col, [])) == 0:
            # No embeddings in collection; write empty results
            result_cols[f'{name}_top{top_n}_output'] = [ [] ]
            result_cols[f'{name}_top{top_n}_score'] = [ [] ]
            result_cols[f'{name}_top{top_n}_document'] = [ [] ]
            continue

        collection_embeddings = np.vstack(df_c[embedding_col].values)
        sim = cosine_similarity(q, collection_embeddings)[0]
        k = min(top_n, sim.shape[0])
        if k <= 0:
            result_cols[f'{name}_top{top_n}_output'] = [ [] ]
            result_cols[f'{name}_top{top_n}_score'] = [ [] ]
            result_cols[f'{name}_top{top_n}_document'] = [ [] ]
            continue

        topn_idx = np.argsort(-sim)[:k]
        outputs = [df_c.iloc[i][output_col] for i in topn_idx]
        scores = [float(sim[i]) for i in topn_idx]
        documents = [df_c.iloc[i][document_col] for i in topn_idx]

        result_cols[f'{name}_top{top_n}_output'] = [outputs]
        result_cols[f'{name}_top{top_n}_score'] = [scores]
        result_cols[f'{name}_top{top_n}_document'] = [documents]

    df_out = pd.DataFrame(result_cols)
    logging.info("Finished single-query similarity. Output columns: %s", list(df_out.columns))
    return df_out

def resolve_topn_columns_for_collection(df, name):
    """
    Given a DataFrame augmented by batch_process_similarity and a collection name,
    infer the available top-N columns for outputs and scores for that collection.

    Returns (output_col, score_col) or (None, None) if not found.
    """
    pattern_output_prefix = f"{name}_top"
    pattern_output_suffix = "_output"
    matches = []
    for col in df.columns:
        if col.startswith(pattern_output_prefix) and col.endswith(pattern_output_suffix):
            try:
                middle = col[len(pattern_output_prefix):-len(pattern_output_suffix)]
                n_val = int(middle)
                matches.append((n_val, col))
            except Exception:
                continue
    if not matches:
        return None, None
    matches.sort(key=lambda x: x[0], reverse=True)
    chosen_n, out_col = matches[0]
    score_col = f"{name}_top{chosen_n}_score"
    if score_col not in df.columns:
        return None, None
    return out_col, score_col


def plot_selected_collection_top5_bar(
    df_one_row,
    normalization=None,
    top_n=5,
    percent_fmt=True,
    title=None,
    ax=None
):
    """
    For a single-row DataFrame with a 'selected_collection' column and per-collection
    top-N output/score columns, plot a bar chart of the selected collection's top-K
    outputs with normalized scores as percentages.

    Args:
        df_one_row (pd.DataFrame): DataFrame containing exactly one row.
        normalization (str): One of {'softmax','minmax','l1','l2','zscore','sigmoid'}.
                             'softmax' yields percentages that sum to 100.
        top_n (int): Number of items to plot (defaults to 5).
        percent_fmt (bool): Multiply normalized values by 100 for percentage display.
        title (str|None): Optional plot title. Defaults to '<collection> top <K>'.
        ax (matplotlib.axes.Axes|None): Optional axes to draw on. If None, creates a new figure.

    Returns:
        (fig, ax, plot_df): Matplotlib objects and a tidy DataFrame with 'output' and 'score_pct'.
    """
    if not isinstance(df_one_row, pd.DataFrame) or len(df_one_row) == 0:
        raise ValueError("df_one_row must be a DataFrame with at least one row")
    row = df_one_row.iloc[0]
    if 'selected_collection' not in df_one_row.columns:
        raise ValueError("Missing 'selected_collection' column in DataFrame")
    collection_name = row['selected_collection']
    if not isinstance(collection_name, str) or not collection_name:
        raise ValueError("'selected_collection' must be a non-empty string")

    out_col, score_col = resolve_topn_columns_for_collection(df_one_row, collection_name)
    if out_col is None or score_col is None:
        raise ValueError(f"Could not resolve top-N columns for collection '{collection_name}'")

    outputs = row[out_col]
    scores = row[score_col]
    if isinstance(outputs, str):
        try:
            outputs = ast.literal_eval(outputs)
        except Exception:
            outputs = [outputs]
    if isinstance(scores, str):
        try:
            scores = ast.literal_eval(scores)
        except Exception:
            scores = [float(scores)] if scores not in (None, '') else []

    outputs = list(outputs) if outputs is not None else []
    scores = list(scores) if scores is not None else []

    if len(outputs) == 0 or len(scores) == 0:
        raise ValueError(f"No outputs/scores found in columns '{out_col}'/'{score_col}'")

    k = min(top_n, len(outputs), len(scores))
    outputs = outputs[:k]
    scores = scores[:k]

    # Normalize scores
    scores_norm = apply_scaling_normalization(scores, method=normalization)
    if scores_norm is None or len(scores_norm) == 0:
        scores_norm = scores

    if percent_fmt:
        scores_plot = [float(s) * 100.0 for s in scores_norm]
    else:
        scores_plot = [float(s) for s in scores_norm]

    plot_df = pd.DataFrame({
        'output': outputs,
        'score_pct' if percent_fmt else 'score': scores_plot
    })

    if plt is None:
        raise ImportError("matplotlib is required for plotting. Please install 'matplotlib'.")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4 + 0.3 * k))
        created_fig = True
    else:
        fig = ax.figure

    labels = outputs
    values = scores_plot

    ax.barh(labels, values, color='#4e79a7')
    ax.invert_yaxis()  # Top item at top
    if percent_fmt:
        ax.set_xlim(0, 100)
        ax.set_xlabel('Score (%)')
    else:
        ax.set_xlabel('Normalized score')

    for i, v in enumerate(values):
        ax.text(v + (1 if percent_fmt else 0.01), i + 0.1, f"{v:.1f}%" if percent_fmt else f"{v:.3f}", fontsize=9)

    ax.set_title(title or f"{collection_name} top {k}")
    plt.tight_layout()

    return (fig if created_fig else None), ax, plot_df


def plot_selected_collection_top5_plotly(
    df_one_row,
    normalization=None,
    top_n=5,
    percent_fmt=True,
    title=None,
    color=None
):
    """
    Streamlit-friendly Plotly version of the 'selected collection' top-K bar chart.

    This mirrors plot_selected_collection_top5_bar but returns a Plotly figure that
    blends well with Streamlit's default theme. Pass Streamlit's theme color for
    perfect integration: color=st.get_option('theme.primaryColor').

    Args:
        df_one_row (pd.DataFrame): DataFrame containing exactly one row.
        normalization (str|None): One of {'softmax','minmax','l1','l2','zscore','sigmoid'} or None.
        top_n (int): Number of items to plot (defaults to 5).
        percent_fmt (bool): Show values as percentages when True.
        title (str|None): Optional plot title.
        color (str|None): Hex color for bars. Defaults to a Streamlit-like blue.

    Returns:
        (fig, plot_df): Plotly Figure and tidy DataFrame used for the chart.
    """
    try:
        import plotly.express as px
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])  # noqa: S603,S607
        import plotly.express as px  # type: ignore  # noqa: E401

    if not isinstance(df_one_row, pd.DataFrame) or len(df_one_row) == 0:
        raise ValueError("df_one_row must be a DataFrame with at least one row")

    row = df_one_row.iloc[0]
    if 'selected_collection' not in df_one_row.columns:
        raise ValueError("Missing 'selected_collection' column in DataFrame")
    collection_name = row['selected_collection']
    if not isinstance(collection_name, str) or not collection_name:
        raise ValueError("'selected_collection' must be a non-empty string")

    out_col, score_col = resolve_topn_columns_for_collection(df_one_row, collection_name)
    if out_col is None or score_col is None:
        raise ValueError(f"Could not resolve top-N columns for collection '{collection_name}'")

    outputs = row[out_col]
    scores = row[score_col]
    if isinstance(outputs, str):
        try:
            outputs = ast.literal_eval(outputs)
        except Exception:
            outputs = [outputs]
    if isinstance(scores, str):
        try:
            scores = ast.literal_eval(scores)
        except Exception:
            scores = [float(scores)] if scores not in (None, '') else []

    outputs = list(outputs) if outputs is not None else []
    scores = list(scores) if scores is not None else []
    if len(outputs) == 0 or len(scores) == 0:
        raise ValueError(f"No outputs/scores found in columns '{out_col}'/'{score_col}'")

    k = min(top_n, len(outputs), len(scores))
    outputs = outputs[:k]
    scores = scores[:k]

    # Normalize scores
    scores_norm = apply_scaling_normalization(scores, method=normalization)
    if scores_norm is None or len(scores_norm) == 0:
        scores_norm = scores

    values = [float(s) * 100.0 for s in scores_norm] if percent_fmt else [float(s) for s in scores_norm]
    value_col = 'score_pct' if percent_fmt else 'score'
    plot_df = pd.DataFrame({'output': outputs, value_col: values})

    if color is None:
        # Streamlit default primaryColor-like blue
        color = '#2c7be5'

    fig = px.bar(
        plot_df,
        x=value_col,
        y='output',
        orientation='h',
        text=value_col,
    )
    fig.update_traces(
        marker_color=color,
        texttemplate='%{text:.1f}%' if percent_fmt else '%{text:.3f}',
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>' + ('Score: %{x:.1f}%' if percent_fmt else 'Score: %{x:.3f}') + '<extra></extra>'
    )
    fig.update_layout(
        template='plotly_white',
        title=title or f"{collection_name} top {k}",
        margin=dict(t=50, r=16, b=24, l=16),
        xaxis_title='Score (%)' if percent_fmt else 'Normalized score',
        yaxis_title='',
        showlegend=False,
        bargap=0.25,
    )
    # Keep top item at top
    fig.update_yaxes(autorange='reversed')

    return fig, plot_df

def select_best_output_across_collections(
    df,
    collections,
    criterion='max_top1_margin',
    normalization=None,
    output_col_name='selected_output',
    collection_col_name='selected_collection'
):
    """
    Select a single best output per row from multiple collections' top-N results.

    Args:
        df (pd.DataFrame): DataFrame returned by batch_process_similarity (already augmented).
        collections (list[dict]): Same list used for batch processing. Only 'name' is required here.
        criterion (str): One of {'max_top1_margin','max','mean','total'}.
            Aliases: 'marginal_diff' -> 'max_top1_margin'.
            - 'max_top1_margin': For each collection, optionally normalize its scores, compute
              (score[0] - score[1]) if available, otherwise score[0]. Pick the collection with
              the largest margin; output is its top-1 output.
            - 'max': Union all ids across collections. Optionally normalize each collection's
              scores. For each id take the maximum score across all occurrences. Pick id with
              highest max; the collection reported is the one where this id achieved that max.
            - 'mean': Union all ids. Optionally normalize each collection's scores. For each id,
              compute the mean score across its occurrences. Report the collection where the id
              has the highest single score.
            - 'total': Union all ids. Optionally normalize; sum scores per id. Report the
              collection where the id has the highest single score.
        normalization (str|None): If provided, apply to each collection's score list independently.
            Options are supported by normalize_scores: 'minmax','zscore','l1','l2','sigmoid'.
        output_col_name (str): Name of the output column to add with the selected id/label.
        collection_col_name (str): Name of the output column to add with the selected collection name.

    Returns:
        pd.DataFrame: Copy of df with two additional columns: selected output and collection used.
    """
    df_out = df.copy()

    # Prepare column name resolver that infers top-N from df columns for each collection
    topn_regex_cache = {}
    def col_names(name):
        # Cache lookup
        if name in topn_regex_cache:
            return topn_regex_cache[name]
        pattern_output_prefix = f"{name}_top"
        pattern_output_suffix = "_output"
        matches = []
        for col in df_out.columns:
            if col.startswith(pattern_output_prefix) and col.endswith(pattern_output_suffix):
                # extract number between 'top' and '_output'
                try:
                    middle = col[len(pattern_output_prefix):-len(pattern_output_suffix)]
                    n_val = int(middle)
                    matches.append((n_val, col))
                except Exception:
                    continue
        if not matches:
            # No columns found for this collection
            topn_regex_cache[name] = (None, None, None)
            return topn_regex_cache[name]
        # Choose the largest topN if multiple
        matches.sort(key=lambda x: x[0], reverse=True)
        chosen_n, out_col = matches[0]
        score_col = f"{name}_top{chosen_n}_score"
        doc_col = f"{name}_top{chosen_n}_document"
        # Validate existence
        if score_col not in df_out.columns or doc_col not in df_out.columns:
            # If any missing, mark as unavailable
            topn_regex_cache[name] = (None, None, None)
        else:
            topn_regex_cache[name] = (out_col, score_col, doc_col)
        return topn_regex_cache[name]

    collection_names = [c.get('name', c.get('output_col', 'collection')) for c in collections]

    selected_outputs = []
    selected_collections = []

    # Back-compat alias
    if criterion == 'marginal_diff':
        criterion = 'max_top1_margin'

    for _, row in df_out.iterrows():
        if criterion == 'max_top1_margin':
            best_name = None
            best_margin = -np.inf
            best_output = None
            for name in collection_names:
                out_col, score_col, _doc_col = col_names(name)
                outputs = row[out_col] if out_col in row else None
                scores = row[score_col] if score_col in row else None
                if not outputs or not scores:
                    continue
                scores_arr = np.array(scores, dtype=float)
                if normalization is not None:
                    scores_arr = normalize_scores(scores_arr, method=normalization)
                if scores_arr.size == 0:
                    continue
                if scores_arr.size > 1:
                    margin = float(scores_arr[0] - scores_arr[1])
                else:
                    margin = float(scores_arr[0])
                if margin > best_margin:
                    best_margin = margin
                    best_name = name
                    best_output = outputs[0]
            selected_outputs.append(best_output)
            selected_collections.append(best_name)
            continue

        # For union-based criteria, build per-id aggregates across all collections
        id_to_max = {}
        id_to_sum = {}
        id_to_count = {}
        id_to_best_collection_for_max = {}

        for name in collection_names:
            out_col, score_col, _doc_col = col_names(name)
            outputs = row[out_col] if out_col in row else None
            scores = row[score_col] if score_col in row else None
            if not outputs or not scores:
                continue
            scores_arr = np.array(scores, dtype=float)
            if normalization is not None:
                scores_arr = normalize_scores(scores_arr, method=normalization)
            for id_val, s in zip(outputs, scores_arr):
                # Update max
                prev = id_to_max.get(id_val, -np.inf)
                if s > prev:
                    id_to_max[id_val] = float(s)
                    id_to_best_collection_for_max[id_val] = name
                # Update sum and count
                id_to_sum[id_val] = id_to_sum.get(id_val, 0.0) + float(s)
                id_to_count[id_val] = id_to_count.get(id_val, 0) + 1

        if not id_to_max:
            selected_outputs.append(None)
            selected_collections.append(None)
            continue

        if criterion == 'max':
            best_id = max(id_to_max, key=lambda k: id_to_max[k])
            best_collection = id_to_best_collection_for_max.get(best_id)
        elif criterion == 'mean':
            means = {k: (id_to_sum[k] / id_to_count[k]) for k in id_to_sum}
            best_id = max(means, key=lambda k: means[k])
            best_collection = id_to_best_collection_for_max.get(best_id)
        elif criterion == 'total':
            best_id = max(id_to_sum, key=lambda k: id_to_sum[k])
            best_collection = id_to_best_collection_for_max.get(best_id)
        else:
            raise ValueError("Unknown criterion. Use one of: 'max_top1_margin','max','mean','total'")

        selected_outputs.append(best_id)
        selected_collections.append(best_collection)

    df_out[output_col_name] = selected_outputs
    df_out[collection_col_name] = selected_collections
    return df_out


def apply_scaling_normalization(data_list, method='l1'):
    """
    Applies various scaling or normalization techniques to a list of numbers.

    Args:
        data_list (list or np.ndarray): The list of numbers to process.
        method (str): The technique to apply. Options are:
                      'l1' (L1 Normalization - default)
                      'l2' (L2 Normalization)
                      'minmax' (Min-Max Scaling to [0, 1] range)
                      'zscore' (Z-score Standardization)
                      'sigmoid' (Sigmoid Activation)
                      'softmax' (Softmax Activation)

    Returns:
        list: The processed list of numbers.
              Returns the original list if it's None, empty, or if an invalid method is chosen
              and cannot be processed, or if the data is unsuitable for the method (e.g., all same values for minmax).
    """
    if data_list is None or not hasattr(data_list, '__len__') or len(data_list) == 0:
        return data_list

    # Convert to numpy array and ensure it's numeric and 1D for element-wise operations
    # For sklearn normalize, it expects 2D, so we reshape.
    try:
        arr_1d = np.array(data_list, dtype=float)
        if arr_1d.ndim > 1: # Ensure it's a flat list of numbers
            print(f"Warning: data_list for method '{method}' has more than 1 dimension. Flattening.")
            arr_1d = arr_1d.flatten()
        if arr_1d.size == 0: # After potential flattening
             return []
    except ValueError:
        print(f"Warning: Could not convert data_list to numeric array for method '{method}'. Returning original.")
        return data_list

    # Replace nan with 0 and inf with large finite numbers
    arr_1d = np.nan_to_num(arr_1d)

    arr_2d = arr_1d.reshape(1, -1) # For sklearn functions that expect 2D input

    if method == 'l1':
        normalized_arr = normalize(arr_2d, norm='l1', axis=1)
        return normalized_arr[0].tolist()
    elif method == 'l2':
        normalized_arr = normalize(arr_2d, norm='l2', axis=1)
        return normalized_arr[0].tolist()
    elif method == 'minmax':
        # Min-Max requires distinct min and max values.
        if np.ptp(arr_1d) == 0: # Peak-to-peak is max - min
            # Handle case where all values are the same (e.g., return all 0s, 0.5s, or original)
            # Returning original or a list of 0s might be common choices.
            # Here, let's return a list of 0s if min=max, assuming 0-1 scaling intent.
            # Or, you could return arr_1d.tolist() if you prefer them to stay as is.
            return [0.0] * len(arr_1d) # or arr_1d.tolist()
        scaler = MinMaxScaler()
        scaled_arr = scaler.fit_transform(arr_2d.T).T # Transpose for fit_transform, then transpose back
        return scaled_arr[0].tolist()
    elif method == 'zscore':
        if np.std(arr_1d) == 0:
            # Handle case with zero standard deviation (all values are the same)
            # Return a list of 0s (as their Z-score would be 0 relative to themselves)
            return [0.0] * len(arr_1d) # or arr_1d.tolist()
        scaler = StandardScaler()
        scaled_arr = scaler.fit_transform(arr_2d.T).T # Transpose for fit_transform, then transpose back
        return scaled_arr[0].tolist()
    elif method == 'sigmoid':
        # arr_1d is already 1D float numpy array
        sig_arr = 1 / (1 + np.exp(-arr_1d))
        return sig_arr.tolist()
    elif method == 'softmax':
        # arr_1d is already 1D float numpy array
        # Subtract max for numerical stability before exponentiating
        exps = np.exp(arr_1d - np.max(arr_1d))
        softmax_arr = exps / np.sum(exps)
        return softmax_arr.tolist()
    else:
        print(f"Warning: Unknown method '{method}'. Returning original list.")
        return data_list


def apply_column_normalization(df, columns, method='l1'):
    """
    Applies a specified normalization method to columns in a DataFrame
    and saves the result to a new column ending in '_l1norm'.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        columns (list of str): The names of the columns to normalize.
        method (str): The normalization method to use ('l1', 'l2', 'min_max', 
                      'zscore', 'softmax', 'sigmoid').
    """
    
    def l1_norm(series):
        """Calculates the L1 norm for a series."""
        norm = np.linalg.norm(series, ord=1)
        return series / norm if norm > 0 else series

    def l2_norm(series):
        """Calculates the L2 norm for a series."""
        norm = np.linalg.norm(series, ord=2)
        return series / norm if norm > 0 else series

    def min_max_scaler(series):
        """Scales a series to the range [0, 1]."""
        min_val = series.min()
        max_val = series.max()
        range_val = max_val - min_val
        if range_val > 0:
            return (series - min_val) / range_val
        else:
            return pd.Series(0.0, index=series.index, name=series.name)

    def z_score(series):
        """Applies Z-score normalization."""
        mean = series.mean()
        std = series.std()
        if std > 0:
            return (series - mean) / std
        else:
            return pd.Series(0.0, index=series.index, name=series.name)

    def softmax(series):
        """Applies softmax function for probabilistic scaling."""
        exps = np.exp(series - np.max(series))
        return exps / np.sum(exps)

    def sigmoid(series):
        """Applies the sigmoid function to scale to (0, 1)."""
        return 1 / (1 + np.exp(-series))

    normalizers = {
        'l1': l1_norm,
        'l2': l2_norm,
        'minmax': min_max_scaler,
        'zscore': z_score,
        'softmax': softmax,
        'sigmoid': sigmoid
    }

    normalizer = normalizers.get(method)
    if not normalizer:
        raise ValueError(f"Unknown normalization method: {method}. Available methods are: {list(normalizers.keys())}")

    for col in columns:
        df[f'{col}_norm'] = normalizer(df[col])


if __name__ == "__main__":
    pass

