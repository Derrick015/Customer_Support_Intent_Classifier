import types
import sys
import pandas as pd


# Pre-stub heavy/optional third-party imports used at app import time
# so that importing app.py does not require those dependencies to be installed.

# Stub chromadb.PersistentClient only (do NOT stub top-level google module to avoid breaking streamlit)
if "chromadb" not in sys.modules:
    chroma_stub = types.ModuleType("chromadb")
    class _PersistentClientStub:  # minimal placeholder; tests will monkeypatch as needed
        def __init__(self, *args, **kwargs):
            pass
    chroma_stub.PersistentClient = _PersistentClientStub
    sys.modules["chromadb"] = chroma_stub

# Stub src.utils to avoid importing heavy Google Cloud clients during app import
if "src" not in sys.modules:
    sys.modules["src"] = types.ModuleType("src")

utils_stub = types.ModuleType("src.utils")

def _stub_process_similarity_single_query(**kwargs):
    return pd.DataFrame()

def _stub_select_best_output_across_collections(df, collections, criterion, normalization):
    return pd.DataFrame(
        {
            "selected_output": ["cancel_order"],
            "selected_collection": ["intent_meaning_collection"],
            "intent_meaning_collection_top5_output": [["cancel_order", "refund_request"]],
        }
    )

def _stub_plot_selected_collection_top5_plotly(**kwargs):
    return None, pd.DataFrame()

utils_stub.process_similarity_single_query = _stub_process_similarity_single_query
utils_stub.select_best_output_across_collections = _stub_select_best_output_across_collections
utils_stub.plot_selected_collection_top5_plotly = _stub_plot_selected_collection_top5_plotly

sys.modules["src.utils"] = utils_stub

# Provide a minimal stub for google.genai.types.EmbedContentConfig without touching top-level 'google'
if "google.genai" not in sys.modules:
    genai_stub = types.ModuleType("google.genai")
    sys.modules["google.genai"] = genai_stub
else:
    genai_stub = sys.modules["google.genai"]

if "google.genai.types" not in sys.modules:
    types_stub = types.ModuleType("google.genai.types")
    sys.modules["google.genai.types"] = types_stub
else:
    types_stub = sys.modules["google.genai.types"]

class _EmbedContentConfigStub:
    def __init__(self, *args, **kwargs):
        pass

types_stub.EmbedContentConfig = _EmbedContentConfigStub
setattr(genai_stub, "types", types_stub)


import app as app_module


def test_classify_query_returns_clean_label(monkeypatch):
    # Ensure caches are cleared between tests (Streamlit cache wrapper provides clear())
    if hasattr(app_module.load_collections, "clear"):
        app_module.load_collections.clear()

    # Stub embedder to avoid external calls
    monkeypatch.setattr(app_module, "embed_single_query", lambda text, client: [[0.1, 0.2, 0.3]])

    # Stub similarity processing (return value unused by our stubbed selector)
    monkeypatch.setattr(app_module, "process_similarity_single_query", lambda **kwargs: pd.DataFrame())

    # Return a one-row DataFrame mimicking selector output
    def fake_select(df, collections, criterion, normalization):
        return pd.DataFrame(
            {
                "selected_output": ["cancel_order"],
                "selected_collection": ["intent_meaning_collection"],
                "intent_meaning_collection_top5_output": [["cancel_order", "refund_request"]],
            }
        )

    monkeypatch.setattr(app_module, "select_best_output_across_collections", fake_select)

    # Dummy client object (not used by stubbed embedder logic beyond type presence)
    fake_client = object()

    collections = [
        {"name": "intent_meaning_collection", "df": pd.DataFrame(), "output_col": "output", "embedding_col": "embedding", "document_col": "document"},
        {"name": "sample_avg_embeddings_collection", "df": pd.DataFrame(), "output_col": "output", "embedding_col": "embedding", "document_col": "document"},
    ]

    result = app_module.classify_query("I want to cancel my order", fake_client, collections)
    assert result == "Cancel Order"


def test_embed_single_query_invokes_client_and_returns_values():
    class _Embedding:
        def __init__(self, values):
            self.values = values

    class _Response:
        def __init__(self):
            self.embeddings = [_Embedding([0.5, 0.6])]

    class _Models:
        def embed_content(self, model, contents, config):
            # Validate key inputs roughly
            assert model == "gemini-embedding-001"
            assert isinstance(contents, str)
            return _Response()

    class _Client:
        def __init__(self):
            self.models = _Models()

    values = app_module.embed_single_query("hello", _Client())
    assert values == [[0.5, 0.6]]


def test_load_collections_happy_path(monkeypatch):
    if hasattr(app_module.load_collections, "clear"):
        app_module.load_collections.clear()

    # Pretend the Chroma path exists
    monkeypatch.setattr(app_module.os.path, "exists", lambda p: True)

    class _FakeCollection:
        def __init__(self, kind):
            self.kind = kind

        def get(self, include):
            # Minimal valid structure
            return {
                "ids": ["1"],
                "documents": [f"{self.kind}_doc"],
                "metadatas": [{"output": "cancel_order"}],
                "embeddings": [[0.1, 0.2, 0.3]],
            }

    class _FakeClient:
        def get_collection(self, name):
            return _FakeCollection(name)

    # Replace Chroma PersistentClient constructor
    monkeypatch.setattr(app_module, "PersistentClient", lambda path: _FakeClient())

    load_func = getattr(app_module.load_collections, "__wrapped__", app_module.load_collections)
    df_intent, df_sample = load_func()

    assert isinstance(df_intent, pd.DataFrame)
    assert isinstance(df_sample, pd.DataFrame)
    for df in (df_intent, df_sample):
        assert set(["embedding", "document", "output"]).issubset(df.columns)
        assert len(df) == 1


def test_load_collections_missing_path_reports_error(monkeypatch):
    if hasattr(app_module.load_collections, "clear"):
        app_module.load_collections.clear()

    # Force path check to fail
    monkeypatch.setattr(app_module.os.path, "exists", lambda p: False)

    # Capture error messages
    errors = []

    def _err(msg):
        errors.append(msg)

    monkeypatch.setattr(app_module.st, "error", _err)

    load_func = getattr(app_module.load_collections, "__wrapped__", app_module.load_collections)
    df_intent, df_sample = load_func()
    assert df_intent is None and df_sample is None
    assert any("ChromaDB path" in e for e in errors)


