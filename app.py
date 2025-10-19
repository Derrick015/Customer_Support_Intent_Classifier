import streamlit as st
import pandas as pd
from chromadb import PersistentClient
from google import genai
from google.genai.types import EmbedContentConfig
from src.utils import process_similarity_single_query, select_best_output_across_collections, plot_selected_collection_top5_plotly
import os

# Page configuration
st.set_page_config(
    page_title="Poly Collections Classifier",
    page_icon="🎯",
    layout="wide"
)

# Initialize Google GenAI client
@st.cache_resource
def get_genai_client():
    """Initialize and cache the Google GenAI client."""
    project = "deron-innovations"
    location = "us-central1"
    client = genai.Client(vertexai=True, project=project, location=location)
    return client

@st.cache_resource
def load_collections():
    """Load ChromaDB collections and convert to DataFrames."""
    try:
        CHROMA_PATH = "emb_collection"
        
        if not os.path.exists(CHROMA_PATH):
            st.error(f"❌ ChromaDB path '{CHROMA_PATH}' not found. Please ensure collections are available.")
            return None, None
        
        client = PersistentClient(path=CHROMA_PATH)
        
        # Load intent_meaning_collection
        try:
            col_intent = client.get_collection("intent_meaning_collection")
            all_items_intent = col_intent.get(include=["documents", "metadatas", "embeddings"])
            
            df_intent_collection = pd.DataFrame({
                'id': all_items_intent['ids'],
                'embedding': list(all_items_intent['embeddings']),
                'document': all_items_intent['documents'],
                'metadatas': all_items_intent['metadatas']
            })
            df_intent_collection['output'] = df_intent_collection['metadatas'].apply(
                lambda x: x.get('output') if isinstance(x, dict) else None
            )
            df_intent_collection.drop(columns=['id', 'metadatas'], inplace=True)
        except Exception as e:
            st.error(f"❌ Failed to load intent_meaning_collection: {e}")
            return None, None
        
        # Load sample_avg_embeddings_collection
        try:
            col_sample = client.get_collection("sample_avg_embeddings_collection")
            all_items_sample = col_sample.get(include=["documents", "metadatas", "embeddings"])
            
            df_avg_embeddings_collection = pd.DataFrame({
                'id': all_items_sample['ids'],
                'embedding': list(all_items_sample['embeddings']),
                'document': all_items_sample['documents'],
                'metadatas': all_items_sample['metadatas']
            })
            df_avg_embeddings_collection['output'] = df_avg_embeddings_collection['metadatas'].apply(
                lambda x: x.get('output') if isinstance(x, dict) else None
            )
            df_avg_embeddings_collection.drop(columns=['id', 'metadatas'], inplace=True)
        except Exception as e:
            st.error(f"❌ Failed to load sample_avg_embeddings_collection: {e}")
            return None, None
        
        return df_intent_collection, df_avg_embeddings_collection
    
    except Exception as e:
        st.error(f"❌ Error loading collections: {e}")
        return None, None

def embed_single_query(input_text, client):
    """Embed a single query using Gemini API."""
    response = client.models.embed_content(
        model='gemini-embedding-001',
        contents=input_text,
        config=EmbedContentConfig(task_type='RETRIEVAL_QUERY'),
    )
    return [e.values for e in response.embeddings]

def classify_query(query, client, collections):
    """Classify a customer query and return the prediction with details."""
    try:
        # Embed the query
        query_emb = embed_single_query(query, client)
        
        # Process similarity
        single_result = process_similarity_single_query(
            query_embedding=query_emb,
            collections=collections,
            top_n=5
        )
        
        # Select best output
        df_selected = select_best_output_across_collections(
            df=single_result,
            collections=collections,
            criterion='total',
            normalization='l1'
        )
        
        # Map to clean labels
        output_mapping = {
            'cancel_order': 'Cancel Order',
            'refund_request': 'Refund Request',
            'technical_issue': 'Technical Issue',
            'track_order': 'Track Order'
        }
        
        df_selected['selected_output_clean'] = df_selected['selected_output'].map(output_mapping)
        final_prediction = df_selected['selected_output_clean'].iloc[0]
        
        # Map all collection output columns to clean labels for visualization
        selected_collection = df_selected['selected_collection'].iloc[0]
        output_col = f"{selected_collection}_top5_output"
        if output_col in df_selected.columns:
            # Apply mapping to the output list
            df_selected[output_col] = df_selected[output_col].apply(
                lambda outputs: [output_mapping.get(o, o) for o in outputs] if isinstance(outputs, list) else outputs
            )
        
        # Get confidence visualization
        primary_color = st.get_option('theme.primaryColor') or '#2c7be5'
        fig, plot_df = plot_selected_collection_top5_plotly(
            df_one_row=df_selected,
            normalization='softmax',  # Use softmax so percentages add up to 100%
            top_n=5,
            percent_fmt=True,
            title=" ",  # Remove the chart title (use space to override default)
            color=primary_color
        )
        
        # Remove any title from the figure
        fig.update_layout(title="")
        
        return final_prediction, fig, df_selected
    
    except Exception as e:
        raise Exception(f"Classification failed: {str(e)}")

# Main app
def main():
    st.title("🎯 PolyVector Customer Intent Classifier")
    st.markdown("""
    This demo showcases PolyVector's semantic classification capabilities for customer service queries.
    Enter a customer query to see the predicted intent category with confidence scores.
    """)
    
    # Display available output categories
    st.markdown("### Available Intent Categories")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("📦 **Cancel Order**")
    with col2:
        st.info("💰 **Refund Request**")
    with col3:
        st.info("🔧 **Technical Issue**")
    with col4:
        st.info("📍 **Track Order**")
    
    # st.markdown("---")
    
    # Load collections
    with st.spinner("Loading classification models..."):
        df_intent, df_sample = load_collections()
    
    if df_intent is None or df_sample is None:
        st.error("Failed to load collections. Please check the emb_collection directory.")
        return
    
    # Initialize client
    try:
        client = get_genai_client()
    except Exception as e:
        st.error(f"❌ Failed to initialize Google GenAI client: {e}")
        st.info("Please ensure your Google Cloud credentials are properly configured.")
        return
    
    # Prepare collections for processing
    collections = [
        {
            'name': 'intent_meaning_collection',
            'df': df_intent,
            'output_col': 'output',
            'embedding_col': 'embedding',
            'document_col': 'document'
        },
        {
            'name': 'sample_avg_embeddings_collection',
            'df': df_sample,
            'output_col': 'output',
            'embedding_col': 'embedding',
            'document_col': 'document'
        }
    ]
    
    # st.success(f"✅ Models loaded successfully! ({len(df_intent)} intent vectors, {len(df_sample)} sample vectors)")
    
    # Query input
    st.markdown("### Enter Customer Query")
    
    # Example queries
    with st.expander("💡 Click to see example queries"):
        st.markdown("""
        - "help my order has still not arrived what is going on"
        - "I want to cancel my recent order"
        - "I need a refund for my purchase"
        - "The website is not loading properly"
        - "Where is my package?"
        - "Can you help me return this item?"
        """)
    
    query = st.text_area(
        "Type your query here:",
        placeholder="e.g., help my order has still not arrived what is going on",
        height=100
    )
    
    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        classify_button = st.button("🚀 Classify", type="primary", use_container_width=True)
    
    # Classify on button click
    if classify_button:
        if not query.strip():
            st.warning("⚠️ Please enter a query to classify.")
        else:
            with st.spinner("🔍 Analyzing query..."):
                try:
                    prediction, fig, df_selected = classify_query(query, client, collections)
                    
                    st.markdown("---")
                    st.markdown("### 📊 Classification Results")
                    
                    # Display prediction
                    st.markdown(f"""
                    <div style="padding: 20px; background-color: #f0f9ff; border-radius: 10px; border-left: 5px solid #2c7be5;">
                        <h2 style="margin: 0; color: #1e40af;">Predicted Intent: {prediction}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### 📈 Confidence Distribution")
                    st.markdown("The chart below shows the confidence scores across different intent categories:")
                    
                    # Display visualization
                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ {str(e)}")
                    st.info("Please try again or contact support if the issue persists.")

if __name__ == "__main__":
    main()

