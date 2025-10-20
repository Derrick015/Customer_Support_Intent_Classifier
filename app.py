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
        # primary_color = st.get_option('theme.primaryColor') or '#2c7be5'
        # fig, plot_df = plot_selected_collection_top5_plotly(
        #     df_one_row=df_selected,
        #     normalization='softmax',  # Use softmax so percentages add up to 100%
        #     top_n=5,
        #     percent_fmt=True,
        #     title=" ",  # Remove the chart title (use space to override default)
        #     color=primary_color
        # )
        
        # # Remove any title from the figure
        # fig.update_layout(title="")
        
        # return final_prediction, fig, df_selected
        return final_prediction
    
    except Exception as e:
        raise Exception(f"Classification failed: {str(e)}")

# Main app
def main():
    st.title("🎧📦 Customer Support Intent Classifier")
    
    # Message section
    st.markdown("""
    <div style="
        padding: 16px 20px; 
        background: linear-gradient(to right, #f0f4ff, #ffffff); 
        border-radius: 12px; 
        border-left: 4px solid #667eea;
        margin-bottom: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    ">
        <p style="margin: 0; font-size: 15px; line-height: 1.6; color: #334155;">
            <strong style="color: #4f46e5;">In this demo,</strong> we classify customer support 
            queries into one of these intents; <strong>Cancel Order</strong>, 
            <strong>Refund Request</strong>, <strong>Technical Issue</strong>, and 
            <strong>Track Order</strong>, to automatically route requests, trigger the right 
            workflows, and shorten resolution times
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar — styled and informative
    st.sidebar.markdown("""
    <div style="
        padding: 20px; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        border-radius: 16px; 
        margin-bottom: 18px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
    ">
        <h2 style="color: white; margin: 0; text-align: center; font-weight: 700; letter-spacing: .3px;">
            About This App
        </h2>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div style="
        padding: 16px; 
        background: linear-gradient(to right, #f8fafc, #ffffff); 
        border-radius: 12px; 
        border-left: 4px solid #667eea;
        margin-bottom: 18px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    ">
        <p style="margin: 0; font-size: 14px; line-height: 1.7; color: #334155;">
            The underlying model used by this app is the <strong style="color: #4f46e5;">Poly Collections Classifier</strong>.  It uses
            embeddings to integrate <strong>multimodal inputs</strong> (text, images, audio, video) in a modular fashion
            for flexible, scalable and accurate classification
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div style="margin-bottom: 18px;">
        <h4 style="color: #0f172a; margin: 0 0 10px 0;">Other Use Cases</h4>
        <div style="display: flex; flex-direction: column; gap: 10px;">
            <div style="padding: 12px; background: linear-gradient(135deg, #7f91ee 0%, #8a69c2 100%); border-radius: 10px; border-left: 3px solid #7f91ee;">
                <strong style="color: #ffffff;">Content Moderation</strong>
            </div>
            <div style="padding: 12px; background: linear-gradient(135deg, #99a6f2 0%, #a480c8 100%); border-radius: 10px; border-left: 3px solid #99a6f2;">
                <strong style="color: #ffffff;">Sentiment Analysis</strong>
            </div>
            <div style="padding: 12px; background: linear-gradient(135deg, #b3bbf6 0%, #bd96cf 100%); border-radius: 10px; border-left: 3px solid #b3bbf6;">
                <strong style="color: #ffffff;">Product Categorisation</strong>
            </div>
            <div style="padding: 12px; background: linear-gradient(135deg, #cdd0fa 0%, #d1acd6 100%); border-radius: 10px; border-left: 3px solid #cdd0fa;">
                <strong style="color: #ffffff;">Many Others</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    with st.sidebar.expander("⚙️ How It Works", expanded=False):
        st.markdown("""
        1. Type in your query based on the provided categories
        2. Click the "Classify" button
        3. The app will classify your query into one of the provided categories
        4. Visualise the confidence distribution of the classification
        """)

    # Display available output categories
    st.markdown("### Available Categories")
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
    
    # # Query input
    # st.markdown("### Enter Customer Query")
    
    # # Example queries
    # with st.expander("💡 Click to see example queries"):
    #     st.markdown("""
    #     - "help my order has still not arrived what is going on"
    #     - "I want to cancel my recent order"
    #     - "I need a refund for my purchase"
    #     - "The website is not loading properly"
    #     - "Where is my package?"
    #     - "Can you help me return this item?"
    #     """)
    
    query = st.text_area(
        "Type your query here:",
        placeholder="e.g., help my order has still not arrived what is going on",
        height=100
    )
    
    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        st.markdown("""
        <style>
        /* Strongest purple gradient for the Classify button */
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: #ffffff !important;
            border: 0 !important;
            border-radius: 8px !important;
            box-shadow: 0 6px 14px rgba(102, 126, 234, 0.35) !important;
            font-weight: 600 !important;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #7585ec 0%, #8459ae 100%) !important;
            box-shadow: 0 10px 22px rgba(118, 75, 162, 0.50) !important;
            transform: translateY(-1px);
        }
        div[data-testid="stButton"] > button[kind="primary"]:active {
            transform: translateY(1px);
        }
        </style>
        """, unsafe_allow_html=True)
        classify_button = st.button("🚀 Classify", type="primary", use_container_width=True)
    
    # Classify on button click
    if classify_button:
        if not query.strip():
            st.warning("⚠️ Please enter a query to classify.")
        else:
            with st.spinner("🔍 Analysing query..."):
                try:
                    prediction = classify_query(query, client, collections)
                    
                    st.markdown("---")
                    
                    # Category-specific styling
                    category_styles = {
                        'Cancel Order': {
                            'gradient': 'linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%)',
                            'icon': '📦',
                            'shadow': 'rgba(255, 107, 107, 0.4)',
                            'accent': '#ff6b6b'
                        },
                        'Refund Request': {
                            'gradient': 'linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%)',
                            'icon': '💰',
                            'shadow': 'rgba(78, 205, 196, 0.4)',
                            'accent': '#4ecdc4'
                        },
                        'Technical Issue': {
                            'gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                            'icon': '🔧',
                            'shadow': 'rgba(102, 126, 234, 0.4)',
                            'accent': '#667eea'
                        },
                        'Track Order': {
                            'gradient': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                            'icon': '📍',
                            'shadow': 'rgba(240, 147, 251, 0.4)',
                            'accent': '#f093fb'
                        }
                    }
                    
                    style = category_styles.get(prediction, category_styles['Technical Issue'])
                    
                    # Display animated prediction card
                    st.markdown(f"""
                    <style>
                    @keyframes slideInScale {{
                        0% {{
                            opacity: 0;
                            transform: translateY(-30px) scale(0.9);
                        }}
                        100% {{
                            opacity: 1;
                            transform: translateY(0) scale(1);
                        }}
                    }}
                    
                    @keyframes pulse {{
                        0%, 100% {{
                            transform: scale(1);
                        }}
                        50% {{
                            transform: scale(1.05);
                        }}
                    }}
                    
                    @keyframes shimmer {{
                        0% {{
                            background-position: -1000px 0;
                        }}
                        100% {{
                            background-position: 1000px 0;
                        }}
                    }}
                    
                    .prediction-card {{
                        background: {style['gradient']};
                        padding: 40px;
                        border-radius: 20px;
                        box-shadow: 0 20px 60px {style['shadow']};
                        animation: slideInScale 0.6s ease-out;
                        position: relative;
                        overflow: hidden;
                        margin: 30px 0;
                    }}
                    
                    .prediction-card::before {{
                        content: '';
                        position: absolute;
                        top: -50%;
                        left: -50%;
                        width: 200%;
                        height: 200%;
                        background: linear-gradient(
                            45deg,
                            transparent,
                            rgba(255, 255, 255, 0.1),
                            transparent
                        );
                        animation: shimmer 3s infinite;
                    }}
                    
                    .prediction-content {{
                        position: relative;
                        z-index: 1;
                        text-align: center;
                    }}
                    
                    .prediction-icon {{
                        font-size: 80px;
                        animation: pulse 2s ease-in-out infinite;
                        display: block;
                        margin-bottom: 20px;
                    }}
                    
                    .prediction-label {{
                        color: rgba(255, 255, 255, 0.9);
                        font-size: 18px;
                        font-weight: 500;
                        letter-spacing: 2px;
                        text-transform: uppercase;
                        margin-bottom: 12px;
                    }}
                    
                    .prediction-text {{
                        color: #ffffff;
                        font-size: 42px;
                        font-weight: 800;
                        margin: 0;
                        text-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
                        letter-spacing: 1px;
                    }}
                    
                    .confidence-badge {{
                        display: inline-block;
                        background: rgba(255, 255, 255, 0.25);
                        backdrop-filter: blur(10px);
                        padding: 12px 28px;
                        border-radius: 50px;
                        color: #ffffff;
                        font-weight: 600;
                        font-size: 16px;
                        margin-top: 20px;
                        border: 2px solid rgba(255, 255, 255, 0.3);
                    }}
                    </style>
                    
                    <div class="prediction-card">
                        <div class="prediction-content">
                            <span class="prediction-icon">{style['icon']}</span>
                            <div class="prediction-label">Predicted Category</div>
                            <h1 class="prediction-text">{prediction}</h1>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # st.markdown("### Confidence Distribution")
                    # st.markdown("The chart below shows the confidence scores across different categories:")
                    # 
                    # # Display visualization
                    # st.plotly_chart(fig, use_container_width=True)
#                             <div class="confidence-badge">✨ Classification Complete</div>
                
                except Exception as e:
                    st.error(f"❌ {str(e)}")
                    st.info("Please try again or contact support if the issue persists.")

if __name__ == "__main__":
    main()

