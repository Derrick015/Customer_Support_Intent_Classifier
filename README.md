# Customer Support Intent Classifier

A sophisticated customer support intent classification system powered by the novel PolyCollections Classification Algorithm I developed, which leverages multimodal embeddings and vector similarity to automatically route customer queries into predefined categories. Built with Streamlit and powered by Google’s Gemini AI embeddings.

## Features

- **Multi-Collection Classification**: Uses multiple vector collections for robust intent detection
- **Real-time Classification**: Instant classification of customer support queries
- **Beautiful UI**: Modern, responsive interface with animated results
- **Docker Support**: Easy deployment with containerization
- **Google Cloud Integration**: Leverages Google Gemini AI for embeddings

## Supported Categories

The system classifies customer queries into four main categories:

- **📦 Cancel Order** - Requests to cancel existing orders
- **💰 Refund Request** - Requests for refunds or returns
- **🔧 Technical Issue** - Technical problems or website issues
- **📍 Track Order** - Inquiries about order status and tracking

## Quick Start

### Prerequisites

- Python 3.11+
- Google Cloud Project with Gemini AI API access
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Derrick015/Customer_Support_Intent_Classifier
   cd Customer_Support_Intent_Classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google Cloud credentials**
   ```bash
   # Set your Google Cloud project
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   
   # Authenticate with Google Cloud
   gcloud auth application-default login
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

The application will be available at `http://localhost:8501`

## Docker Deployment

### Build and run with Docker

```bash
# Build the Docker image


docker build -t customer-support-intent-classifier .

# Run the container
docker run -p 8080:8080 customer-support-intent-classifier
```

### Deploy to Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/customer-support-intent-classifier

# Deploy to Cloud Run
gcloud run deploy customer-support-intent-classifier \
  --image gcr.io/PROJECT_ID/customer-support-intent-classifier \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Architecture

### Core Components

1. **Vector Collections**: 
   - `intent_meaning_collection`: Contains intent-based embeddings
   - `sample_avg_embeddings_collection`: Contains sample-based embeddings

2. **Classification Pipeline**:
   - Query embedding generation using Gemini AI
   - Multi-collection similarity computation
   - Best output selection across collections
   - Confidence scoring and visualization

3. **Web Interface**:
   - Streamlit-based responsive UI
   - Real-time classification
   - Animated result display

### Key Technologies

- **Frontend**: Streamlit
- **ML/AI**: Google Gemini AI, ChromaDB
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Deployment**: Docker, Google Cloud Run
- **Vector Storage**: ChromaDB with persistent storage

## How It Works

1. **Input Processing**: User enters a customer support query
2. **Embedding Generation**: Query is converted to vector embeddings using Gemini AI
3. **Similarity Search**: Embeddings are compared against multiple vector collections
4. **Collection Selection**: Best matching collection is identified using various criteria
5. **Output Classification**: Final category is determined and displayed with confidence

### Classification Criteria

The system supports multiple criteria for selecting the best output:

- **`max_top1_margin`**: Uses the margin between top-1 and top-2 scores
- **`max`**: Selects the output with the highest maximum score across collections
- **`mean`**: Uses mean scores across collections
- **`total`**: Uses total aggregated scores

## Configuration

### Environment Variables

```bash
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1

# Streamlit Configuration (for Docker)
STREAMLIT_SERVER_PORT=8080
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```


## Project Structure

```
Poly_Collections_Classifier/
├── app.py                          # Main Streamlit application
├── src/
│   └── utils.py                    # Core classification utilities
├── emb_collection/                 # ChromaDB vector collections
│   ├── chroma.sqlite3             # Database file
│   └── [collection-uuid]/         # Collection data directories
├── dataset/
│   └── output_df_exploded.xlsx    # Training data
├── tests/
│   └── test_app.py                # Unit tests
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project configuration
├── Dockerfile                     # Container configuration
└── README.md                      # This file
```

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## Performance

- **Classification Speed**: ~1-2 seconds per query
- **Accuracy**: High accuracy on customer support intent classification
- **Scalability**: Supports batch processing for multiple queries
- **Memory Usage**: Optimised for efficient vector operations

## Security

- **Authentication**: Uses Google Cloud Application Default Credentials
- **Data Privacy**: No customer data is stored permanently
- **API Security**: Secure communication with Google Cloud services

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Derrick Ofori**
- Email: mrderrick.of@gmail.com
- GitHub: [@Derrick015](https://github.com/Derrick015)

## Acknowledgments

- Google Gemini AI for powerful embedding capabilities
- Streamlit team for the excellent web framework
- ChromaDB for efficient vector storage and retrieval
- The open-source community for various supporting libraries

---

