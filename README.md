# RAG Query System with AstraDB and NVIDIA Reranking

This application provides a Retrieval-Augmented Generation (RAG) system that uses AstraDB as the vector database and NVIDIA's reranking technology to enhance the quality of retrieved documents.

[Application URL](https://brian-ogrady-rag-with-reranking-streamlit-app-7pycae.streamlit.app/)
## Overview

The RAG Query System combines:
- AstraDB for vector storage and initial similarity search
- NVIDIA AI reranking to improve document relevance
- Language models (OpenAI) for generating answers
- A Streamlit interface for user interaction

## Features

- **Two-Stage Retrieval**: Initial vector search followed by high-quality reranking
- **Customizable Retrieval Parameters**: Configure both the vector search and reranking stages
- **Document Visualization**: See which documents were used to generate answers
- **Conversation History**: Keep track of your Q&A session
- **Configurable Language Models**: Choose between different LLM options
- **Custom Prompting**: Modify the prompt template for different use cases

## Requirements

- Python 3.8+
- AstraDB account with vector database
- OpenAI API key
- NVIDIA AI reranking endpoint
- Streamlit account (for deployment)

## Setup

### Local Development

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/rag-query-system.git
   cd rag-query-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your credentials:
   ```
   OPENAI_API_KEY=your_openai_key_here
   ASTRA_DB_APPLICATION_TOKEN=your_astra_token_here
   ASTRA_DB_API_ENDPOINT=your_astra_endpoint_here
   RERANKER_URL=your_nvidia_reranker_url_here
   ```

4. Run the app:
   ```bash
   ./run_app.sh
   ```
   or
   ```bash
   streamlit run app.py
   ```

### Streamlit Cloud Deployment

1. Push your code to GitHub (make sure not to include your `.env` file)

2. Deploy on Streamlit Community Cloud by connecting to your repository

3. Set the required secrets in the Streamlit dashboard:
   ```toml
   [secrets]
   OPENAI_API_KEY = "your-openai-key-here"
   ASTRA_DB_APPLICATION_TOKEN = "your-astra-token-here"
   ASTRA_DB_API_ENDPOINT = "your-astra-endpoint-here"
   RERANKER_URL = "your-nvidia-reranker-url-here"
   ```

## Architecture

### RAGRetriever Class

The core component is the `RAGRetriever` class which implements:

- A vector database connection to AstraDB
- Integration with NVIDIA's reranking model
- LangChain's `ContextualCompressionRetriever` for combining both approaches
- Methods for querying, document retrieval, and customization

### Streamlit Interface

The Streamlit app provides:
- Configuration controls for all aspects of the system
- A chat-like interface for asking questions
- Document inspection for understanding retrieval
- Debug information to help troubleshoot

## Tuning Guide

For optimal results:

1. **Vector Database Parameters**:
   - `similarity_top_k`: The number of documents to retrieve from the vector database (default: 20)
   - Increasing this captures more potentially relevant documents but may include less relevant ones

2. **Reranking Parameters**:
   - `rerank_top_n`: The number of documents to keep after reranking (default: 3)
   - Adjust based on context window limitations and desired specificity

3. **LLM Settings**:
   - Model selection and temperature affect answer style and accuracy
   - Lower temperature (0.0-0.3) for factual responses
   - Higher temperature (0.5-0.7) for more creative responses

## Implementation Details

This system uses:
- LangChain for orchestration
- AstraDB's Vectorize feature for server-side embeddings
- NVIDIA AI's reranking for improved relevance
- OpenAI's models for generating answers
- Streamlit for the user interface

## Contact

For questions, issues or suggestions, please [open an issue](https://github.com/your-username/rag-query-system/issues) on GitHub.