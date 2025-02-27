import streamlit as st
import os
import sys
from dotenv import load_dotenv
import uuid
from datetime import datetime

# Add path to RagRetriever
sys.path.append(os.path.expanduser("~/src/python"))
from src.python.RagRetriever import RAGRetriever

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="RAG Query System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Cache the RAG Retriever initialization
@st.cache_resource
def get_rag_retriever(collection_name, rerank_top_n=3, similarity_top_k=20):
    try:
        return RAGRetriever(
            collection_name=collection_name,
            rerank_top_n=rerank_top_n,
            similarity_top_k=similarity_top_k
        )
    except ValueError as e:
        st.error(f"Error initializing RAGRetriever: {e}")
        return None

# Sidebar for configuration
st.sidebar.title("RAG Configuration")

# Collection selection
collection_name = st.sidebar.text_input(
    "AstraDB Collection Name",
    value="pdfs"
)

# Model selection
model_name = st.sidebar.selectbox(
    "LLM Model",
    options=["gpt-4o", "gpt-3.5-turbo"],
    index=0
)

# Retrieval settings
similarity_top_k = st.sidebar.slider(
    "Initial vector search results",
    min_value=5,
    max_value=30,
    value=20
)

rerank_top_n = st.sidebar.slider(
    "Results after reranking",
    min_value=1,
    max_value=10,
    value=3
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.1
)

# Show retrieved documents option
show_documents = st.sidebar.checkbox("Show retrieved documents", value=True)

# Advanced settings expander
with st.sidebar.expander("Advanced Settings"):
    search_type = st.selectbox(
        "Search Type",
        options=["similarity", "mmr"],
        index=0
    )
    
    custom_prompt = st.text_area(
        "Custom Prompt Template",
        height=200,
        value="""You are an AI assistant providing helpful information based on the context provided.
        
Context:
{context}

Question:
{question}

Answer the question based on the context provided. If you cannot find the answer in the context, say so and provide general information on the topic if possible."""
    )

# Initialize or update the RAG retriever
retriever = None  # Initialize the variable
try:
    if st.sidebar.button("Apply Settings"):
        # Get or create retriever
        retriever = get_rag_retriever(collection_name, rerank_top_n, similarity_top_k)
        
        if retriever:
            # Update configurations
            retriever.configure_retriever(
                similarity_top_k=similarity_top_k, 
                rerank_top_n=rerank_top_n, 
                search_type=search_type
            )
            retriever.customize_llm(model_name=model_name, temperature=temperature)
            retriever.customize_prompt(custom_prompt)
            st.sidebar.success("Settings applied successfully!")
    else:
        # Initialize with default settings
        retriever = get_rag_retriever(collection_name, rerank_top_n, similarity_top_k)
except Exception as e:
    st.sidebar.error(f"Error configuring retriever: {str(e)}")

# Main area
st.title("RAG Query System")
st.markdown("Ask questions and get answers based on your vector database.")

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
query = st.chat_input("Ask a question...")

if query and retriever:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get documents and response
                documents, response = retriever.get_documents_and_query(
                    query, 
                    similarity_top_k=similarity_top_k,
                    rerank_top_n=rerank_top_n
                )
                
                # Display the response
                st.markdown(response.content)
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response.content})
                
                # Show retrieved documents if enabled
                if show_documents and documents:
                    with st.expander(f"Retrieved {len(documents)} documents"):
                        for i, doc in enumerate(documents):
                            st.markdown(f"**Document {i+1}**")
                            st.markdown(f"```\n{doc.page_content}\n```")
                            if hasattr(doc, 'metadata') and doc.metadata:
                                st.json(doc.metadata)
                            st.divider()
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
elif query and not retriever:
    st.error("RAG system is not properly initialized. Please check your configuration.")

# Debugging information in an expander
with st.expander("Debug Information"):
    st.write("Session ID:", st.session_state.session_id)
    st.write("Collection Name:", collection_name)
    st.write("Model:", model_name)
    st.write("Vector DB Results:", similarity_top_k)
    st.write("Reranked Results:", rerank_top_n)
    st.write("Temperature:", temperature)
    st.write("Current Time:", datetime.now().isoformat())
    if retriever:
        st.write("RAG System: Initialized")
    else:
        st.write("RAG System: Not initialized")

# Footer with additional information
st.markdown("---")
st.markdown("Built with Streamlit, LangChain, AstraDB, and NVIDIA Reranking")
