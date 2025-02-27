import os
import uuid
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_astradb import AstraDBVectorStore
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.messages import AIMessage
from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain.retrievers import ContextualCompressionRetriever


ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_API_ENDPOINT = st.secrets["ASTRA_DB_API_ENDPOINT"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
RERANKER_URL = st.secrets["NV_RERANK_URL"]


class RAGRetriever:

    def __init__(
        self, 
        collection_name: str, 
        model_name: str = "gpt-4o",
        rerank_top_n: int = 3,
        similarity_top_k: int = 20,
    ) -> None:
        """Initialize the RAG retriever with AstraDB vector store.
        
        Args:
            collection_name: Name of the existing AstraDB collection
            model_name: Name of the OpenAI model to use
            rerank_top_n: Number of documents to keep after reranking
            similarity_top_k: Number of documents to retrieve from vector DB
            
        Raises:
            ValueError: If RERANKER_URL is not configured in environment variables
        """
        if not RERANKER_URL:
            raise ValueError("RERANKER_URL not found in environment variables. Reranking is required.")
        
        self.llm = ChatOpenAI(model=model_name)
        
        self.vector_store = AstraDBVectorStore(
            collection_name=collection_name,
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            autodetect_collection=True
        )
        
        self.rerank_top_n = rerank_top_n
        self.similarity_top_k = similarity_top_k
        
        self.base_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": similarity_top_k}
        )
        
        self.reranker = NVIDIARerank(
            base_url=RERANKER_URL,
            model="nvidia/llama-3.2-nv-rerankqa-1b-v2",
            top_n=rerank_top_n
        )
        
        self.retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker,
            base_retriever=self.base_retriever
        )
        
        self.prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant providing helpful information based on the context provided.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer the question based on the context provided. If you cannot find the answer in the context, 
        say so and provide general information on the topic if possible.
        """)
        
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
        )
    
    def query(self, question: str, session_id: Optional[str] = None) -> AIMessage:
        """
        Query the RAG system with a question.
        
        Args:
            question: Question to ask the RAG system
            session_id: Optional session identifier for tracing
            
        Returns:
            AIMessage: Response message from the LLM containing the answer
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
            
        response = self.chain.invoke(question)
        
        return response
    
    def configure_retriever(
        self, 
        similarity_top_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None, 
        search_type: str = "similarity"
    ) -> None:
        """
        Configure the retriever settings.
        
        Args:
            similarity_top_k: Number of documents to retrieve from vector DB
            rerank_top_n: Number of documents to keep after reranking
            filter: Filter to apply to the search
            search_type: Type of search to perform ('similarity', 'mmr', etc.)
        """
        if similarity_top_k is not None:
            self.similarity_top_k = similarity_top_k
            
        if rerank_top_n is not None:
            self.rerank_top_n = rerank_top_n
            
        search_kwargs: Dict[str, Any] = {"k": self.similarity_top_k}
        if filter:
            search_kwargs["filter"] = filter
            
        self.base_retriever = self.vector_store.as_retriever(
            search_kwargs=search_kwargs,
            search_type=search_type
        )
        
        self.reranker.top_n = self.rerank_top_n
        
        self.retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker,
            base_retriever=self.base_retriever
        )
        
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
        )
    
    def customize_prompt(self, prompt_template: str) -> None:
        """
        Customize the prompt template.
        
        Args:
            prompt_template: Custom prompt template string
        """
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
        )
    
    def customize_llm(
        self, 
        model_name: str = "gpt-4o", 
        temperature: float = 0.0, 
        max_tokens: Optional[int] = None
    ) -> None:
        """
        Customize the LLM settings.
        
        Args:
            model_name: Name of the model to use
            temperature: Temperature setting for generation
            max_tokens: Maximum tokens to generate
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
        )
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents for a query.
        
        Args:
            query: Query text
            
        Returns:
            List[Document]: List of relevant documents
        """
        return self.retriever.get_relevant_documents(query)
    
    def similarity_search(
        self, 
        query: str, 
        similarity_top_k: Optional[int] = None, 
        rerank_top_n: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform a similarity search, applying compression if enabled.
        
        Args:
            query: Query text
            similarity_top_k: Number of documents to retrieve from vector DB
            rerank_top_n: Number of documents to keep after reranking
            filter: Filter to apply to the search
            
        Returns:
            List[Document]: List of retrieved and potentially reranked documents
        """
        temp_update = (similarity_top_k is not None and similarity_top_k != self.similarity_top_k) or \
                     (rerank_top_n is not None and rerank_top_n != self.rerank_top_n) or \
                     (filter is not None)
        
        if temp_update:
            original_similarity_top_k = self.similarity_top_k
            original_rerank_top_n = self.rerank_top_n
            original_filter = self.base_retriever.search_kwargs.get("filter")
            
            self.configure_retriever(
                similarity_top_k=similarity_top_k or self.similarity_top_k,
                rerank_top_n=rerank_top_n or self.rerank_top_n,
                filter=filter
            )
            
            documents = self.get_relevant_documents(query)
            
            self.configure_retriever(
                similarity_top_k=original_similarity_top_k,
                rerank_top_n=original_rerank_top_n,
                filter=original_filter
            )
            
            return documents
        
        return self.get_relevant_documents(query)
    
    def get_documents_and_query(
        self, 
        query: str, 
        similarity_top_k: Optional[int] = None, 
        rerank_top_n: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Document], AIMessage]:
        """
        Perform a search and query in one operation, useful for debugging and UI.
        
        Args:
            query: Query text
            similarity_top_k: Number of documents to retrieve from vector DB
            rerank_top_n: Number of documents to keep after reranking
            filter: Filter to apply to the search
            
        Returns:
            Tuple[List[Document], AIMessage]: Retrieved documents and LLM response
        """
        documents = self.similarity_search(
            query, 
            similarity_top_k=similarity_top_k, 
            rerank_top_n=rerank_top_n, 
            filter=filter
        )
        
        response = self.query(query)
        
        return documents, response


if __name__ == "__main__":
    try:
        rag = RAGRetriever(
            rerank_top_n=3,
            similarity_top_k=20
        )
        
        response = rag.query("What is RAG and how does it relate to LangChain?")
        print("Response:", response.content)
        
        rag.configure_retriever(similarity_top_k=15, rerank_top_n=5)
        
        custom_prompt = """
        You are a technical expert providing detailed information on AI topics.
        
        Context information:
        {context}
        
        User question:
        {question}
        
        Provide a comprehensive answer with technical details when relevant.
        """
        rag.customize_prompt(custom_prompt)
    except ValueError as e:
        print(f"Error initializing RAG: {e}")