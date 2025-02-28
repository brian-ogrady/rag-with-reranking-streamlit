import os

from typing import List

import streamlit as st

from llama_index.core import SimpleDirectoryReader
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI
from llama_index.finetuning import generate_qa_embedding_pairs

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

def load_corpus(files: List[str]) -> List[TextNode]:

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()

    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=True)

    return nodes

def generate_questions_from_files(file_directory: str, max_files: int = 100) -> List[str]:
    file_paths = ["/".join([file_directory, name]) for name  in os.listdir(file_directory)[:max_files]]
    os.listdir(file_directory)[:max_files]
    nodes = load_corpus(file_paths)
    qa_dataset: EmbeddingQAFinetuneDataset = generate_qa_embedding_pairs(
        llm=OpenAI(model="gpt-3.5-turbo"),
        nodes=nodes,
        output_path="train_dataset.json",
    )
    return [q for q in qa_dataset.queries.values()]