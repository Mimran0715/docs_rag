from langchain_ollama import OllamaEmbeddings
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
import os

def generate_embeddings(model: str, documents: List[Document], persist_directory:str, batch_size:int = 50) -> Chroma:
    embeddings = OllamaEmbeddings(model=model)
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    for i in range(0, len(documents), batch_size):
        batch = documents[i: i+batch_size]
        print(f"Embedding batch {i // batch_size + 1} / {(len(documents) - 1) // batch_size + 1}") # double check
        vector_store.add_documents(batch)
    return vector_store

def load_vector_store(model: str, persist_directory: str = "./chroma_langchain_db") -> Chroma:
    embeddings = OllamaEmbeddings(model=model)
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

def get_vector_store(model: str, chunked_documents: List[Document], persist_directory: str = "./chroma_langchain_db") -> Chroma:
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("Loading existing vector store from disk")
        return load_vector_store(model, persist_directory=persist_directory)
    else:
        print("No existing vector store found...generating")
        return generate_embeddings(model, chunked_documents, persist_directory=persist_directory)

def query(q:str, vector_store: Chroma, k:int = 4) -> List[Document]:
    return vector_store.similarity_search(q, k=k)