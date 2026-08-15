from langchain_ollama import OllamaEmbeddings
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
import os
from rank_bm25 import BM25Okapi
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


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

def bm_25_raw(corpus: List[Document], q: str):
    tokenized_corpus = [doc.page_content.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = q.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    top_scores = bm25.get_top_n(tokenized_query, corpus, n=1)
    return doc_scores, top_scores

def build_bm25(documents: List[Document], k:int=4) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever

def query_bm25(q:str, retriever:BM25Retriever) -> List[Document]:
    return retriever.invoke(q)

def build_ensemble(retrievers:List[BaseRetriever], weights: List[float]) -> EnsembleRetriever:
    print(f"Embeddings.py - weights: {weights}")
    return EnsembleRetriever(retrievers=retrievers, weights=weights)