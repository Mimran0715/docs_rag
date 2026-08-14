import document_loader
import embeddings
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from typing import List, Dict
from langchain_core.documents import Document
from collections import defaultdict

load_dotenv()
DOCS_DIRECTORY = os.environ.get("DOCS_DIRECTORY")
QA_TEST = os.environ.get("QA_TEST")

def parse_qa_text() -> List[Dict[str, object]]:
    qa_pairs = []
    current = {}
    with open(QA_TEST) as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    qa_pairs.append(current)
                    current = {}
                continue
            if line.startswith("Q"):
                current["question"] = line[len("Q:"):].strip()
            elif line.startswith("A"):
                current["answer"] = line[len("A:"):].strip()
            elif line.startswith("KEYWORDS"):
                keywords_str = line[len("KEYWORDS:"):].strip()
                current["keywords"] = [k.strip() for k in keywords_str.split(",")]
    if current:
        qa_pairs.append(current)

    return qa_pairs

def test_qa_parsing() -> None:
    results = parse_qa_text()
    for i in range(len(results)):
        print(f"QA pair {i} - {results[i]}")

def eval_query(query:str, vector_store: Chroma) -> List[Document]:
    results = embeddings.query(query, vector_store)
    if results: 
        return results
    return []

def eval_queries(qa_pairs: List[Dict[str, object]]) -> List[float]:
    pass

def check_retrival_hits(results: List[Document], keywords: List[str], threshold:float = 0.3) -> bool:
    combined_text = " ".join(doc.page_content.lower() for doc in results)
    hits = sum(1 for kw in keywords if kw.lower() in combined_text)
    return (hits / len(keywords)) >= threshold

def main():
    persist_directory = "./chroma_langchain_db"
    embedding_model = 'nomic-embed-text'

    print("Eval: Loading docs...")
    docs = document_loader.load_docs(doc_dir=DOCS_DIRECTORY)
    print("Eval: Chunking docs...")
    chunked_docs = document_loader.chunk_docs(docs)
    vector_store = embeddings.get_vector_store(model=embedding_model, chunked_documents=chunked_docs, persist_directory=persist_directory)

    qa_pairs = parse_qa_text()

    hits = []
    misses = [] # tracking losses
    for qa in qa_pairs:
        results = embeddings.query(q=qa["question"], vector_store=vector_store)
        print(f"Retrieved {len(results)} chunks")

        hit = check_retrival_hits(results, qa["keywords"])
        hits.append(hit)
        if not hit: 
            misses.append({"question": qa["question"], "answer": qa["answer"], "keywords": qa["keywords"], "retrieved": [r.page_content[:150] for r in results]})
       
    hit_metric = sum(hits) / len(hits)
    print(f"Retrieval hit rate: {hit_metric:.2%}")

    print(f"\nMisses ({len(misses)}):")
    for m in misses:
        print(f"\nQ: {m['question']}")
        print(f"Expected keywords: {m['keywords']}")
        print(f"Top retrieved chunk: {m['retrieved'][0]}")

    results_with_scores = vector_store.similarity_search_with_score("What tool is commonly used to visualize Prometheus data?", k=15)
    for doc, score in results_with_scores:
        print(f"{score:.4f} | {doc.page_content[:100]}")

if __name__ == "__main__":
    main()