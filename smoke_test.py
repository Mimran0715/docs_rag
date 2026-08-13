import document_loader
import embeddings
from dotenv import load_dotenv
import os

load_dotenv()
DOCS_DIRECTORY = os.environ.get("DOCS_DIRECTORY")

def main():
    persist_directory = "./chroma_langchain_db"
    embedding_model = 'nomic-embed-text'
    docs = document_loader.load_docs(doc_dir=DOCS_DIRECTORY)
    print(f"Loaded and split into {len(docs)} chunks")
    print(docs[0].page_content[:200])
    print(docs[0].metadata)

    print(f"Checking metadata")
    for doc in docs[:5]:
        print(f"Doc metadata {doc.metadata}")

    chunked_docs = document_loader.chunk_docs(docs)
    print(f"Chunked into {len(chunked_docs)} chunks")
    print(chunked_docs[0].page_content[:50])
    print(chunked_docs[0].metadata)

    vector_store = embeddings.get_vector_store(model=embedding_model, chunked_documents=chunked_docs, persist_directory=persist_directory)
    results = embeddings.query(q="how do you add data targets", vector_store=vector_store)
    if results:
        print(f'Example result: {results[0].page_content}')

if __name__ == "__main__":
    main()
