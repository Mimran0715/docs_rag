from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from dotenv import load_dotenv
from langchain_core.documents import Document
import os
from typing import List
import re

load_dotenv()
DOCS_DIRECTORY = os.environ.get("DOCS_DIRECTORY")

def strip_frontmatter(text: str) -> str:
    '''stripping yaml frontmatter from documentation for chunking'''
    return re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)


def load_docs(doc_dir: str) -> List[Document]:
    """Loading documentation files (md) and splitting by headers"""
    loader = DirectoryLoader(doc_dir,
                              glob="**/*.md",
                              loader_cls=TextLoader,
                              loader_kwargs={"autodetect_encoding": True}, 
                              show_progress=True)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

    texts = []
    for doc in documents:
        split_docs = text_splitter.split_text(strip_frontmatter(doc.page_content))
        texts.extend(split_docs)
    return texts

def chunk_docs(docs: List[Document], chunk_size: int =800, chunk_overlap: int = 100) -> List[Document]:
    """Futher splitting oversized header sections into smaller chunks"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def main():
    docs = load_docs(doc_dir=DOCS_DIRECTORY)
    print(f"Loaded and split into {len(docs)} chunks")
    print(docs[0].page_content[:200])
    print(docs[0].metadata)

    print(f"Checking metadata")
    for doc in docs[:5]:
        print(f"Doc metadata {doc.metadata}")

    chunked_docs = chunk_docs(docs)
    print(f"Chunked into {len(chunked_docs)} chunks")
    print(chunked_docs[0].page_content[:50])
    print(chunked_docs[0].metadata)

if __name__ == "__main__":
    main()