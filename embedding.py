from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")
    return chunks

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )