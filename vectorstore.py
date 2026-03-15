import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def create_vectorstore(chunks: List[Document], embedding_model):
    return FAISS.from_documents(chunks, embedding_model)

def save_vectorstore(vectorstore, path: str):
    vectorstore.save_local(path)

def load_vectorstore(path: str, embedding_model):
    if not os.path.exists(path):
        return None
    return FAISS.load_local(
        path,
        embedding_model,
        allow_dangerous_deserialization=True
    )