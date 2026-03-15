import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
    JSONLoader,
)

def load_file(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path)
        elif ext == ".csv":
            loader = CSVLoader(file_path)
        elif ext in [".xls", ".xlsx"]:
            loader = UnstructuredExcelLoader(file_path)
        elif ext in [".doc", ".docx"]:
            loader = UnstructuredWordDocumentLoader(file_path)
        elif ext == ".json":
            loader = JSONLoader(file_path=file_path, jq_schema=".", text_content=False)
        else:
            return []

        return loader.load()

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def load_documents_from_folder(folder_path: str) -> List[Document]:
    all_docs = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            docs = load_file(file_path)
            all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} documents")
    return all_docs