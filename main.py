from data_loader import load_documents_from_folder
from embedding import chunk_documents, get_embedding_model
from vectorstore import create_vectorstore, save_vectorstore, load_vectorstore
from search import answer_query

DATA_PATH = "data"
INDEX_PATH = "faiss_index"

def build_index():
    docs = load_documents_from_folder(DATA_PATH)
    chunks = chunk_documents(docs)
    embedding_model = get_embedding_model()
    vectorstore = create_vectorstore(chunks, embedding_model)
    save_vectorstore(vectorstore, INDEX_PATH)
    return vectorstore

def load_or_build():
    embedding_model = get_embedding_model()
    vectorstore = load_vectorstore(INDEX_PATH, embedding_model)

    if vectorstore is None:
        vectorstore = build_index()

    return vectorstore
chat_history = []
if __name__ == "__main__":
    vectorstore = load_or_build()

    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        answer, sources = answer_query(
            vectorstore,
            query,
            threshold=1.15,
            debug=True,
            history=chat_history
        )

        print("\nAnswer:\n", answer)

        if sources:
            print("\nSources:")
            for doc in sources:
                print("-", doc.metadata.get("source", "Unknown"))

        # Save to memory
        chat_history.append({
            "question": query,
            "answer": answer
        })