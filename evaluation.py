from vectorstore import load_vectorstore
from embedding import get_embedding_model

INDEX_PATH = "faiss_index"

embedding_model = get_embedding_model()
vectorstore = load_vectorstore(INDEX_PATH, embedding_model)

test_cases = [
    {
        "question": "What is gradient descent?",
        "expected_keyword": "gradient"
    },
    {
        "question": "What does LangChain do?",
        "expected_keyword": "chain"
    }
]

for test in test_cases:
    docs = vectorstore.similarity_search(test["question"], k=3)

    combined_text = " ".join([doc.page_content.lower() for doc in docs])

    if test["expected_keyword"].lower() in combined_text:
        print(f"PASS: {test['question']}")
    else:
        print(f"FAIL: {test['question']}")