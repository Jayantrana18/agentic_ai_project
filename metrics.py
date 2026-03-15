from vectorstore import load_vectorstore
from embedding import get_embedding_model

INDEX_PATH = "faiss_index"

embedding_model = get_embedding_model()
vectorstore = load_vectorstore(INDEX_PATH, embedding_model)

test_queries = [
    "What are the basics of LangChain?",
    "Explain the architecture of LangChain.",
    "Is Donald Trump founder of LangChain?",
    "What is the vector store used for?"
]

threshold = 1.15

correct_retrieval = 0
fallback_count = 0

for query in test_queries:
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)

    if not docs_with_scores:
        fallback_count += 1
        continue

    best_score = docs_with_scores[0][1]

    print(f"\nQuery: {query}")
    print("Best Score:", best_score)

    if best_score <= threshold:
        correct_retrieval += 1
    else:
        fallback_count += 1

print("\n--- Metrics ---")
print("Retrieval Accepted:", correct_retrieval)
print("Fallback Triggered:", fallback_count)