from langgraph.graph import StateGraph, END
from typing import TypedDict

class RAGState(TypedDict):
    question: str
    context: str
    need_retrieval: bool
    answer: str

def decide(state: RAGState):
    question = state["question"]

    def decide(state: RAGState):
        llm = get_llm()

        decision_prompt = f"""
                    You are a classifier.

                    Determine if the following question requires retrieving documents 
                    from a knowledge base to answer correctly.

                    Respond ONLY with:
                    YES
                    or
                    NO

                    Question:
                    {state['question']}
                    """

    response = llm.invoke(decision_prompt).content.strip()

    need_retrieval = response.upper() == "YES"

    return {"need_retrieval": need_retrieval}

    need_retrieval = not any(k in question.lower() for k in simple_keywords)

    return {"need_retrieval": need_retrieval}

def retrieve(state: RAGState):
    docs = vectorstore.similarity_search(state["question"], k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    return {"context": context}

def generate(state: RAGState):
    llm = get_llm()
    prompt = get_prompt()

    chain = prompt | llm

    response = chain.invoke({
        "context": state.get("context", ""),
        "question": state["question"]
    })

    return {"answer": response.content}

graph = StateGraph(RAGState)

graph.add_node("decide", decide)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.set_entry_point("decide")

graph.add_conditional_edges(
    "decide",
    lambda state: "retrieve" if state["need_retrieval"] else "generate",
)

graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()

result = app.invoke({"question": "Explain the policy in the PDF"})
print(result["answer"])