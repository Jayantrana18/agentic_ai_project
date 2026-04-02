import os
import sys

# Add parent dir to sys.path so we can import from the main project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langgraph.graph import StateGraph, END
from typing import TypedDict
from langgraph.checkpoint.memory import MemorySaver

from search import get_llm, get_prompt
from main import load_or_build

vectorstore = load_or_build()

class RAGState(TypedDict):
    question: str
    context: str
    need_retrieval: bool
    answer: str
    feedback: str

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

def retrieve(state: RAGState):
    docs = vectorstore.similarity_search(state["question"], k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    return {"context": context}

def generate(state: RAGState):
    llm = get_llm()
    prompt = get_prompt()

    chain = prompt | llm

    # Incorporate feedback if available to refine the answer
    question = state["question"]
    if state.get("feedback"):
        question += f"\n\n[USER FEEDBACK TO IMPROVE ANSWER]:\n{state['feedback']}\n\nPlease regenerate the answer taking this feedback into consideration."

    response = chain.invoke({
        "context": state.get("context", ""),
        "question": question
    })

    # Clear feedback after generation so it doesn't infinitely loop
    return {"answer": response.content, "feedback": ""}

def human_review(state: RAGState):
    # This is a dummy node just to hold the interrupt
    pass

def should_regenerate(state: RAGState):
    # If the user provided feedback, we route back to generate
    if state.get("feedback"):
        return "generate"
    return END

graph = StateGraph(RAGState)

graph.add_node("decide", decide)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_node("human_review", human_review)

graph.set_entry_point("decide")

graph.add_conditional_edges(
    "decide",
    lambda state: "retrieve" if state["need_retrieval"] else "generate",
)

graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "human_review")
graph.add_conditional_edges("human_review", should_regenerate)

# Set up the checkpointer
memory = MemorySaver()

# Compile the graph with an interrupt before the "human_review" node
app = graph.compile(checkpointer=memory, interrupt_before=["human_review"])

if __name__ == "__main__":
    thread_config = {"configurable": {"thread_id": "1"}}
    question = "Explain the policy in the PDF"
    
    print(f"\n[1] Starting the graph for question: '{question}'...")
    
    # Run the app until it gets interrupted
    for event in app.stream({"question": question, "feedback": ""}, config=thread_config):
        for k, v in event.items():
            print(f"Executed node: {k}")
            
    while True:
        # The graph is now paused before human_review
        snapshot = app.get_state(thread_config)
        
        if not snapshot.next:
            # We reached the END
            print("\n--- DONE ---")
            break
            
        print("\n--- HUMAN IN THE LOOP: CAUGHT INTERRUPT ---")
        answer = snapshot.values.get("answer", "No answer generated")
        print("\nGenerated Answer:")
        print("-" * 50)
        print(answer)
        print("-" * 50)
        
        user_input = input("\nDo you approve this answer? (y/n): ")
        
        if user_input.lower() == 'y':
            print("\nAnswer approved! Finishing execution...")
            # Resume without feedback
            app.update_state(thread_config, {"feedback": ""})
            for event in app.stream(None, config=thread_config):
                pass
            print("Graph completed.")
            break
        else:
            feedback = input("Please provide feedback on how to improve the answer: ")
            print("\n[2] Resuming graph execution to regenerate answer...")
            # Update state with the feedback so `should_regenerate` will route us back
            app.update_state(thread_config, {"feedback": feedback})
            
            # Pass None to resume execution from the pause point
            for event in app.stream(None, config=thread_config):
                for k, v in event.items():
                    print(f"Executed node: {k}")