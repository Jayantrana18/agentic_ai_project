import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )


def should_retrieve(query: str):
    llm = get_llm()

    prompt = f"""
You are deciding whether a question requires retrieving information
from a document knowledge base.

Respond ONLY with:
YES
or
NO

Question:
{query}
"""

    response = llm.invoke(prompt).content.strip().upper()
    return response == "YES"


def check_medical_emergency(query: str):
    emergency_keywords = [
        "chest pain",
        "heart attack",
        "stroke",
        "difficulty breathing",
        "can't breathe",
        "severe bleeding",
        "unconscious",
        "seizure",
        "fainting",
        "sudden weakness",
        "severe head injury"
    ]

    query_lower = query.lower()
    return any(keyword in query_lower for keyword in emergency_keywords)


def detect_symptom_query(query: str):

    symptom_keywords = [
        "i have",
        "i feel",
        "my symptoms",
        "i am feeling",
        "i am having",
        "pain in",
        "fever",
        "headache",
        "nausea",
        "vomiting",
        "dizziness",
        "fatigue",
        "cough",
        "sore throat",
        "stomach pain"
    ]

    query_lower = query.lower()
    return any(keyword in query_lower for keyword in symptom_keywords)


def classify_medical_query(query: str):

    llm = get_llm()

    prompt = f"""
You are classifying a healthcare question.

Choose ONE category from:

DISEASE
SYMPTOMS
TREATMENT
PREVENTION
GENERAL

Respond with ONLY the category name.

Question:
{query}
"""

    response = llm.invoke(prompt).content.strip().upper()
    return response


def get_prompt():

    template = """
You are a healthcare knowledge assistant.

Answer using ONLY the provided medical documents.

Rules:
- Do NOT give medical diagnosis.
- Do NOT prescribe medication.
- If the information is not in the documents, say:
"I could not find reliable information in the medical sources."

Encourage users to consult healthcare professionals.

Context:
{context}

Question:
{question}
"""

    return ChatPromptTemplate.from_template(template)


def rewrite_query(query: str, history=None):

    llm = get_llm()

    history_text = ""

    if history:
        history_text = "\n".join(
            [f"Q: {h['question']}\nA: {h['answer']}" for h in history[-3:]]
        )

    prompt = f"""
You are helping improve document retrieval.

Given the previous conversation and the latest user question,
rewrite the latest question to make it clear and specific.

Return ONLY the rewritten question.

Previous conversation:
{history_text}

Latest question:
{query}
"""

    response = llm.invoke(prompt)

    return response.content.strip()


def answer_query(vectorstore, query: str, k: int = 4, threshold: float = 1.15, debug=False, history=None):

    symptom_message = ""

    # Step 0 — Emergency detection
    if check_medical_emergency(query):

        emergency_message = """
🚨 **Possible Medical Emergency Detected**

Your symptoms may indicate a serious medical condition.

Please seek immediate medical help.

📞 **Emergency Numbers (India)**
• Ambulance: **102 / 108**
• National Emergency Helpline: **112**

🏥 **What you should do immediately**
1. Call emergency services
2. Ask someone nearby to assist you
3. Go to the nearest hospital

⚠️ This AI assistant cannot provide emergency medical care.
"""

        return emergency_message, []

    # Step 1 — Symptom detection
    if detect_symptom_query(query):

        symptom_message = """
⚠️ **Symptom-based query detected**

I cannot diagnose medical conditions or provide treatment.

I will provide general educational information from medical sources.

Please consult a healthcare professional for medical advice.
"""

    # Step 2 — Medical query classification
    query_type = classify_medical_query(query)

    if debug:
        print("\nQuery Type:", query_type)

    # Step 3 — Decide if retrieval needed
    if not should_retrieve(query):

        llm = get_llm()

        response = llm.invoke(query)

        final_answer = symptom_message + "\n\n" + response.content

        return final_answer, []

    # Step 4 — Rewrite query
    rewritten_query = rewrite_query(query, history)

    if debug:
        print("\nRewritten Query:", rewritten_query)

    # Step 5 — Retrieve documents
    docs = vectorstore.similarity_search(rewritten_query, k=k)

    if not docs:
        return "I could not find reliable information in the medical sources.", []

    # Step 6 — Prepare context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Add query type hint
    context = f"Query Type: {query_type}\n\n" + context

    # Step 7 — Generate answer
    llm = get_llm()

    prompt = get_prompt()

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": query
    })

    final_answer = symptom_message + "\n\n" + response.content

    return final_answer, docs