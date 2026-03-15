import streamlit as st
from vectorstore import load_vectorstore
from embedding import get_embedding_model
from search import answer_query

st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("📚 RAG Knowledge Assistant")
st.sidebar.header("⚙️ Settings")

mode = st.sidebar.radio(
    "Answering Mode",
    ["Hybrid (RAG + LLM fallback)", "Strict RAG (No fallback)"]
)

show_scores = st.sidebar.checkbox("Show Similarity Scores", value=True)
INDEX_PATH = "faiss_index"

@st.cache_resource
def load_system():
    embedding_model = get_embedding_model()
    return load_vectorstore(INDEX_PATH, embedding_model)

vectorstore = load_system()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask something...")

if query:
    with st.spinner("Thinking..."):
        strict_mode = mode == "Strict RAG (No fallback)"

        answer, sources = answer_query(
            vectorstore,
            query,
            threshold=1.15,
            debug=False,
            history=st.session_state.chat_history,
            force_strict=strict_mode
        )

        st.session_state.chat_history.append({
            "question": query,
            "answer": answer,
            "sources": sources
        })

for chat in st.session_state.chat_history:
    st.chat_message("user").write(chat["question"])
    st.chat_message("assistant").write(chat["answer"])

    if chat["sources"]:
        with st.expander("Sources"):
            for s in chat["sources"]:
                source = s.metadata.get("source", "Unknown")
                page = s.metadata.get("page", "N/A")

                st.write(f"📄 {source} | Page: {page}")

                preview = s.page_content[:300]
                st.caption(preview + "...")