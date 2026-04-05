import streamlit as st
import time
import pandas as pd
from vectorstore import load_vectorstore
from embedding import get_embedding_model
from search import answer_query


# just added for making some change in order to check the CI/CD 
# Import our new database logging system
from database import init_db, log_interaction, engine

# Ensure DB is created
init_db()

st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("📚 MedRAG Knowledge Assistant")
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

# -------------------------------------------------------------
# TWO MAIN TABS: Chat and Analytics
# -------------------------------------------------------------
tab1, tab2 = st.tabs(["💬 Chat Assistant", "📊 Analytics Dashboard"])

with tab1:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.chat_input("Ask something...")

    if query:
        with st.spinner("Thinking..."):
            strict_mode = mode == "Strict RAG (No fallback)"

            # Start timing
            start_time = time.time()
            
            answer, sources = answer_query(
                vectorstore,
                query,
                threshold=1.15,
                debug=False,
                history=st.session_state.chat_history
            )
            
            # End timing
            response_time = time.time() - start_time
            num_sources = len(sources) if sources else 0
            
            # --- POSTGRESQL / SQLITE LOGGING ---
            # Logs the question, answer, how many sources it used, and how fast it was
            log_interaction(query, answer, num_sources, response_time)

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

# -------------------------------------------------------------
# ANALYTICS DASHBOARD TAB (USING PANDAS)
# -------------------------------------------------------------
with tab2:
    st.header("📈 System Analytics & Logging")
    st.write("This dashboard connects directly to the system's database to analyze query trends, model performance, and reliability using **Pandas**.")
    
    # Reload button
    if st.button("Refresh Analytics"):
        st.rerun()

    try:
        # Read the SQLite/PostgreSQL Database directly into a Pandas DataFrame
        df = pd.read_sql_table("query_logs", engine)
        
        if not df.empty:
            st.metric("Total Queries Processed", len(df))
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Query Response Times")
                # Pandas plotting wrapper
                st.line_chart(df[['id', 'response_time']].set_index('id'))
                
            with col2:
                st.subheader("System Reliability / Speed")
                avg_time = df['response_time'].mean()
                st.metric("Avg Generation Speed", f"{avg_time:.2f} seconds")
                
                # Check how many were RAG vs Fallback/Emergency (0 sources = no retrieval)
                total_fallbacks = len(df[df['sources_used'] == 0])
                st.metric("Zero-Context Answers (LLM Fallback)", total_fallbacks)
            
            st.subheader("📋 Recent Database Logs")
            # Create a clean display version using Pandas Dataframe handling
            display_df = df.copy()
            st.dataframe(
                display_df.sort_values(by='timestamp', ascending=False)
            )
        else:
            st.info("No queries found yet. Ask a question in the Chat Assistant tab to populate this dashboard!")
            
    except Exception as e:
        st.error(f"Could not load an analytics table: {e}")
