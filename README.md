# RAG Knowledge Assistant 📚

A Retrieval-Augmented Generation (RAG) assistant that queries books and documents using local embeddings (Sentence-Transformers) and LLM endpoints via Langchain-Groq. Built with a user-friendly Streamlit interface.

## Features ✨

* **Hybrid & Strict RAG Modes:** Choose between strict context retrieval without hallucination fallback or hybrid mode using an LLM.
* **Vector Indexing:** High-performance similarity search powered by FAISS.
* **Source Transparency:** Exposes page numbers and excerpts from the original documents matching the retrieved data.
* **Conversation History:** Maintains memory of the current user session context.

## Prerequisites 🛠️

* Python 3.8 or higher
* Recommended to use a Virtual Environment (`venv`)

## Installation 📦

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd RAG_books
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory based on the APIs required, e.g.:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage 🚀

1. **Running the Application:**
   After installing all requirements, launch the Streamlit server:
   ```bash
   streamlit run app.py
   ```

2. **Interacting with the App:**
   * Adjust settings on the left sidebar to change modes (Hybrid vs. Strict RAG).
   * Put your dataset files in the `data/` directory (if supported and handled by your `data_loader.py` script prior to indexing).
   * Ensure that the `faiss_index/` has been correctly populated with document vectors.

## Project Structure 📁

- `app.py`: Streamlit main application logic and frontend user interface.
- `search.py`: Handles querying logic and generation via Langchain.
- `embedding.py`: Manages the local or remote embedding model (via Sentence-Transformers).
- `vectorstore.py`: Abstractions to load and build the FAISS vector database.
- `data_loader.py`: Script to process input documents (e.g. PDFs) into text chunks.
- `evaluation.py`: To assess response quality and retrieval performance (if implemented).
- `requirements.txt`: Python package requirements.
- `faiss_index/`: Default directory for storing the serialized FAISS vector store.
- `data/`: Directory intended for placing raw input documents.

## License 📜

[MIT License](LICENSE) (You can update this section depending on your specific project license).
