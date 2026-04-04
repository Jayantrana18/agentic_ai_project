FROM python:3.10-slim

WORKDIR /app

# System deps (smaller + faster)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create required dirs (important for your project)
RUN mkdir -p data faiss_index

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]