FROM python:3.10-slim

WORKDIR /app

# Install only needed deps
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --prefer-binary --no-cache-dir -r requirements.txt

# Copy AFTER install (for caching)
COPY . .

RUN mkdir -p data faiss_index

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]