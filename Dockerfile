FROM python:3.10

WORKDIR /app

# Fix SSL + certificates
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && update-ca-certificates

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]