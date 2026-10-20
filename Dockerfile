# Hugging Face Spaces (Docker, free tier) / any Docker host.
# Bakes the handbook index into the image at build time so the
# container starts with zero warmup. Spaces expects port 7860.
FROM python:3.12-slim

WORKDIR /app

# git is needed only to fetch the handbook corpus at build time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_handbook_qa ./rag_handbook_qa

RUN git clone --depth 1 https://github.com/BhagyashriT/java-backend-interview-handbook /tmp/corpus \
 && python -m rag_handbook_qa.ingest --corpus /tmp/corpus --out index/chunks.json \
 && rm -rf /tmp/corpus

ENV INDEX_PATH=index/chunks.json \
    EMBEDDING=tfidf \
    LLM_PROVIDER=stub \
    DEFAULT_TOP_K=4

EXPOSE 7860

CMD ["sh", "-c", "uvicorn rag_handbook_qa.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
