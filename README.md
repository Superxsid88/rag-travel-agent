# RAG Travel Assistant (Low-Cost, Local-First)

A Retrieval-Augmented Generation assistant for bus/travel FAQs. Designed to run **locally** (CPU) or **cheaply** on **GCP Cloud Run**.

## 🚦 Why this project?
- Demonstrates **end-to-end RAG** with embeddings, vector store, and LLM generation.
- **Local-first** using `sentence-transformers` (MiniLM), with optional OpenAI for higher quality.
- Uses **Chroma** as a free, embedded vector DB.

## ⚙️ Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app/index_data.py  # one-time indexing
uvicorn app.api:app --reload
```
Ask:
```bash
curl -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"query":"How to cancel a ticket?"}'
```

## 🔧 Env
Copy `.env.example` to `.env`:
```env
USE_OPENAI=false
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DIR=./storage
```

## 🐳 Docker
```bash
docker build -t rag-travel-assistant .
docker run -p 8000:8000 --env-file .env rag-travel-assistant
```

## ☁️ GCP Cloud Run (cost-capped)
```bash
gcloud builds submit --tag gcr.io/$PROJECT_ID/rag-travel-assistant
gcloud run deploy rag-travel-assistant --image gcr.io/$PROJECT_ID/rag-travel-assistant --platform managed --allow-unauthenticated --memory 512Mi --cpu 1 --max-instances=2
```

## 📄 Data
Sample docs in `data/travel_docs`. Replace with BitlaSoft-specific policies later.

## 📜 License
MIT


### HF Local Generation (no OpenAI needed)
Set in `.env`:

```
USE_HF_LOCAL=true
HF_LOCAL_MODEL=google/flan-t5-small
```
This uses a small local `transformers` model on CPU or your RTX 3050.
