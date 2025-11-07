import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import httpx

load_dotenv()

USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./storage")

class RAGPipeline:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        self.collection = self.client.get_or_create_collection(name="travel_docs", embedding_function=self.emb_fn)

    async def query(self, question: str) -> str:
        res = self.collection.query(query_texts=[question], n_results=4)
        contexts = [d for d in (res.get("documents") or [[]])[0]]
        context = "\n\n".join(contexts) if contexts else "No docs found."
        if USE_OPENAI:
            return await self._openai_generate(question, context)
        return f"""Answer (grounded):
Question: {question}
Context:
{context}
Response:
- {self._extractive(context, question)}
"""

    def _extractive(self, ctx: str, q: str) -> str:
        return " ".join(ctx.strip().split()[:80])

    async def _openai_generate(self, question: str, context: str) -> str:
        if not OPENAI_API_KEY:
            return "OPENAI_API_KEY not set."
        prompt = f"""You are a bus/travel assistant. Answer using ONLY the context. If unknown, say 'Not in policy'.
Context:
{context}

Question: {question}
Return a concise answer."""
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post("https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": OPENAI_MODEL,
                    "temperature": 0.2,
                    "messages": [
                        {"role": "system", "content": "Be concise and factual."},
                        {"role": "user", "content": prompt}
                    ]
                })
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
