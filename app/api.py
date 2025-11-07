import os
from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import RAGPipeline

app = FastAPI(title="RAG Travel Assistant", version="0.1.0")
rag = RAGPipeline()

class AskRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask(req: AskRequest):
    answer = await rag.query(req.query)
    return {"answer": answer}
