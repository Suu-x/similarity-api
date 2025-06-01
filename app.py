from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import openai
import os

# Load your OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

class SimilarityResponse(BaseModel):
    matches: List[str]

async def get_embeddings(texts: List[str]) -> List[List[float]]:
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [e.embedding for e in response.data]

@app.post("/similarity", response_model=SimilarityResponse)
async def similarity(req: SimilarityRequest):
    texts = req.docs
    query = req.query

    all_embeddings = await get_embeddings([query] + texts)
    query_emb = np.array(all_embeddings[0])
    doc_embs = np.array(all_embeddings[1:])

    similarities = np.dot(doc_embs, query_emb) / (
        np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(query_emb)
    )

    top_indices = similarities.argsort()[::-1][:3]
    top_docs = [texts[i] for i in top_indices]

    return SimilarityResponse(matches=top_docs)
