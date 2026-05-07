from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ingest import ingest_documents
from utils.config import get_settings
from utils.gemini import FALLBACK_ANSWER, generate_grounded_answer
from utils.vector_store import get_vector_store


app = FastAPI(title="Company RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
def ingest() -> dict:
    try:
        return ingest_documents()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    settings = get_settings()
    try:
        vector_store = get_vector_store()
        docs = vector_store.similarity_search(request.question, k=settings.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    if not docs:
        return ChatResponse(answer=FALLBACK_ANSWER, sources=[])

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
        for doc in docs
    )
    sources = sorted(
        {
            source
            for doc in docs
            if (source := doc.metadata.get("source"))
        }
    )

    try:
        answer = generate_grounded_answer(context=context, question=request.question)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {exc}") from exc

    if answer.strip() == FALLBACK_ANSWER:
        sources = []

    return ChatResponse(answer=answer, sources=sources)
