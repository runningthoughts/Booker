"""
FastAPI backend for the Booker application.
"""

import json
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from booker.qa import answer_question, answer_question_stream
from booker.retriever import BookerRetriever
from booker.settings import BOOKS_ROOT

app = FastAPI(title="Booker API", description="RAG-based book Q&A system")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/library", StaticFiles(directory="library"), name="library")


def resolve_paths(book_id: str):
    base = BOOKS_ROOT / book_id / "build"
    return {
        "db":    base / "db" / "booker.db",
        "index": base / "indexes" / "booker.faiss",
        "meta":  BOOKS_ROOT / book_id / "assets" / "book_meta.json",
        "cover": BOOKS_ROOT / book_id / "assets" / "cover.jpg"
    }


class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str
    k: int = 5


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.post("/ask/{book_id}")
async def ask_question(book_id: str, request: QuestionRequest) -> Dict[str, Any]:
    """
    Answer a question based on the ingested books.
    
    Args:
        book_id: The book identifier
        request: Question request containing the question and optional k parameter
    
    Returns:
        Dictionary containing the answer and sources
    """
    try:
        paths = resolve_paths(book_id)
        retriever = BookerRetriever(paths["db"], paths["index"], paths.get("meta"), paths.get("cover"))
        try:
            result = answer_question(request.question, retriever, k=request.k)
            return result
        finally:
            retriever.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/{book_id}/stream")
async def ask_question_stream(book_id: str, request: QuestionRequest):
    """
    Answer a question with streaming response.
    
    Args:
        book_id: The book identifier
        request: Question request containing the question and optional k parameter
    
    Returns:
        Server-sent events stream with answer chunks and sources
    """
    try:
        paths = resolve_paths(book_id)
        retriever = BookerRetriever(paths["db"], paths["index"], paths.get("meta"), paths.get("cover"))
        
        def generate_stream():
            try:
                for chunk in answer_question_stream(request.question, retriever, k=request.k):
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                retriever.close()
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 