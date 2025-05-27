"""
FastAPI backend for the Booker application.
"""

import json
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from booker.qa import answer_question, answer_question_stream

app = FastAPI(title="Booker API", description="RAG-based book Q&A system")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post("/ask")
async def ask_question(request: QuestionRequest) -> Dict[str, Any]:
    """
    Answer a question based on the ingested books.
    
    Args:
        request: Question request containing the question and optional k parameter
    
    Returns:
        Dictionary containing the answer and sources
    """
    try:
        result = answer_question(request.question, k=request.k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
async def ask_question_stream(request: QuestionRequest):
    """
    Answer a question with streaming response.
    
    Args:
        request: Question request containing the question and optional k parameter
    
    Returns:
        Server-sent events stream with answer chunks and sources
    """
    try:
        def generate_stream():
            for chunk in answer_question_stream(request.question, k=request.k):
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        
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