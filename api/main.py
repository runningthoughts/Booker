"""
FastAPI backend for the Booker application.
"""

import json
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from booker.qa import answer_question, answer_question_stream
from booker.retriever import BookerRetriever
from booker.utils import resolve_book_paths, get_book_asset_url


class StaticFilesWithoutCaching(StaticFiles):
    """StaticFiles subclass that disables HTTP caching for development."""
    def is_not_modified(self, *args, **kwargs) -> bool:
        return False


app = FastAPI(title="Booker API", description="RAG-based book Q&A system")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Only mount static files if running locally (library folder exists)
library_path = Path("library")
if library_path.exists():
    app.mount("/library", StaticFilesWithoutCaching(directory="library"), name="library")


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


@app.get("/")
@app.head("/")
async def root():
    """Root endpoint for Render health checks."""
    return {"message": "Booker API is running", "status": "ok"}


@app.get("/config")
async def get_config():
    """Get configuration values for the frontend."""
    from booker import settings
    return {
        "data_base_url": settings.DATA_BASE_URL,
        "is_production": settings.IS_PRODUCTION
    }


@app.get("/book/{book_id}/assets")
async def get_book_assets(book_id: str):
    """
    Get asset URLs for a book (title.json, cover image).
    Returns URLs that work in both local and S3 environments.
    
    Args:
        book_id: The book identifier
    
    Returns:
        Dictionary containing asset URLs
    """
    data = {
        "title_url": get_book_asset_url(book_id, "assets/title.json"),
        "cover_url": get_book_asset_url(book_id, "assets/cover.png")
    }
    
    # Add no-cache headers to prevent caching of metadata
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    
    return JSONResponse(content=data, headers=headers)


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
        paths = resolve_book_paths(book_id)
        retriever = BookerRetriever(paths["db"], paths["index"])
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
        paths = resolve_book_paths(book_id)
        retriever = BookerRetriever(paths["db"], paths["index"])
        
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