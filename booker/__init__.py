"""
Booker: A RAG-based book Q&A system with intelligent routing.

This package provides functionality to ingest books (PDF/EPUB), create embeddings,
and answer questions based on the book content. Supports intelligent routing between
local book content and global web search based on book metadata.
"""

from .qa import answer_question, load_book_meta, call_llm_router
from .retriever import BookerRetriever
from .ingest_book import BookerIngestor
from .models import BookMeta
from .web_search import answer_from_web

__version__ = "2.0.0" 