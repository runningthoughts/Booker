"""
Retrieval module for finding similar chunks using FAISS and DuckDB.
"""

import json
import pickle
import re
import pathlib
from functools import lru_cache
from typing import List, Dict, Any, Optional
from pathlib import Path

import duckdb
import faiss
import numpy as np
import openai

from . import settings

# Initialize OpenAI client
openai.api_key = settings.OPENAI_API_KEY

# Constants for sidecar functionality
KEYWORD_BOOST = 0.25   # weight added to FAISS similarity; tune freely
_WORD_RE = re.compile(r"\w+")


@lru_cache(maxsize=16)
def _load_sidecar(book_id: str) -> Optional[Dict[str, Any]]:
    """Load sidecar JSON for a book, with caching."""
    try:
        # Try to find the sidecar file in the book's build directory
        from .utils import get_book_build_path
        sidecar_dir = get_book_build_path(book_id) / "sidecars"
        sidecar_path = sidecar_dir / f"{book_id}_summaries.json"
        
        if sidecar_path.exists():
            with sidecar_path.open(encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    
    # Fallback to library structure
    try:
        from .utils import get_books_root
        books_root = get_books_root()
        sidecar_path = books_root / book_id / "build" / "sidecars" / f"{book_id}_summaries.json"
        
        if sidecar_path.exists():
            with sidecar_path.open(encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    
    return None


def _keyword_overlap_score(query: str, card: Dict[str, Any]) -> float:
    """Calculate keyword/entity overlap score between query and chunk card."""
    if not card:
        return 0.0
    
    # Extract and normalize query words
    query_words = set(_WORD_RE.findall(query.lower()))
    if not query_words:
        return 0.0
    
    # Extract target words from keywords and entities
    target_words = set()
    
    # Add keywords
    keywords = card.get("keywords", [])
    if isinstance(keywords, list):
        for keyword in keywords:
            if isinstance(keyword, str):
                target_words.update(_WORD_RE.findall(keyword.lower()))
    
    # Add entities
    entities = card.get("entities", [])
    if isinstance(entities, list):
        for entity in entities:
            if isinstance(entity, str):
                target_words.update(_WORD_RE.findall(entity.lower()))
    
    if not target_words:
        return 0.0
    
    # Calculate overlap score
    intersection = query_words.intersection(target_words)
    return len(intersection) / len(target_words)


class BookerRetriever:
    """Handles semantic search and retrieval of book chunks."""
    
    def __init__(self, db_path: Path, index_path: Path, book_id: str = None):
        """Initialize the retriever with FAISS index and database connection."""
        self.db_conn = duckdb.connect(str(db_path))
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.index_path = index_path
        self.index_meta_path = index_path.parent / "booker.pkl"
        self.book_id = book_id
        
        # Load sidecar if book_id is provided
        self._sidecar = _load_sidecar(book_id) if book_id else None
        
        self._load_index()
    
    def _load_index(self) -> None:
        """Load FAISS index and metadata."""
        if self.index_path.exists() and self.index_meta_path.exists():
            self.faiss_index = faiss.read_index(str(self.index_path))
            with open(self.index_meta_path, 'rb') as f:
                self.chunk_metadata = pickle.load(f)
        else:
            raise FileNotFoundError("FAISS index not found. Please run ingestion first.")
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for a query string."""
        response = openai.embeddings.create(
            model=settings.EMBED_MODEL,
            input=[query]
        )
        embedding = np.array([response.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(embedding)
        return embedding
    
    def _mmr_rerank(self, query_embedding: np.ndarray, candidate_embeddings: List[np.ndarray], 
                   candidate_indices: List[int], lambda_param: float = 0.7, k: int = 5) -> List[int]:
        """
        Maximal Marginal Relevance (MMR) reranking to reduce redundancy.
        
        Args:
            query_embedding: The query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            candidate_indices: Original indices of candidates
            lambda_param: Balance between relevance and diversity (0-1)
            k: Number of results to return
        
        Returns:
            List of reranked indices
        """
        if not candidate_embeddings or k <= 0:
            return []
        
        selected_indices = []
        remaining_indices = list(range(len(candidate_embeddings)))
        
        # Select first item (most similar to query)
        similarities = [np.dot(query_embedding[0], emb) for emb in candidate_embeddings]
        best_idx = np.argmax(similarities)
        selected_indices.append(candidate_indices[best_idx])
        remaining_indices.remove(best_idx)
        
        # Select remaining items using MMR
        while len(selected_indices) < k and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance to query
                relevance = np.dot(query_embedding[0], candidate_embeddings[idx])
                
                # Maximum similarity to already selected items
                max_similarity = 0
                for selected_orig_idx in selected_indices:
                    selected_idx = candidate_indices.index(selected_orig_idx)
                    similarity = np.dot(candidate_embeddings[idx], candidate_embeddings[selected_idx])
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append(mmr_score)
            
            # Select item with highest MMR score
            best_mmr_idx = np.argmax(mmr_scores)
            selected_idx = remaining_indices[best_mmr_idx]
            selected_indices.append(candidate_indices[selected_idx])
            remaining_indices.remove(selected_idx)
        
        return selected_indices
    
    def similar_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar chunks for a given query.
        
        Args:
            query: The search query
            k: Number of chunks to return
        
        Returns:
            List of chunk dictionaries with text, summaries, and metadata
        """
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not loaded")
        
        # Get query embedding
        query_embedding = self._get_query_embedding(query)
        
        # Search FAISS index (get more candidates for MMR)
        search_k = min(k * 3, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(query_embedding, search_k)
        
        # Apply sidecar-based re-ranking if available
        if self._sidecar and "chunks" in self._sidecar:
            boosts = np.zeros_like(scores)
            for col, idx in enumerate(indices[0]):
                if idx == -1:
                    continue
                # Get chunk card from sidecar (chunk_id is 1-based, but idx is 0-based)
                chunk_id = idx + 1
                if chunk_id <= len(self._sidecar["chunks"]):
                    card = self._sidecar["chunks"][chunk_id - 1]  # Convert to 0-based for list access
                    boosts[0, col] = _keyword_overlap_score(query, card) * KEYWORD_BOOST
            scores += boosts  # in-place; higher still means more similar
        
        # Get candidate embeddings for MMR
        candidate_embeddings = []
        candidate_indices = []
        
        for idx in indices[0]:
            if idx != -1:  # Valid index
                # Reconstruct embedding from FAISS index
                embedding = self.faiss_index.reconstruct(int(idx))
                candidate_embeddings.append(embedding)
                candidate_indices.append(int(idx))
        
        # Apply MMR reranking
        reranked_indices = self._mmr_rerank(
            query_embedding, candidate_embeddings, candidate_indices, k=k
        )
        
        # Fetch chunk data from database
        results = []
        for faiss_idx in reranked_indices:
            # Find chunk_id from metadata - the faiss_idx corresponds to the order in which chunks were added
            # Since we use faiss_index.ntotal as chunk_id, we can derive it
            chunk_id = faiss_idx + 1  # FAISS indices are 0-based, chunk_ids start from 1
            
            # Verify this chunk exists in our metadata
            chunk_exists = any(meta.get("chunk_id") == chunk_id for meta in self.chunk_metadata)
            if not chunk_exists:
                continue
            
            # Get chunk and summary data
            chunk_data = self.db_conn.execute("""
                SELECT c.chunk_id, c.book_id, c.file_name, c.chapter_no, c.chapter_title,
                       c.page_start, c.page_end, c.text, s.summary, s.keywords, s.entities
                FROM chunks c
                LEFT JOIN summaries s ON c.chunk_id = s.chunk_id
                WHERE c.chunk_id = ?
            """, (chunk_id,)).fetchone()
            
            if chunk_data:
                result = {
                    "chunk_id": chunk_data[0],
                    "book_id": chunk_data[1],
                    "file_name": chunk_data[2],
                    "chapter_no": chunk_data[3],
                    "chapter_title": chunk_data[4],
                    "page_start": chunk_data[5],
                    "page_end": chunk_data[6],
                    "text": chunk_data[7],
                    "summary": chunk_data[8],
                    "keywords": chunk_data[9],
                    "entities": chunk_data[10]
                }
                results.append(result)
        
        return results
    
    def close(self) -> None:
        """Close database connection."""
        self.db_conn.close() 