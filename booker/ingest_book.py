"""
Book ingestion module for processing PDFs and EPUBs into searchable chunks.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import duckdb
import faiss
import numpy as np
import openai
import spacy
import tiktoken
from pypdf import PdfReader

from . import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai.api_key = settings.OPENAI_API_KEY

# Load spaCy model for entity extraction
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None


class BookerIngestor:
    """Handles the ingestion of books into the Booker system."""
    
    def __init__(self):
        """Initialize the ingestor with tokenizer and database connection."""
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.db_conn = duckdb.connect(str(settings.DB_PATH))
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.chunk_metadata: List[Dict[str, Any]] = []
        self._setup_database()
        self._load_or_create_index()
    
    def _setup_database(self) -> None:
        """Create database tables if they don't exist."""
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id      INTEGER PRIMARY KEY,
                book_id       TEXT,
                file_name     TEXT,
                chapter_no    INTEGER,
                chapter_title TEXT,
                page_start    INTEGER,
                page_end      INTEGER,
                text          TEXT
            )
        """)
        
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                chunk_id   INTEGER REFERENCES chunks(chunk_id),
                summary    TEXT,
                keywords   TEXT,
                entities   TEXT
            )
        """)
        logger.info("Database tables initialized")
    
    def _load_or_create_index(self) -> None:
        """Load existing FAISS index or create a new one."""
        if settings.INDEX_PATH.exists() and settings.INDEX_META_PATH.exists():
            self.faiss_index = faiss.read_index(str(settings.INDEX_PATH))
            with open(settings.INDEX_META_PATH, 'rb') as f:
                self.chunk_metadata = pickle.load(f)
            logger.info(f"Loaded existing FAISS index with {self.faiss_index.ntotal} vectors")
        else:
            # Create new index (3072 dimensions for text-embedding-3-large)
            self.faiss_index = faiss.IndexFlatIP(3072)
            self.chunk_metadata = []
            logger.info("Created new FAISS index")
    
    def _save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.faiss_index, str(settings.INDEX_PATH))
        with open(settings.INDEX_META_PATH, 'wb') as f:
            pickle.dump(self.chunk_metadata, f)
        logger.info("FAISS index saved")
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks based on token count."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + settings.CHUNK_SIZE, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if end >= len(tokens):
                break
            start = end - settings.CHUNK_OVERLAP
        
        return chunks
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts."""
        response = openai.embeddings.create(
            model=settings.EMBED_MODEL,
            input=texts
        )
        return [data.embedding for data in response.data]
    
    def _summarize_chunk(self, text: str) -> str:
        """Generate a summary for a text chunk."""
        response = openai.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Summarise this excerpt from the book in ≤2 sentences, dense with key info, no new facts."
                },
                {"role": "user", "content": text}
            ],
            max_tokens=100,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities and keywords from text using spaCy."""
        if nlp is None:
            return {"entities": [], "keywords": []}
        
        doc = nlp(text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        keywords = [token.lemma_.lower() for token in doc 
                   if not token.is_stop and not token.is_punct and token.is_alpha]
        
        return {"entities": entities, "keywords": list(set(keywords))}
    
    def _generate_chapter_summary(self, recaps: List[str]) -> str:
        """Generate a chapter summary from individual chunk recaps."""
        combined_recaps = " ".join(recaps)
        
        # Trim to 2000 tokens if needed
        tokens = self.tokenizer.encode(combined_recaps)
        if len(tokens) > 2000:
            tokens = tokens[:2000]
            combined_recaps = self.tokenizer.decode(tokens)
        
        response = openai.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Summarise the following bullet-point recaps of a book chapter in one coherent paragraph, ≤100 tokens."
                },
                {"role": "user", "content": combined_recaps}
            ],
            max_tokens=100,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from a PDF file."""
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def process_file(self, file_path: Path) -> None:
        """Process a single book file (PDF or EPUB)."""
        logger.info(f"Processing file: {file_path.name}")
        
        # Extract text based on file type
        if file_path.suffix.lower() == '.pdf':
            text = self._extract_text_from_pdf(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path.suffix}")
            return
        
        # Chunk the text
        chunks = self._chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Process chunks in batches
        chunk_cards = []
        recaps = []
        
        for i in range(0, len(chunks), settings.BATCH_SIZE):
            batch_chunks = chunks[i:i + settings.BATCH_SIZE]
            
            # Get embeddings for batch
            embeddings = self._get_embeddings(batch_chunks)
            
            # Process each chunk in the batch
            for j, (chunk_text, embedding) in enumerate(zip(batch_chunks, embeddings)):
                chunk_idx = i + j
                
                # Add to FAISS index
                embedding_array = np.array([embedding], dtype=np.float32)
                faiss.normalize_L2(embedding_array)
                self.faiss_index.add(embedding_array)
                
                # Insert into database
                chunk_id = self.faiss_index.ntotal  # Use FAISS index as chunk_id
                self.db_conn.execute("""
                    INSERT INTO chunks (chunk_id, book_id, file_name, chapter_no, chapter_title, page_start, page_end, text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (chunk_id, file_path.stem, file_path.name, 1, "Chapter 1", 
                     chunk_idx * 10, (chunk_idx + 1) * 10, chunk_text))
                
                # Generate summary
                recap = self._summarize_chunk(chunk_text)
                recaps.append(recap)
                
                # Extract entities and keywords
                entities_data = self._extract_entities(chunk_text)
                
                # Insert summary
                self.db_conn.execute("""
                    INSERT INTO summaries (chunk_id, summary, keywords, entities)
                    VALUES (?, ?, ?, ?)
                """, (chunk_id, recap, 
                     " ".join(entities_data["keywords"]), 
                     json.dumps(entities_data["entities"])))
                
                # Create chunk card
                chunk_card = {
                    "chunk_id": chunk_id,
                    "embedding_index": self.faiss_index.ntotal - 1,
                    "page_range": f"{chunk_idx * 10}-{(chunk_idx + 1) * 10}",
                    "summary": recap,
                    "keywords": entities_data["keywords"],
                    "entities": entities_data["entities"]
                }
                chunk_cards.append(chunk_card)
                
                # Update metadata
                self.chunk_metadata.append({
                    "chunk_id": chunk_id,
                    "file_name": file_path.name,
                    "page_start": chunk_idx * 10,
                    "page_end": (chunk_idx + 1) * 10
                })
        
        # Generate chapter summary
        chapter_summary = self._generate_chapter_summary(recaps)
        
        # Create sidecar JSON
        sidecar_data = {
            "file": file_path.name,
            "chapter_no": 1,
            "chapter_title": "Chapter 1",
            "chapter_summary": chapter_summary,
            "chunks": chunk_cards
        }
        
        sidecar_path = settings.SIDECAR_DIR / f"{file_path.stem}_summaries.json"
        with open(sidecar_path, 'w') as f:
            json.dump(sidecar_data, f, indent=2)
        
        logger.info(f"Created sidecar file: {sidecar_path.name}")
        
        # Save index and commit database changes
        self._save_index()
        self.db_conn.commit()
        
        logger.info(f"Successfully processed {file_path.name}")
    
    def ingest_all_books(self) -> None:
        """Process all books in the data directory."""
        pdf_files = sorted(settings.DATA_DIR.glob("*.pdf"))
        epub_files = sorted(settings.DATA_DIR.glob("*.epub"))
        
        all_files = pdf_files + epub_files
        
        if not all_files:
            logger.warning("No PDF or EPUB files found in data directory")
            return
        
        logger.info(f"Found {len(all_files)} files to process")
        
        for file_path in all_files:
            try:
                self.process_file(file_path)
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                continue
        
        logger.info("Book ingestion completed")
    
    def close(self) -> None:
        """Close database connection."""
        self.db_conn.close()


def main():
    """Main entry point for book ingestion."""
    ingestor = BookerIngestor()
    try:
        ingestor.ingest_all_books()
    finally:
        ingestor.close()


if __name__ == "__main__":
    main() 