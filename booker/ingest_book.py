"""
Book ingestion module for processing PDFs and EPUBs into searchable chunks.
"""

import argparse
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
from .settings import BOOKS_ROOT
from .splitters import HeadingAwareTextSplitter

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
    
    def __init__(self, db_path: Path, index_dir: Path, sidecar_dir: Path):
        """Initialize the ingestor with tokenizer and database connection."""
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.text_splitter = HeadingAwareTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.db_path = db_path
        self.index_path = index_dir / "booker.faiss"
        self.index_meta_path = index_dir / "booker.pkl"
        self.sidecar_dir = sidecar_dir
        
        # Ensure directories exist
        for path in [self.db_path.parent, index_dir, sidecar_dir]:
            path.mkdir(parents=True, exist_ok=True)
        
        self.db_conn = duckdb.connect(str(self.db_path))
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
                text          TEXT,
                heading       TEXT,
                heading_level INTEGER,
                heading_type  TEXT
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
        if self.index_path.exists() and self.index_meta_path.exists():
            self.faiss_index = faiss.read_index(str(self.index_path))
            with open(self.index_meta_path, 'rb') as f:
                self.chunk_metadata = pickle.load(f)
            logger.info(f"Loaded existing FAISS index with {self.faiss_index.ntotal} vectors")
        else:
            # Create new index (3072 dimensions for text-embedding-3-large)
            self.faiss_index = faiss.IndexFlatIP(3072)
            self.chunk_metadata = []
            logger.info("Created new FAISS index")
    
    def _save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.faiss_index, str(self.index_path))
        with open(self.index_meta_path, 'wb') as f:
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
        
        # Chunk the text using heading-aware splitter
        documents = self.text_splitter.split_text(text)
        logger.info(f"Created {len(documents)} chunks")
        
        # Process chunks in batches
        chunk_cards = []
        recaps = []
        
        for i in range(0, len(documents), settings.BATCH_SIZE):
            batch_docs = documents[i:i + settings.BATCH_SIZE]
            batch_texts = [doc.text for doc in batch_docs]
            
            # Get embeddings for batch
            embeddings = self._get_embeddings(batch_texts)
            
            # Process each chunk in the batch
            for j, (doc, embedding) in enumerate(zip(batch_docs, embeddings)):
                chunk_idx = i + j
                
                # Add to FAISS index
                embedding_array = np.array([embedding], dtype=np.float32)
                faiss.normalize_L2(embedding_array)
                self.faiss_index.add(embedding_array)
                
                # Insert into database
                chunk_id = self.faiss_index.ntotal  # Use FAISS index as chunk_id
                self.db_conn.execute("""
                    INSERT INTO chunks (chunk_id, book_id, file_name, chapter_no, chapter_title, 
                                      page_start, page_end, text, heading, heading_level, heading_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (chunk_id, file_path.stem, file_path.name, 1, "Chapter 1", 
                     chunk_idx * 10, (chunk_idx + 1) * 10, doc.text,
                     doc.metadata.get('heading'), doc.metadata.get('heading_level'), 
                     doc.metadata.get('heading_type')))
                
                # Generate summary
                recap = self._summarize_chunk(doc.text)
                recaps.append(recap)
                
                # Extract entities and keywords
                entities_data = self._extract_entities(doc.text)
                
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
                    "entities": entities_data["entities"],
                    "heading": doc.metadata.get('heading'),
                    "heading_level": doc.metadata.get('heading_level'),
                    "heading_type": doc.metadata.get('heading_type')
                }
                chunk_cards.append(chunk_card)
                
                # Update metadata
                self.chunk_metadata.append({
                    "chunk_id": chunk_id,
                    "file_name": file_path.name,
                    "page_start": chunk_idx * 10,
                    "page_end": (chunk_idx + 1) * 10,
                    "heading": doc.metadata.get('heading'),
                    "heading_level": doc.metadata.get('heading_level'),
                    "heading_type": doc.metadata.get('heading_type')
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
        
        sidecar_path = self.sidecar_dir / f"{file_path.stem}_summaries.json"
        with open(sidecar_path, 'w') as f:
            json.dump(sidecar_data, f, indent=2)
        
        logger.info(f"Created sidecar file: {sidecar_path.name}")
        
        # Save index and commit database changes
        self._save_index()
        self.db_conn.commit()
        
        logger.info(f"Successfully processed {file_path.name}")
    
    def ingest_all_books(self, data_dir: Path) -> None:
        """Process all books in the data directory."""
        pdf_files = sorted(data_dir.glob("*.pdf"))
        epub_files = sorted(data_dir.glob("*.epub"))
        
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--books-root", default=None,
                        help="Parent folder where all publications live (default: library)")
    parser.add_argument("--book-id", required=True,
                        help="Folder name of the publication to ingest")
    args = parser.parse_args()

    books_root = Path(args.books_root) if args.books_root else BOOKS_ROOT
    book_id    = args.book_id
    src_dir    = books_root / book_id / "source"
    base_dir   = books_root / book_id / "build"
    data_dir   = src_dir
    db_path    = base_dir / "db" / "booker.db"
    index_dir  = base_dir / "indexes"
    sidecar_dir= base_dir / "sidecars"
    for p in (base_dir, db_path.parent, index_dir, sidecar_dir):
        p.mkdir(parents=True, exist_ok=True)

    ingestor = BookerIngestor(db_path, index_dir, sidecar_dir)
    try:
        ingestor.ingest_all_books(data_dir)
    finally:
        ingestor.close()


if __name__ == "__main__":
    main() 