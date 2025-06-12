"""
Book ingestion module for processing PDFs and EPUBs into searchable chunks.
Supports both individual books and hierarchical "bodies of work".
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

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


class SourceType(Enum):
    """Enumeration for different source material types."""
    BOOK = "book"  # Individual files directly in source directory
    BODY_OF_WORK = "body_of_work"  # Numbered subdirectories with hierarchy


class ImportanceLevel(Enum):
    """Enumeration for content importance levels."""
    PRIMARY = 0      # Main content (from 0/ directory)
    SECONDARY = 1    # Supporting material (from 1/ directory)
    TERTIARY = 2     # Additional supporting material (from 2/ directory)
    QUATERNARY = 3   # Further supporting material (from 3/ directory)


class BookerIngestor:
    """Handles the ingestion of books and bodies of work into the Booker system."""
    
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
        """Create database tables if they don't exist and migrate existing ones."""
        # Create tables if they don't exist
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
                heading_type  TEXT,
                source_type   TEXT,
                importance_level INTEGER,
                source_directory TEXT
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
        
        # Migrate existing chunks table if needed
        try:
            # Check if new columns exist
            result = self.db_conn.execute("PRAGMA table_info(chunks)").fetchall()
            existing_columns = {row[1] for row in result}  # row[1] is column name
            
            # Add missing columns
            if 'source_type' not in existing_columns:
                self.db_conn.execute("ALTER TABLE chunks ADD COLUMN source_type TEXT DEFAULT 'book'")
                logger.info("Added source_type column to chunks table")
            
            if 'importance_level' not in existing_columns:
                self.db_conn.execute("ALTER TABLE chunks ADD COLUMN importance_level INTEGER DEFAULT 0")
                logger.info("Added importance_level column to chunks table")
            
            if 'source_directory' not in existing_columns:
                self.db_conn.execute("ALTER TABLE chunks ADD COLUMN source_directory TEXT DEFAULT ''")
                logger.info("Added source_directory column to chunks table")
                
        except Exception as e:
            logger.warning(f"Error during database migration: {e}")
        
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
    
    def _detect_source_type(self, source_dir: Path) -> SourceType:
        """Detect whether source is a book or body of work format."""
        # Check for numbered subdirectories (0, 1, 2, etc.)
        numbered_dirs = []
        for item in source_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                numbered_dirs.append(int(item.name))
        
        # If we have a 0 directory, it's a body of work
        if 0 in numbered_dirs:
            logger.info(f"Detected body of work format with directories: {sorted(numbered_dirs)}")
            return SourceType.BODY_OF_WORK
        else:
            # Look for files directly in source directory
            pdf_files = list(source_dir.glob("*.pdf"))
            epub_files = list(source_dir.glob("*.epub"))
            if pdf_files or epub_files:
                logger.info(f"Detected book format with {len(pdf_files)} PDFs and {len(epub_files)} EPUBs")
                return SourceType.BOOK
            else:
                logger.warning("No recognizable content found in source directory")
                return SourceType.BOOK  # Default fallback
    
    def _get_importance_level(self, directory_number: int) -> ImportanceLevel:
        """Map directory number to importance level."""
        importance_map = {
            0: ImportanceLevel.PRIMARY,
            1: ImportanceLevel.SECONDARY,
            2: ImportanceLevel.TERTIARY,
            3: ImportanceLevel.QUATERNARY
        }
        return importance_map.get(directory_number, ImportanceLevel.QUATERNARY)
    
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
    
    def _summarize_chunk(self, text: str, importance_level: ImportanceLevel) -> str:
        """Generate a summary for a text chunk, considering its importance level."""
        importance_context = {
            ImportanceLevel.PRIMARY: "primary source material",
            ImportanceLevel.SECONDARY: "supporting material", 
            ImportanceLevel.TERTIARY: "additional supporting material",
            ImportanceLevel.QUATERNARY: "supplementary material"
        }
        
        context = importance_context.get(importance_level, "source material")
        
        response = openai.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"Summarise this excerpt from {context} in ≤2 sentences, dense with key info, no new facts."
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
    
    def _generate_chapter_summary(self, recaps: List[str], importance_level: ImportanceLevel) -> str:
        """Generate a chapter summary from individual chunk recaps."""
        combined_recaps = " ".join(recaps)
        
        # Trim to 2000 tokens if needed
        tokens = self.tokenizer.encode(combined_recaps)
        if len(tokens) > 2000:
            tokens = tokens[:2000]
            combined_recaps = self.tokenizer.decode(tokens)
        
        importance_context = {
            ImportanceLevel.PRIMARY: "primary source material",
            ImportanceLevel.SECONDARY: "supporting material",
            ImportanceLevel.TERTIARY: "additional supporting material", 
            ImportanceLevel.QUATERNARY: "supplementary material"
        }
        
        context = importance_context.get(importance_level, "source material")
        
        response = openai.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"Summarise the following bullet-point recaps of {context} in one coherent paragraph, ≤100 tokens."
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
    
    def process_file(self, file_path: Path, source_type: SourceType, 
                    importance_level: ImportanceLevel = ImportanceLevel.PRIMARY, 
                    source_directory: str = "") -> List[Dict[str, Any]]:
        """Process a single book file (PDF or EPUB) and return chunk cards."""
        logger.info(f"Processing file: {file_path.name} (importance: {importance_level.name})")
        
        # Extract text based on file type
        if file_path.suffix.lower() == '.pdf':
            text = self._extract_text_from_pdf(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path.suffix}")
            return []
        
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
                                      page_start, page_end, text, heading, heading_level, heading_type,
                                      source_type, importance_level, source_directory)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (chunk_id, file_path.parent.parent.name, file_path.name, 1, "Chapter 1", 
                     chunk_idx * 10, (chunk_idx + 1) * 10, doc.text,
                     doc.metadata.get('heading'), doc.metadata.get('heading_level'), 
                     doc.metadata.get('heading_type'), source_type.value, importance_level.value,
                     source_directory))
                
                # Generate summary
                recap = self._summarize_chunk(doc.text, importance_level)
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
                    "heading_type": doc.metadata.get('heading_type'),
                    "source_type": source_type.value,
                    "importance_level": importance_level.value,
                    "importance_name": importance_level.name,
                    "source_directory": source_directory,
                    "file_name": file_path.name
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
                    "heading_type": doc.metadata.get('heading_type'),
                    "source_type": source_type.value,
                    "importance_level": importance_level.value,
                    "source_directory": source_directory
                })
        
        return chunk_cards, recaps
    
    def process_body_of_work(self, source_dir: Path, work_id: str) -> None:
        """Process a hierarchical body of work with numbered subdirectories."""
        logger.info(f"Processing body of work: {work_id}")
        
        # Find all numbered subdirectories
        numbered_dirs = []
        for item in source_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                numbered_dirs.append((int(item.name), item))
        
        # Sort by number (importance order)
        numbered_dirs.sort(key=lambda x: x[0])
        
        all_chunk_cards = {}  # Organized by directory
        all_recaps = {}
        
        for dir_num, dir_path in numbered_dirs:
            importance_level = self._get_importance_level(dir_num)
            logger.info(f"Processing directory {dir_num}/ with importance level: {importance_level.name}")
            
            # Find files in this directory
            pdf_files = sorted(dir_path.glob("*.pdf"))
            epub_files = sorted(dir_path.glob("*.epub"))
            all_files = pdf_files + epub_files
            
            dir_chunk_cards = []
            dir_recaps = []
            
            for file_path in all_files:
                try:
                    file_cards, file_recaps = self.process_file(
                        file_path, 
                        SourceType.BODY_OF_WORK, 
                        importance_level,
                        f"{dir_num}/"
                    )
                    dir_chunk_cards.extend(file_cards)
                    dir_recaps.extend(file_recaps)
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {e}")
                    continue
            
            if dir_chunk_cards:
                all_chunk_cards[dir_num] = dir_chunk_cards
                all_recaps[dir_num] = dir_recaps
        
        # Generate hierarchical sidecar
        sidecar_data = {
            "work_id": work_id,
            "source_type": SourceType.BODY_OF_WORK.value,
            "structure": "hierarchical",
            "directories": {}
        }
        
        # Add directory-level summaries and chunks
        for dir_num in sorted(all_chunk_cards.keys()):
            importance_level = self._get_importance_level(dir_num)
            
            # Generate directory summary
            if all_recaps[dir_num]:
                dir_summary = self._generate_chapter_summary(all_recaps[dir_num], importance_level)
            else:
                dir_summary = f"No content processed for directory {dir_num}/"
            
            sidecar_data["directories"][str(dir_num)] = {
                "directory": f"{dir_num}/",
                "importance_level": importance_level.value,
                "importance_name": importance_level.name,
                "description": {
                    0: "Primary source material - main content",
                    1: "Supporting material - secondary importance", 
                    2: "Additional supporting material - tertiary importance",
                    3: "Supplementary material - quaternary importance"
                }.get(dir_num, "Additional material"),
                "directory_summary": dir_summary,
                "chunk_count": len(all_chunk_cards[dir_num]),
                "chunks": all_chunk_cards[dir_num]
            }
        
        # Generate overall work summary focusing on primary content
        if 0 in all_recaps and all_recaps[0]:
            work_summary = self._generate_chapter_summary(all_recaps[0], ImportanceLevel.PRIMARY)
        else:
            # Fallback to first available directory
            first_dir = min(all_recaps.keys()) if all_recaps else None
            if first_dir is not None:
                work_summary = self._generate_chapter_summary(
                    all_recaps[first_dir], 
                    self._get_importance_level(first_dir)
                )
            else:
                work_summary = "No content available for summary"
        
        sidecar_data["work_summary"] = work_summary
        sidecar_data["total_chunks"] = sum(len(cards) for cards in all_chunk_cards.values())
        
        # Save hierarchical sidecar
        sidecar_path = self.sidecar_dir / f"{work_id}_body_of_work.json"
        with open(sidecar_path, 'w') as f:
            json.dump(sidecar_data, f, indent=2)
        
        logger.info(f"Created hierarchical sidecar file: {sidecar_path.name}")
        
        # Save index and commit database changes
        self._save_index()
        self.db_conn.commit()
        
        logger.info(f"Successfully processed body of work: {work_id}")
    
    def process_book(self, source_dir: Path, book_id: str) -> None:
        """Process a traditional book with files directly in source directory."""
        logger.info(f"Processing book: {book_id}")
        
        pdf_files = sorted(source_dir.glob("*.pdf"))
        epub_files = sorted(source_dir.glob("*.epub"))
        all_files = pdf_files + epub_files
        
        if not all_files:
            logger.warning("No PDF or EPUB files found in source directory")
            return
        
        all_chunk_cards = []
        all_recaps = []
        
        for file_path in all_files:
            try:
                file_cards, file_recaps = self.process_file(
                    file_path, 
                    SourceType.BOOK, 
                    ImportanceLevel.PRIMARY,
                    ""
                )
                all_chunk_cards.extend(file_cards)
                all_recaps.extend(file_recaps)
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                continue
        
        # Generate book summary
        if all_recaps:
            book_summary = self._generate_chapter_summary(all_recaps, ImportanceLevel.PRIMARY)
        else:
            book_summary = "No content available for summary"
        
        # Create traditional sidecar
        sidecar_data = {
            "book_id": book_id,
            "source_type": SourceType.BOOK.value,
            "structure": "flat",
            "chapter_no": 1,
            "chapter_title": "Chapter 1",
            "chapter_summary": book_summary,
            "total_chunks": len(all_chunk_cards),
            "chunks": all_chunk_cards
        }
        
        sidecar_path = self.sidecar_dir / f"{book_id}_summaries.json"
        with open(sidecar_path, 'w') as f:
            json.dump(sidecar_data, f, indent=2)
        
        logger.info(f"Created sidecar file: {sidecar_path.name}")
        
        # Save index and commit database changes
        self._save_index()
        self.db_conn.commit()
        
        logger.info(f"Successfully processed book: {book_id}")
    
    def ingest_all_content(self, source_dir: Path, content_id: str) -> None:
        """Process all content in the source directory, auto-detecting format."""
        source_type = self._detect_source_type(source_dir)
        
        if source_type == SourceType.BODY_OF_WORK:
            self.process_body_of_work(source_dir, content_id)
        else:
            self.process_book(source_dir, content_id)
    
    def close(self) -> None:
        """Close database connection."""
        self.db_conn.close()


def main():
    """Main entry point for content ingestion."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--books-root", default=None,
                        help="Parent folder where all publications live (default: library)")
    parser.add_argument("--book-id", required=True,
                        help="Folder name of the publication to ingest")
    parser.add_argument("--profile", action="store_true", 
                        help="Generate book metadata profile after ingestion")
    parser.add_argument("--llm-summary", action="store_true",
                        help="Use LLM for book profiling summary (otherwise use first 500 chars)")
    args = parser.parse_args()

    content_id = args.book_id  # Can be book or body of work
    
    # Determine paths - handle both old and new utility system
    if args.books_root:
        books_root = Path(args.books_root)
        src_dir = books_root / content_id / "source"
        base_dir = books_root / content_id / "build"
    else:
        # Try to use new utility functions if available
        try:
            from .utils import get_book_source_path, get_book_build_path
            src_dir = get_book_source_path(content_id)
            base_dir = get_book_build_path(content_id)
        except ImportError:
            # Fallback to old behavior
            src_dir = BOOKS_ROOT / content_id / "source"
            base_dir = BOOKS_ROOT / content_id / "build"
    
    data_dir = src_dir
    db_path = base_dir / "db" / "booker.db"
    index_dir = base_dir / "indexes"
    sidecar_dir = base_dir / "sidecars"
    
    # Ensure directories exist
    for p in (base_dir, db_path.parent, index_dir, sidecar_dir):
        p.mkdir(parents=True, exist_ok=True)

    ingestor = BookerIngestor(db_path, index_dir, sidecar_dir)
    try:
        ingestor.ingest_all_content(data_dir, content_id)
    finally:
        ingestor.close()
    
    # Run profiling if requested
    if args.profile:
        import subprocess
        import sys
        import pathlib
        
        print("Running book profiling...")
        try:
            cmd = [
                sys.executable,
                str(pathlib.Path(__file__).parent.parent / "scripts" / "profile_book.py"),
                "--book-id", content_id,
                "--db-path", str(db_path)
            ]
            
            # Add no-LLM flag for speed (unless user specifically requested LLM summary)
            if not args.llm_summary:
                cmd.append("--no-llm-summary")
            
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            logger.error(f"Book profiling failed: {e}")
        except Exception as e:
            logger.error(f"Error running book profiling: {e}")


if __name__ == "__main__":
    main() 