"""
Tests for the profile_book.py script functionality.
"""

import json
import tempfile
import unittest.mock
from pathlib import Path

import pytest
import duckdb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.profile_book import (
    extract_years_from_text, 
    extract_topics_with_keybert,
    generate_llm_summary,
    get_book_chunks,
    estimate_page_count,
    profile_book
)
from booker.models import BookMeta


class TestProfileBook:
    """Test cases for book profiling functionality."""
    
    def test_extract_years_from_text(self):
        """Test year extraction from text."""
        text = "The algorithm was developed in 1995 and improved in 2010. It became popular around 2020."
        years = extract_years_from_text(text)
        
        assert 1995 in years
        assert 2010 in years  
        assert 2020 in years
        assert len(years) == 3
    
    def test_extract_years_edge_cases(self):
        """Test year extraction edge cases."""
        # No years
        assert extract_years_from_text("No years here") == []
        
        # Years outside reasonable range
        text_with_old_years = "The year 999 and 3000 should not be extracted"
        years = extract_years_from_text(text_with_old_years)
        assert 999 not in years
        assert 3000 not in years
        
        # Years in different contexts
        text_with_context = "In 1850, the technique was first used. By 2025, it will be obsolete."
        years = extract_years_from_text(text_with_context)
        assert 1850 in years
        assert 2025 in years
    
    @pytest.fixture
    def mock_keybert(self):
        """Mock KeyBERT for testing."""
        with unittest.mock.patch('scripts.profile_book.kw_model') as mock_kw:
            mock_kw.extract_keywords.return_value = [
                ("machine learning", 0.8),
                ("neural networks", 0.7),
                ("deep learning", 0.6),
                ("artificial intelligence", 0.5)
            ]
            yield mock_kw
    
    def test_extract_topics_with_keybert(self, mock_keybert):
        """Test topic extraction using KeyBERT."""
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are used in deep learning applications."
        ]
        
        topics = extract_topics_with_keybert(texts, top_k=4)
        
        assert "machine learning" in topics
        assert "neural networks" in topics
        assert "deep learning" in topics
        assert "artificial intelligence" in topics
        assert len(topics) == 4
    
    def test_extract_topics_with_keybert_failure(self, mock_keybert):
        """Test topic extraction when KeyBERT fails."""
        mock_keybert.extract_keywords.side_effect = Exception("KeyBERT failed")
        
        texts = ["Some text here"]
        topics = extract_topics_with_keybert(texts)
        
        assert topics == []
    
    @pytest.fixture
    def mock_openai_summary(self):
        """Mock OpenAI for LLM summary generation."""
        with unittest.mock.patch('openai.chat.completions.create') as mock_create:
            mock_response = unittest.mock.MagicMock()
            mock_response.choices = [
                unittest.mock.MagicMock(
                    message=unittest.mock.MagicMock(
                        content="This book provides a comprehensive introduction to machine learning, "
                               "covering algorithms, neural networks, and practical applications in data science."
                    )
                )
            ]
            mock_create.return_value = mock_response
            yield mock_create
    
    def test_generate_llm_summary(self, mock_openai_summary):
        """Test LLM summary generation."""
        text = "Machine learning is a powerful tool. " * 100  # Long text
        summary = generate_llm_summary(text, max_words=20)
        
        assert "machine learning" in summary.lower()
        assert "comprehensive" in summary.lower()
        assert len(summary) > 50  # Reasonable summary length
    
    def test_generate_llm_summary_failure(self, mock_openai_summary):
        """Test LLM summary generation when API fails."""
        mock_openai_summary.side_effect = Exception("OpenAI API failed")
        
        text = "This is some text to summarize."
        summary = generate_llm_summary(text)
        
        # Should fallback to truncated text
        assert summary == text[:500] + "..." if len(text) > 500 else text
    
    @pytest.fixture
    def setup_test_database(self):
        """Set up test database with sample book data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            
            conn = duckdb.connect(str(db_path))
            
            # Create chunks table
            conn.execute("""
                CREATE TABLE chunks (
                    chunk_id      INTEGER PRIMARY KEY,
                    book_id       TEXT,
                    file_name     TEXT,
                    page_start    INTEGER,
                    page_end      INTEGER,
                    text          TEXT,
                    chapter_no    INTEGER,
                    chapter_title TEXT,
                    heading       TEXT,
                    heading_level INTEGER
                )
            """)
            
            # Insert test data
            test_chunks = [
                (1, "test_book", "chapter1.pdf", 1, 20, 
                 "Machine learning was first developed in 1950. It has evolved significantly since then, "
                 "with major breakthroughs in 1980 and 2010. Today, it is used in various applications.",
                 1, "Introduction", "Machine Learning Basics", 1),
                (2, "test_book", "chapter2.pdf", 21, 40,
                 "Neural networks are computational models inspired by biological neural networks. "
                 "They became popular in the 1990s and saw a resurgence in 2012 with deep learning.",
                 2, "Neural Networks", "Deep Learning", 1),
                (3, "test_book", "chapter3.pdf", 41, 60,
                 "Support vector machines are powerful algorithms for classification and regression. "
                 "They were developed in 1995 and remain popular for many applications today.",
                 3, "Algorithms", "SVM", 1),
                (4, "test_book", "chapter4.pdf", 61, 80,
                 "The future of AI looks promising, with developments expected through 2030 and beyond.",
                 4, "Future", "AI Future", 1)
            ]
            
            for chunk in test_chunks:
                conn.execute("""
                    INSERT INTO chunks (chunk_id, book_id, file_name, page_start, page_end, text, 
                                      chapter_no, chapter_title, heading, heading_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, chunk)
            
            conn.commit()
            conn.close()
            
            yield db_path
    
    def test_get_book_chunks(self, setup_test_database):
        """Test retrieving book chunks from database."""
        chunks = get_book_chunks(setup_test_database, "test_book")
        
        assert len(chunks) == 4
        assert chunks[0]["book_id"] == "test_book"
        assert chunks[0]["file_name"] == "chapter1.pdf"
        assert "machine learning" in chunks[0]["text"].lower()
    
    def test_get_book_chunks_nonexistent(self, setup_test_database):
        """Test retrieving chunks for non-existent book."""
        chunks = get_book_chunks(setup_test_database, "nonexistent_book")
        assert chunks == []
    
    def test_estimate_page_count(self):
        """Test page count estimation."""
        chunks = [
            {"page_end": 20},
            {"page_end": 40},
            {"page_end": 60},
            {"page_end": 80}
        ]
        
        page_count = estimate_page_count(chunks)
        assert page_count == 80
    
    def test_estimate_page_count_empty(self):
        """Test page count estimation with empty chunks."""
        assert estimate_page_count([]) is None
        assert estimate_page_count([{"page_end": None}]) is None
    
    def test_profile_book_integration(self, setup_test_database, mock_keybert, mock_openai_summary):
        """Test complete book profiling integration."""
        # Test with LLM summary
        book_meta = profile_book("test_book", setup_test_database, use_llm_summary=True)
        
        assert isinstance(book_meta, BookMeta)
        assert book_meta.title == "Test Book"
        assert book_meta.min_year == 1950
        assert book_meta.max_year == 2030
        assert book_meta.pages == 80
        assert len(book_meta.topics) == 4
        assert "comprehensive" in book_meta.abstract.lower()
    
    def test_profile_book_no_llm_summary(self, setup_test_database, mock_keybert):
        """Test book profiling without LLM summary."""
        book_meta = profile_book("test_book", setup_test_database, use_llm_summary=False)
        
        assert isinstance(book_meta, BookMeta)
        assert book_meta.title == "Test Book"
        assert len(book_meta.abstract) <= 500
        assert "machine learning" in book_meta.abstract.lower()
    
    def test_profile_book_no_chunks(self, setup_test_database):
        """Test profiling when no chunks exist."""
        with pytest.raises(ValueError, match="No chunks found"):
            profile_book("nonexistent_book", setup_test_database)
    
    def test_book_meta_json_serialization(self, setup_test_database, mock_keybert, mock_openai_summary):
        """Test that BookMeta can be serialized to JSON."""
        book_meta = profile_book("test_book", setup_test_database, use_llm_summary=True)
        
        # Convert to dict and serialize
        meta_dict = book_meta.model_dump()
        json_str = json.dumps(meta_dict)
        
        # Deserialize and verify
        loaded_dict = json.loads(json_str)
        loaded_meta = BookMeta(**loaded_dict)
        
        assert loaded_meta.title == book_meta.title
        assert loaded_meta.topics == book_meta.topics
        assert loaded_meta.min_year == book_meta.min_year
        assert loaded_meta.max_year == book_meta.max_year 