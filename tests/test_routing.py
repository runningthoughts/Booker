"""
Tests for the intelligent routing functionality in booker.qa module.
"""

import json
import tempfile
import unittest.mock
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from booker.qa import load_book_meta, call_llm_router
from booker.models import BookMeta


class TestRouting:
    """Test cases for intelligent routing functionality."""
    
    def test_load_book_meta_not_exists(self):
        """Test loading metadata when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            book_id = "nonexistent_book"
            meta = load_book_meta(book_id, Path(temp_dir))
            assert meta is None
    
    def test_load_book_meta_exists(self):
        """Test loading metadata when file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            book_id = "test_book"
            
            # Create metadata file
            meta_data = {
                "title": "Test Book",
                "pub_year": 2020,
                "abstract": "A comprehensive guide to testing.",
                "topics": ["testing", "python", "software"],
                "not_covered": ["advanced topics"],
                "min_year": 2000,
                "max_year": 2023,
                "pages": 300
            }
            
            library_dir = Path(temp_dir) / book_id
            library_dir.mkdir(parents=True)
            
            meta_file = library_dir / "book_meta.json"
            with open(meta_file, "w") as f:
                json.dump(meta_data, f)
            
            # Load and verify
            meta = load_book_meta(book_id, Path(temp_dir))
            assert meta is not None
            assert isinstance(meta, dict)
            assert meta["title"] == "Test Book"
            assert meta["pub_year"] == 2020
            assert "testing" in meta["topics"]
    
    def test_call_llm_router_local(self):
        """Test LLM router returning LOCAL decision."""
        question = "What are Python functions?"
        top_chunks = [{"text": "Python functions are reusable blocks of code"}]
        meta = {
            "title": "Python Guide",
            "topics": ["python", "programming", "functions"],
            "not_covered": ["web frameworks"]
        }
        
        with unittest.mock.patch('openai.chat.completions.create') as mock_openai:
            mock_response = unittest.mock.MagicMock()
            mock_response.choices = [
                unittest.mock.MagicMock(
                    message=unittest.mock.MagicMock(content="LOCAL")
                )
            ]
            mock_openai.return_value = mock_response
            
            decision = call_llm_router(question, top_chunks, meta)
            assert decision == "LOCAL"
    
    def test_call_llm_router_global(self):
        """Test LLM router returning GLOBAL decision.""" 
        question = "What's the current weather?"
        top_chunks = [{"text": "Some book content about programming"}]
        meta = {
            "title": "Python Guide",
            "topics": ["python", "programming"],
            "not_covered": ["weather", "current events"]
        }
        
        with unittest.mock.patch('openai.chat.completions.create') as mock_openai:
            mock_response = unittest.mock.MagicMock()
            mock_response.choices = [
                unittest.mock.MagicMock(
                    message=unittest.mock.MagicMock(content="GLOBAL")
                )
            ]
            mock_openai.return_value = mock_response
            
            decision = call_llm_router(question, top_chunks, meta)
            assert decision == "GLOBAL"
    
    def test_call_llm_router_reject(self):
        """Test LLM router returning REJECT decision."""
        question = "What's your favorite color?"
        top_chunks = [{"text": "Some book content about programming"}]
        meta = {
            "title": "Python Guide", 
            "topics": ["python", "programming"],
            "not_covered": ["personal preferences"]
        }
        
        with unittest.mock.patch('openai.chat.completions.create') as mock_openai:
            mock_response = unittest.mock.MagicMock()
            mock_response.choices = [
                unittest.mock.MagicMock(
                    message=unittest.mock.MagicMock(content="REJECT")
                )
            ]
            mock_openai.return_value = mock_response
            
            decision = call_llm_router(question, top_chunks, meta)
            assert decision == "REJECT"
    
    def test_call_llm_router_fallback(self):
        """Test LLM router fallback when API fails."""
        question = "What are Python functions?"
        top_chunks = [{"text": "Python functions are reusable blocks of code"}]
        meta = {
            "title": "Python Guide",
            "topics": ["python", "programming"],
            "not_covered": []
        }
        
        with unittest.mock.patch('openai.chat.completions.create') as mock_openai:
            mock_openai.side_effect = Exception("OpenAI API failed")
            
            decision = call_llm_router(question, top_chunks, meta)
            assert decision == "LOCAL"  # Should fallback to LOCAL
    
    def test_book_meta_model_validation(self):
        """Test BookMeta model validation."""
        # Valid data
        valid_data = {
            "title": "Test Book",
            "pub_year": 2020,
            "abstract": "A comprehensive guide to testing.",
            "topics": ["testing", "python", "software"],
            "not_covered": ["advanced topics"],
            "min_year": 2000,
            "max_year": 2023,
            "pages": 300
        }
        
        book_meta = BookMeta(**valid_data)
        assert book_meta.title == "Test Book"
        assert book_meta.pub_year == 2020
        assert len(book_meta.topics) == 3
        assert book_meta.pages == 300
        
        # Test JSON serialization
        json_str = book_meta.model_dump_json()
        loaded_meta = BookMeta.model_validate_json(json_str)
        assert loaded_meta.title == book_meta.title 