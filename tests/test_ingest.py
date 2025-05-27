"""
Tests for the book ingestion functionality.
"""

import json
import tempfile
import unittest.mock
from pathlib import Path

import pytest
import duckdb
import faiss

from booker.ingest_book import BookerIngestor
from booker import settings


class TestBookIngestion:
    """Test cases for book ingestion functionality."""
    
    @pytest.fixture
    def mock_openai_embedding(self):
        """Mock OpenAI embedding API calls."""
        with unittest.mock.patch('openai.embeddings.create') as mock_create:
            # Return a mock embedding response
            mock_response = unittest.mock.MagicMock()
            mock_response.data = [
                unittest.mock.MagicMock(embedding=[0.1] * 3072)
            ]
            mock_create.return_value = mock_response
            yield mock_create
    
    @pytest.fixture
    def mock_openai_chat(self):
        """Mock OpenAI chat completion API calls."""
        with unittest.mock.patch('openai.chat.completions.create') as mock_create:
            # Return a mock chat response
            mock_response = unittest.mock.MagicMock()
            mock_response.choices = [
                unittest.mock.MagicMock(
                    message=unittest.mock.MagicMock(
                        content="This is a test summary of the book excerpt."
                    )
                )
            ]
            mock_create.return_value = mock_response
            yield mock_create
    
    @pytest.fixture
    def temp_directories(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test directories
            data_dir = temp_path / "data"
            db_dir = temp_path / "db"
            index_dir = temp_path / "indexes"
            sidecar_dir = temp_path / "sidecars"
            
            for dir_path in [data_dir, db_dir, index_dir, sidecar_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Mock settings paths
            with unittest.mock.patch.multiple(
                settings,
                DATA_DIR=data_dir,
                DB_PATH=db_dir / "booker.db",
                INDEX_PATH=index_dir / "booker.faiss",
                INDEX_META_PATH=index_dir / "booker.pkl",
                SIDECAR_DIR=sidecar_dir
            ):
                yield {
                    "data_dir": data_dir,
                    "db_path": db_dir / "booker.db",
                    "index_path": index_dir / "booker.faiss",
                    "index_meta_path": index_dir / "booker.pkl",
                    "sidecar_dir": sidecar_dir
                }
    
    def create_sample_pdf_content(self, data_dir: Path) -> Path:
        """Create a sample PDF file for testing."""
        # Create a simple text file that simulates PDF content
        sample_file = data_dir / "sample.pdf"
        
        # Since we're mocking PDF reading, we'll create a simple text file
        # and mock the PDF reader to return this content
        sample_content = """
        Chapter 1: Introduction to Testing
        
        This is a sample book about testing methodologies. Testing is crucial
        for software development and ensures that applications work as expected.
        
        There are many types of testing including unit testing, integration testing,
        and end-to-end testing. Each type serves a different purpose in the
        software development lifecycle.
        
        Unit testing focuses on testing individual components or functions in isolation.
        Integration testing verifies that different components work together correctly.
        End-to-end testing validates the entire application workflow from start to finish.
        """
        
        # Create an empty PDF file (content will be mocked)
        sample_file.write_text(sample_content)
        return sample_file
    
    def test_ingest_sample_pdf(self, mock_openai_embedding, mock_openai_chat, temp_directories):
        """Test ingesting a sample PDF file."""
        # Create sample PDF
        sample_pdf = self.create_sample_pdf_content(temp_directories["data_dir"])
        
        # Mock PDF reading
        sample_text = """
        Chapter 1: Introduction to Testing
        
        This is a sample book about testing methodologies. Testing is crucial
        for software development and ensures that applications work as expected.
        
        There are many types of testing including unit testing, integration testing,
        and end-to-end testing. Each type serves a different purpose in the
        software development lifecycle.
        """
        
        with unittest.mock.patch('booker.ingest_book.PdfReader') as mock_pdf_reader:
            # Mock PDF reader to return our sample text
            mock_page = unittest.mock.MagicMock()
            mock_page.extract_text.return_value = sample_text
            mock_reader = unittest.mock.MagicMock()
            mock_reader.pages = [mock_page]
            mock_pdf_reader.return_value = mock_reader
            
            # Create ingestor and process file
            ingestor = BookerIngestor()
            ingestor.process_file(sample_pdf)
            ingestor.close()
        
        # Verify FAISS index was created
        assert temp_directories["index_path"].exists()
        assert temp_directories["index_meta_path"].exists()
        
        # Verify database was created and populated
        assert temp_directories["db_path"].exists()
        
        # Check database content
        conn = duckdb.connect(str(temp_directories["db_path"]))
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        summaries = conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
        conn.close()
        
        assert chunks > 0, "No chunks were created"
        assert summaries > 0, "No summaries were created"
        assert chunks == summaries, "Mismatch between chunks and summaries"
        
        # Verify sidecar JSON was created
        sidecar_files = list(temp_directories["sidecar_dir"].glob("*.json"))
        assert len(sidecar_files) == 1, "Sidecar file was not created"
        
        # Verify sidecar content
        with open(sidecar_files[0]) as f:
            sidecar_data = json.load(f)
        
        assert "file" in sidecar_data
        assert "chapter_summary" in sidecar_data
        assert "chunks" in sidecar_data
        assert len(sidecar_data["chunks"]) > 0
        
        # Verify OpenAI API was called
        assert mock_openai_embedding.called, "OpenAI embedding API was not called"
        assert mock_openai_chat.called, "OpenAI chat API was not called"
    
    def test_ingest_all_books(self, mock_openai_embedding, mock_openai_chat, temp_directories):
        """Test ingesting all books in the data directory."""
        # Create multiple sample files
        sample_pdf1 = self.create_sample_pdf_content(temp_directories["data_dir"])
        sample_pdf2 = temp_directories["data_dir"] / "another_sample.pdf"
        sample_pdf2.write_text("Another sample book content for testing.")
        
        # Mock PDF reading for both files
        sample_text = "Sample book content for testing purposes."
        
        with unittest.mock.patch('booker.ingest_book.PdfReader') as mock_pdf_reader:
            # Mock PDF reader to return our sample text
            mock_page = unittest.mock.MagicMock()
            mock_page.extract_text.return_value = sample_text
            mock_reader = unittest.mock.MagicMock()
            mock_reader.pages = [mock_page]
            mock_pdf_reader.return_value = mock_reader
            
            # Create ingestor and process all files
            ingestor = BookerIngestor()
            ingestor.ingest_all_books()
            ingestor.close()
        
        # Verify multiple sidecar files were created
        sidecar_files = list(temp_directories["sidecar_dir"].glob("*.json"))
        assert len(sidecar_files) == 2, f"Expected 2 sidecar files, got {len(sidecar_files)}"
        
        # Verify database contains data from both files
        conn = duckdb.connect(str(temp_directories["db_path"]))
        unique_files = conn.execute("SELECT COUNT(DISTINCT file_name) FROM chunks").fetchone()[0]
        conn.close()
        
        assert unique_files == 2, f"Expected 2 unique files in database, got {unique_files}"
    
    def test_empty_data_directory(self, mock_openai_embedding, mock_openai_chat, temp_directories):
        """Test behavior when data directory is empty."""
        # Ensure data directory is empty
        for file in temp_directories["data_dir"].glob("*"):
            file.unlink()
        
        # Create ingestor and try to process
        ingestor = BookerIngestor()
        ingestor.ingest_all_books()  # Should handle empty directory gracefully
        ingestor.close()
        
        # Verify no sidecar files were created
        sidecar_files = list(temp_directories["sidecar_dir"].glob("*.json"))
        assert len(sidecar_files) == 0, "Sidecar files should not be created for empty directory"


if __name__ == "__main__":
    pytest.main([__file__]) 