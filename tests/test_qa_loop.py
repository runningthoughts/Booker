"""
Tests for the question-answering functionality.
"""

import tempfile
import unittest.mock
from pathlib import Path

import pytest
import duckdb
import faiss
import numpy as np

from booker.qa import answer_question
from booker.retriever import BookerRetriever
from booker import settings


class TestQALoop:
    """Test cases for question-answering functionality."""
    
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
            # Return a mock chat response with expected content
            mock_response = unittest.mock.MagicMock()
            mock_response.choices = [
                unittest.mock.MagicMock(
                    message=unittest.mock.MagicMock(
                        content="Based on the provided excerpts, testing is crucial for software development. "
                               "Unit testing focuses on individual components [Source 1], while integration "
                               "testing verifies that components work together [Source 2]."
                    )
                )
            ]
            mock_create.return_value = mock_response
            yield mock_create
    
    @pytest.fixture
    def setup_test_data(self):
        """Set up test database and FAISS index with sample data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test directories
            db_dir = temp_path / "db"
            index_dir = temp_path / "indexes"
            
            for dir_path in [db_dir, index_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            db_path = db_dir / "booker.db"
            index_path = index_dir / "booker.faiss"
            index_meta_path = index_dir / "booker.pkl"
            
            # Create and populate test database
            conn = duckdb.connect(str(db_path))
            
            # Create tables
            conn.execute("""
                CREATE TABLE chunks (
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
            
            conn.execute("""
                CREATE TABLE summaries (
                    chunk_id   INTEGER REFERENCES chunks(chunk_id),
                    summary    TEXT,
                    keywords   TEXT,
                    entities   TEXT
                )
            """)
            
            # Insert sample data
            sample_chunks = [
                (1, "testing_book", "testing_guide.pdf", 1, "Introduction", 1, 10,
                 "Unit testing is a software testing method where individual components are tested in isolation. "
                 "It helps ensure that each part of the application works correctly on its own."),
                (2, "testing_book", "testing_guide.pdf", 1, "Introduction", 11, 20,
                 "Integration testing verifies that different components of the system work together correctly. "
                 "It tests the interfaces and interaction between integrated components."),
                (3, "testing_book", "testing_guide.pdf", 2, "Advanced Topics", 21, 30,
                 "End-to-end testing validates the entire application workflow from start to finish. "
                 "It simulates real user scenarios and tests the complete system.")
            ]
            
            for chunk in sample_chunks:
                conn.execute("""
                    INSERT INTO chunks (chunk_id, book_id, file_name, chapter_no, chapter_title, 
                                      page_start, page_end, text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, chunk)
            
            # Insert sample summaries
            sample_summaries = [
                (1, "Unit testing tests individual components in isolation.", 
                 "unit testing components isolation", '[]'),
                (2, "Integration testing verifies component interactions.", 
                 "integration testing components interfaces", '[]'),
                (3, "End-to-end testing validates complete workflows.", 
                 "end-to-end testing workflows validation", '[]')
            ]
            
            for summary in sample_summaries:
                conn.execute("""
                    INSERT INTO summaries (chunk_id, summary, keywords, entities)
                    VALUES (?, ?, ?, ?)
                """, summary)
            
            conn.commit()
            conn.close()
            
            # Create test FAISS index
            index = faiss.IndexFlatIP(3072)
            
            # Add sample embeddings (normalized random vectors)
            for i in range(3):
                embedding = np.random.rand(3072).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)  # Normalize
                index.add(embedding.reshape(1, -1))
            
            # Save FAISS index
            faiss.write_index(index, str(index_path))
            
            # Create metadata
            import pickle
            metadata = [
                {"chunk_id": 1, "embedding_index": 0, "file_name": "testing_guide.pdf", 
                 "page_start": 1, "page_end": 10},
                {"chunk_id": 2, "embedding_index": 1, "file_name": "testing_guide.pdf", 
                 "page_start": 11, "page_end": 20},
                {"chunk_id": 3, "embedding_index": 2, "file_name": "testing_guide.pdf", 
                 "page_start": 21, "page_end": 30}
            ]
            
            with open(index_meta_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Mock settings paths
            with unittest.mock.patch.multiple(
                settings,
                DB_PATH=db_path,
                INDEX_PATH=index_path,
                INDEX_META_PATH=index_meta_path
            ):
                yield {
                    "db_path": db_path,
                    "index_path": index_path,
                    "index_meta_path": index_meta_path
                }
    
    def test_answer_known_question(self, mock_openai_embedding, mock_openai_chat, setup_test_data):
        """Test answering a question about a known topic."""
        question = "What is unit testing?"
        
        # Mock FAISS search to return relevant chunks
        with unittest.mock.patch('faiss.IndexFlatIP.search') as mock_search:
            # Return indices that correspond to our test data
            mock_search.return_value = (
                np.array([[0.9, 0.8, 0.7]]),  # scores
                np.array([[0, 1, 2]])         # indices
            )
            
            # Mock FAISS reconstruct to return embeddings
            with unittest.mock.patch('faiss.IndexFlatIP.reconstruct') as mock_reconstruct:
                mock_reconstruct.return_value = np.random.rand(3072).astype(np.float32)
                
                result = answer_question(question, k=3)
        
        # Verify the result structure
        assert "answer" in result
        assert "sources" in result
        assert isinstance(result["sources"], list)
        
        # Verify the answer contains expected content
        answer = result["answer"]
        assert "testing" in answer.lower()
        assert "components" in answer.lower() or "individual" in answer.lower()
        
        # Verify sources are provided
        sources = result["sources"]
        assert len(sources) > 0
        
        # Check source structure
        for source in sources:
            assert "source_id" in source
            assert "file_name" in source
            assert "page_start" in source
            assert "page_end" in source
            assert "text" in source
        
        # Verify OpenAI APIs were called
        assert mock_openai_embedding.called, "OpenAI embedding API should be called for query"
        assert mock_openai_chat.called, "OpenAI chat API should be called for answer generation"
    
    def test_answer_unknown_question(self, mock_openai_embedding, mock_openai_chat, setup_test_data):
        """Test answering a question about an unknown topic."""
        question = "What is quantum computing?"
        
        # Mock FAISS search to return no relevant results
        with unittest.mock.patch('faiss.IndexFlatIP.search') as mock_search:
            # Return low scores to simulate no relevant matches
            mock_search.return_value = (
                np.array([[0.1, 0.05, 0.02]]),  # low scores
                np.array([[0, 1, 2]])           # indices
            )
            
            # Mock FAISS reconstruct
            with unittest.mock.patch('faiss.IndexFlatIP.reconstruct') as mock_reconstruct:
                mock_reconstruct.return_value = np.random.rand(3072).astype(np.float32)
                
                # Mock chat completion to return appropriate response for unknown topic
                mock_openai_chat.return_value.choices[0].message.content = (
                    "I couldn't find relevant information about quantum computing in the provided book excerpts."
                )
                
                result = answer_question(question, k=3)
        
        # Verify the result indicates no relevant information was found
        answer = result["answer"]
        assert "couldn't find" in answer.lower() or "no information" in answer.lower()
    
    def test_retriever_similar_chunks(self, mock_openai_embedding, setup_test_data):
        """Test the retriever's similar_chunks method."""
        query = "unit testing methodology"
        
        # Mock FAISS search
        with unittest.mock.patch('faiss.IndexFlatIP.search') as mock_search:
            mock_search.return_value = (
                np.array([[0.9, 0.8]]),  # scores
                np.array([[0, 1]])       # indices
            )
            
            # Mock FAISS reconstruct
            with unittest.mock.patch('faiss.IndexFlatIP.reconstruct') as mock_reconstruct:
                mock_reconstruct.return_value = np.random.rand(3072).astype(np.float32)
                
                retriever = BookerRetriever()
                chunks = retriever.similar_chunks(query, k=2)
                retriever.close()
        
        # Verify chunks were returned
        assert len(chunks) > 0
        
        # Verify chunk structure
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "file_name" in chunk
            assert "text" in chunk
            assert "page_start" in chunk
            assert "page_end" in chunk
        
        # Verify embedding API was called
        assert mock_openai_embedding.called
    
    def test_empty_retrieval_results(self, mock_openai_embedding, mock_openai_chat, setup_test_data):
        """Test behavior when retrieval returns no results."""
        question = "What is the meaning of life?"
        
        # Mock retriever to return empty results
        with unittest.mock.patch('booker.qa.BookerRetriever') as mock_retriever_class:
            mock_retriever = unittest.mock.MagicMock()
            mock_retriever.similar_chunks.return_value = []
            mock_retriever_class.return_value = mock_retriever
            
            result = answer_question(question, k=5)
        
        # Verify appropriate response for no results
        assert "couldn't find" in result["answer"].lower()
        assert len(result["sources"]) == 0
    
    def test_qa_with_different_k_values(self, mock_openai_embedding, mock_openai_chat, setup_test_data):
        """Test QA with different numbers of retrieved chunks."""
        question = "What are the different types of testing?"
        
        for k in [1, 3, 5]:
            with unittest.mock.patch('faiss.IndexFlatIP.search') as mock_search:
                # Return k results
                scores = [0.9 - i * 0.1 for i in range(k)]
                indices = list(range(min(k, 3)))  # We only have 3 chunks in test data
                
                mock_search.return_value = (
                    np.array([scores[:len(indices)]]),
                    np.array([indices])
                )
                
                with unittest.mock.patch('faiss.IndexFlatIP.reconstruct') as mock_reconstruct:
                    mock_reconstruct.return_value = np.random.rand(3072).astype(np.float32)
                    
                    result = answer_question(question, k=k)
            
            # Verify we get appropriate number of sources (limited by available data)
            expected_sources = min(k, 3)  # We only have 3 chunks in test data
            assert len(result["sources"]) <= expected_sources


if __name__ == "__main__":
    pytest.main([__file__]) 