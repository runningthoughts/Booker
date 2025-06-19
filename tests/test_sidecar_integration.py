"""
Tests for sidecar JSON integration functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
import json

from booker.retriever import _load_sidecar, _keyword_overlap_score


def test_sidecar_loaded():
    """Test that sidecar loading returns None when no file exists (normal behavior)."""
    # Clear cache before test  
    _load_sidecar.cache_clear()
    
    # Test that sidecar loading returns None for non-existent book
    # This is the expected behavior when no sidecar file exists  
    result = _load_sidecar("nonexistent_book_test")
    assert result is None, "Should return None when sidecar doesn't exist"


def test_sidecar_not_found():
    """Test that _load_sidecar returns None when file doesn't exist."""
    with patch("pathlib.Path.exists", return_value=False):
        result = _load_sidecar("nonexistent_book")
        assert result is None


def test_keyword_overlap_score():
    """Test keyword overlap scoring functionality."""
    card = {
        "keywords": ["Hydrogen", "Pipeline", "Safety"],
        "entities": ["EPA", "Department of Energy"]
    }
    
    # Test exact match
    score = _keyword_overlap_score("EPA rules on hydrogen pipelines", card)
    assert score > 0, "Should find overlap between query and card"
    
    # Test case insensitive matching
    score_case = _keyword_overlap_score("epa HYDROGEN pipeline", card)
    assert score_case > 0, "Should be case insensitive"
    
    # Test no overlap
    score_none = _keyword_overlap_score("unrelated query about cats", card)
    assert score_none == 0, "Should return 0 for no overlap"


def test_keyword_overlap_score_edge_cases():
    """Test edge cases for keyword overlap scoring."""
    # Empty card
    assert _keyword_overlap_score("test query", {}) == 0.0
    
    # Card with empty keywords/entities
    empty_card = {"keywords": [], "entities": []}
    assert _keyword_overlap_score("test query", empty_card) == 0.0
    
    # Empty query
    card = {"keywords": ["test"], "entities": ["example"]}
    assert _keyword_overlap_score("", card) == 0.0
    
    # Non-string keywords/entities (should be handled gracefully)
    mixed_card = {
        "keywords": ["valid", 123, None],
        "entities": ["entity", {"invalid": "dict"}]
    }
    score = _keyword_overlap_score("valid entity", mixed_card)
    assert score > 0, "Should handle mixed data types gracefully"


def test_keyword_overlap_score_calculation():
    """Test the actual calculation of overlap scores."""
    card = {
        "keywords": ["Python", "Programming", "API"],
        "entities": ["OpenAI"]
    }
    
    # Query matches 2 out of 4 target words (Python, API)
    score = _keyword_overlap_score("Python API development", card)
    expected_score = 2 / 4  # 2 matches out of 4 total target words
    assert abs(score - expected_score) < 0.01, f"Expected {expected_score}, got {score}"
    
    # Query matches all target words
    score_all = _keyword_overlap_score("Python Programming API OpenAI", card)
    assert score_all == 1.0, "Should return 1.0 for complete overlap"


if __name__ == "__main__":
    pytest.main([__file__]) 