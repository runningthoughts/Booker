"""
Global settings and configuration for the Booker application.
"""

import os
from pathlib import Path

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("ragtagKey")
if not OPENAI_API_KEY:
    raise RuntimeError("Environment variable 'ragtagKey' is required but not set")

EMBED_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"

# Chunking Configuration
CHUNK_SIZE = 800  # tokens
CHUNK_OVERLAP = 80  # tokens

# Directory Paths
PROJECT_ROOT = Path(__file__).parent.parent
BOOKS_ROOT = Path(os.getenv("BOOKS_ROOT", PROJECT_ROOT / "library"))

# Batch processing
BATCH_SIZE = 16  # For OpenAI API calls