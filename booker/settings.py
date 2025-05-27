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
CHUNK_SIZE = 1500  # tokens
CHUNK_OVERLAP = 200  # tokens

# Directory Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = PROJECT_ROOT / "db" / "booker.db"
INDEX_PATH = PROJECT_ROOT / "indexes" / "booker.faiss"
INDEX_META_PATH = PROJECT_ROOT / "indexes" / "booker.pkl"
SIDECAR_DIR = PROJECT_ROOT / "sidecars"

# Ensure directories exist
for path in [DB_PATH.parent, INDEX_PATH.parent, SIDECAR_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Batch processing
BATCH_SIZE = 16  # For OpenAI API calls 