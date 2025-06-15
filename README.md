# 📚 Booker - RAG-based Book Q&A System

Booker is an intelligent question-answering system that allows you to ask questions about your book collection and get accurate answers with citations. It uses Retrieval-Augmented Generation (RAG) to combine semantic search with large language models.

## 🌟 Features

- **PDF/EPUB Ingestion**: Automatically processes books in your data directory
- **Semantic Search**: Uses FAISS for fast similarity search with embeddings
- **Intelligent Chunking**: Splits books into optimal chunks with overlap
- **Smart Summarization**: Generates summaries for each chunk and chapter
- **Citation Support**: Provides source references with page numbers
- **Modern Web UI**: Beautiful React-based chat interface
- **RESTful API**: FastAPI backend with streaming support
- **Comprehensive Testing**: Full test suite with mocked API calls

## 🏗️ Architecture

```
booker/
├── library/                 # Books root directory (configurable via BOOKS_ROOT)
│   └── <book-id>/          # Individual book directories
│       ├── source/         # Place your PDF/EPUB files here
│       ├── build/          # Generated data
│       │   ├── db/         # DuckDB database
│       │   ├── indexes/    # FAISS indexes
│       │   └── sidecars/   # Auto-generated JSON summaries
│       └── assets/         # Optional metadata and cover images
├── booker/                  # Core Python package
│   ├── __init__.py
│   ├── settings.py          # Configuration and environment variables
│   ├── ingest_book.py       # Book processing and ingestion
│   ├── retriever.py         # Semantic search and retrieval
│   └── qa.py               # Question answering logic
├── api/                     # FastAPI backend
│   └── main.py
├── ui/                      # React frontend
│   ├── src/
│   │   ├── BookerChat.jsx   # Main chat component
│   │   └── main.jsx
│   ├── index.html
│   └── vite.config.js
└── tests/                  # Test suite
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+ (for the frontend)
- OpenAI API key

### New: Intelligent Routing 🧠

Booker now supports intelligent routing between local book content and global web search! See [docs/ROUTING.md](docs/ROUTING.md) for full details.

**Key Features:**
- **Smart metadata generation** with KeyBERT topic extraction
- **LLM-powered routing** (LOCAL/GLOBAL/REJECT decisions)
- **Environment-controlled** global search (disabled by default)
- **Backward compatible** - existing setups continue working

### 1. Environment Setup

```bash
# Set your OpenAI API key
export ragtagKey="your-openai-api-key-here"

# Optional: Set custom books root directory (defaults to ./library)
export BOOKS_ROOT="/path/to/your/books/collection"
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt

# Download spaCy model for entity extraction
python -m spacy download en_core_web_sm
```

### 3. Install Frontend Dependencies

```bash
cd ui
npm install react react-dom
npm install -D vite @vitejs/plugin-react
cd ..
```

### 4. Add Your Books

Create a book directory and place your PDF or EPUB files in the source folder:

```bash
# Create directory structure for a book
mkdir -p library/The_Gilded_Cage/source
cp /path/to/your/book.pdf library/The_Gilded_Cage/source/
```

### 5. Ingest Your Books

```bash
# Basic ingestion
python -m booker.ingest_book --book-id The_Gilded_Cage

# With intelligent routing metadata (NEW!)
python -m booker.ingest_book --book-id The_Gilded_Cage --profile
```

This will:
- Extract text from your books
- Create semantic embeddings
- Generate summaries
- Build searchable indexes
- Create citation metadata
- **NEW**: Generate book metadata for intelligent routing (with `--profile` flag)

### 6. Start the System

**Option A: Use the startup script (recommended)**
```bash
./start_booker.sh
```

**Option B: Start manually**

Backend:
```bash
PYTHONPATH=. python api/main.py
```

Frontend (in another terminal):
```bash
cd ui && npm run dev
```

The API will be available at `http://localhost:8000`  
The web interface will be available at `http://localhost:3000`

## 📖 Usage

### Web Interface

1. Open your browser to `http://localhost:3000?bookId=The_Gilded_Cage`
2. Type your question in the chat box
3. Get intelligent answers with source citations
4. Click on citations to see the source material

Note: You must specify the `bookId` parameter in the URL to select which book to query.

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Ask a Question
```bash
curl -X POST http://localhost:8000/ask/The_Gilded_Cage \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "k": 5}'
```

#### Streaming Response
```bash
curl -X POST http://localhost:8000/ask/The_Gilded_Cage/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain neural networks", "k": 3}'
```

### Python API

```python
from pathlib import Path
from booker.qa import answer_question
from booker.retriever import BookerRetriever
from booker.settings import BOOKS_ROOT

# Set up paths for a specific book
book_id = "The_Gilded_Cage"
base_dir = BOOKS_ROOT / book_id / "build"
db_path = base_dir / "db" / "booker.db"
index_path = base_dir / "indexes" / "booker.faiss"

# Create retriever and ask question
retriever = BookerRetriever(db_path, index_path)
try:
    result = answer_question("What is deep learning?", retriever)
    print(result["answer"])
    for source in result["sources"]:
        print(f"Source: {source['file_name']}, Pages: {source['page_start']}-{source['page_end']}")
finally:
    retriever.close()
```

## 📊 Local Visualization

Booker includes a powerful Spotlight-based visualization tool with semantic clustering for exploring book embeddings and metadata! This is an optional dev-only feature that requires additional dependencies.

### Installation

```bash
# 1. install the dev extras
pip install -r requirements-viz.txt
```

### Usage

```bash
# 2. run the explorer
python -m booker.viz <publication_id>       # auto clusters  
python -m booker.viz <publication_id> --k 20 # manual cluster count

# Additional options:
python -m booker.viz <publication_id> --no-cluster          # skip clustering
python -m booker.viz <publication_id> --no-extra-metadata   # skip DuckDB merge
```

### Features

The enhanced visualization now includes:

- **Semantic Clustering**: Automatic k-means clustering (~25 points per cluster)
- **DuckDB Integration**: Merges heading levels and importance scores if available  
- **Smart Layouts**: Adaptive UMAP parameters based on dataset size
- **Intelligent Coloring**: Auto-selects color scheme (clusters → headings → levels)
- **Production-Safe**: Heavy dependencies are dev-only, won't impact Render deployments

### What You'll See

- Interactive Spotlight interface with your book's vector embeddings
- Semantic clusters colored by k-means labels
- Rich metadata filtering and analysis
- Optimized UMAP dimensionality reduction
- Source text and summaries for each chunk

## 🔧 Configuration

### Environment Variables

- `ragtagKey`: Your OpenAI API key (required)
- `BOOKS_ROOT`: Parent directory for all books (optional, defaults to `./library`)
The intelligent routing system automatically enables when book metadata is present, no additional configuration needed.

### Settings (booker/settings.py)

- `EMBED_MODEL`: OpenAI embedding model (default: "text-embedding-3-large")
- `LLM_MODEL`: OpenAI chat model (default: "gpt-4o-mini")
- `CHUNK_SIZE`: Token size for text chunks (default: 1500)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `BATCH_SIZE`: Batch size for API calls (default: 16)

## 🗄️ Database Schema

### Chunks Table
```sql
CREATE TABLE chunks (
    chunk_id      INTEGER PRIMARY KEY,
    book_id       TEXT,
    file_name     TEXT,
    chapter_no    INTEGER,
    chapter_title TEXT,
    page_start    INTEGER,
    page_end      INTEGER,
    text          TEXT
);
```

### Summaries Table
```sql
CREATE TABLE summaries (
    chunk_id   INTEGER REFERENCES chunks(chunk_id),
    summary    TEXT,
    keywords   TEXT,
    entities   JSON
);
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_ingest.py
pytest tests/test_qa_loop.py

# Run new routing tests
pytest tests/test_routing.py
pytest tests/test_profile_book.py

# Run with verbose output
pytest -v
```

The tests use mocked OpenAI API calls to avoid costs during development.

## 📁 File Formats

### Supported Input Formats
- PDF files (`.pdf`)
- EPUB files (`.epub`) - coming soon

### Generated Files

#### Sidecar JSON (`sidecars/<filename>_summaries.json`)
```json
{
  "file": "book.pdf",
  "chapter_no": 1,
  "chapter_title": "Introduction",
  "chapter_summary": "This chapter introduces...",
  "chunks": [
    {
      "chunk_id": 1,
      "embedding_index": 0,
      "page_range": "1-10",
      "summary": "Brief summary...",
      "keywords": ["keyword1", "keyword2"],
      "entities": [{"text": "Entity", "label": "PERSON"}]
    }
  ]
}
```

## 🔍 How It Works

### Core System
1. **Ingestion**: Books are processed into overlapping text chunks
2. **Embedding**: Each chunk is converted to a vector using OpenAI's embedding model
3. **Indexing**: Vectors are stored in a FAISS index for fast similarity search
4. **Summarization**: Each chunk gets a summary using GPT-4o-mini
5. **Storage**: Text, metadata, and summaries are stored in DuckDB

### NEW: Intelligent Routing
6. **Profiling**: Book metadata is generated with topics, year ranges, and abstracts
7. **Retrieval**: Questions are embedded and matched against the index
8. **Routing**: LLM decides LOCAL (use book), GLOBAL (web search), or REJECT (off-topic)
9. **Generation**: Answer using appropriate source with proper context
10. **Citation**: Sources are tracked and provided with answers

## 🛠️ Development

### Project Structure

- `booker/`: Core library code
  - `models.py`: Pydantic models (NEW)
  - `web_search.py`: Web search functionality (NEW)
- `api/`: FastAPI web service
- `scripts/`: Standalone scripts
  - `profile_book.py`: Book metadata generation (NEW)
- `docs/`: Documentation
  - `ROUTING.md`: Intelligent routing guide (NEW)
- `ui/`: React frontend application
- `tests/`: Comprehensive test suite
- `library/`: Your book collections organized by book-id (not in git)
- `library/<book-id>/build/`: Generated data per book (not in git)

### Code Standards

- Type hints for all functions
- Comprehensive docstrings
- PEP 8 compliance
- No global variables (except in settings.py)
- Proper error handling and logging

### Adding New Features

1. Add core logic to the `booker/` package
2. Update the API endpoints in `api/main.py`
3. Enhance the UI in `ui/src/BookerChat.jsx`
4. Add tests in the `tests/` directory

## 🚨 Troubleshooting

### Common Issues

1. **Missing API Key**: Ensure `ragtagKey` environment variable is set
2. **spaCy Model**: Run `python -m spacy download en_core_web_sm`
3. **Empty Results**: Check that books are in the correct `library/<book-id>/source/` directory and ingestion completed
4. **Port Conflicts**: Change ports in `api/main.py` and `ui/vite.config.js`
5. **Book Not Found**: Ensure the book-id matches the directory name and ingestion was successful

### Logs

Check the console output for detailed logging during ingestion and querying.

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support

For issues and questions, please check the troubleshooting section or create an issue in the repository.

---

**Happy reading and questioning! 📚✨** 