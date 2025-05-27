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
├── data/                    # Place your PDF/EPUB files here
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
├── sidecars/               # Auto-generated JSON summaries
├── db/                     # DuckDB database (created at runtime)
├── indexes/                # FAISS indexes (created at runtime)
└── tests/                  # Test suite
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+ (for the frontend)
- OpenAI API key

### 1. Environment Setup

```bash
# Set your OpenAI API key
export ragtagKey="your-openai-api-key-here"
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

Place your PDF or EPUB files in the `data/` directory:

```bash
cp /path/to/your/books/*.pdf data/
```

### 5. Ingest Your Books

```bash
python -m booker.ingest_book
```

This will:
- Extract text from your books
- Create semantic embeddings
- Generate summaries
- Build searchable indexes
- Create citation metadata

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

1. Open your browser to `http://localhost:3000`
2. Type your question in the chat box
3. Get intelligent answers with source citations
4. Click on citations to see the source material

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Ask a Question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "k": 5}'
```

#### Streaming Response
```bash
curl -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain neural networks", "k": 3}'
```

### Python API

```python
from booker.qa import answer_question

result = answer_question("What is deep learning?")
print(result["answer"])
for source in result["sources"]:
    print(f"Source: {source['file_name']}, Pages: {source['page_start']}-{source['page_end']}")
```

## 🔧 Configuration

### Environment Variables

- `ragtagKey`: Your OpenAI API key (required)

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

1. **Ingestion**: Books are processed into overlapping text chunks
2. **Embedding**: Each chunk is converted to a vector using OpenAI's embedding model
3. **Indexing**: Vectors are stored in a FAISS index for fast similarity search
4. **Summarization**: Each chunk gets a summary using GPT-4o-mini
5. **Storage**: Text, metadata, and summaries are stored in DuckDB
6. **Retrieval**: Questions are embedded and matched against the index
7. **Generation**: Retrieved chunks provide context for answer generation
8. **Citation**: Sources are tracked and provided with answers

## 🛠️ Development

### Project Structure

- `booker/`: Core library code
- `api/`: FastAPI web service
- `ui/`: React frontend application
- `tests/`: Comprehensive test suite
- `data/`: Your book files (not in git)
- `db/`, `indexes/`, `sidecars/`: Generated data (not in git)

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
3. **Empty Results**: Check that books are in the `data/` directory and ingestion completed
4. **Port Conflicts**: Change ports in `api/main.py` and `ui/vite.config.js`

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