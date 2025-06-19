# Booker Ingestion Flow

This diagram shows the complete data flow for processing documents in the Booker system, from source files to searchable databases.

## Flow Diagram

```mermaid
flowchart TD
    A["main()"] --> B["BookerIngestor.__init__()"]
    B --> B1["Setup DuckDB Tables<br/>(chunks, summaries)"]
    B --> B2["Load/Create FAISS Index<br/>(3072 dimensions)"]
    B --> B3["Initialize HeadingAwareTextSplitter<br/>(chunk_size, chunk_overlap)"]
    
    A --> C["ingest_all_content()"]
    C --> D{"Detect Source Type"}
    D -->|"Files in root"| E["process_book()"]
    D -->|"Numbered subdirs (0/, 1/, 2/)"| F["process_body_of_work()"]
    
    E --> G["For each PDF/EPUB file"]
    F --> F1["For each numbered directory<br/>(0=PRIMARY, 1=SECONDARY, etc.)"]
    F1 --> G
    
    G --> H["process_file()"]
    H --> H1["Extract text from PDF<br/>(pypdf.PdfReader)"]
    H1 --> H2["text_splitter.split_text()<br/>(HeadingAwareTextSplitter)"]
    H2 --> H3["Process chunks in batches<br/>(BATCH_SIZE)"]
    
    H3 --> I["For each chunk batch"]
    I --> J["OpenAI API Call #1<br/>_get_embeddings()<br/>Model: text-embedding-3-large<br/>Input: chunk texts<br/>Output: 3072-dim vectors"]
    
    I --> K["For each chunk"]
    K --> K1["Add embedding to FAISS Index<br/>(IndexFlatIP, normalized L2)"]
    K1 --> K2["Insert into DuckDB chunks table<br/>(chunk_id, text, metadata, etc.)"]
    
    K --> L["OpenAI API Call #2<br/>_summarize_chunk()<br/>Model: LLM_MODEL<br/>Input: chunk text + importance context<br/>Output: ≤2 sentence summary"]
    
    K --> M["Extract entities/keywords<br/>(spaCy en_core_web_sm)"]
    
    L --> N["Insert into DuckDB summaries table<br/>(summary, keywords, entities)"]
    M --> N
    
    K --> O["Create chunk card metadata<br/>(for sidecar JSON)"]
    
    H --> P["Collect all chunk recaps"]
    P --> Q["OpenAI API Call #3<br/>_generate_chapter_summary()<br/>Model: LLM_MODEL<br/>Input: combined recaps (≤2000 tokens)<br/>Output: ≤100 token summary"]
    
    E --> R["Generate book sidecar JSON<br/>(flat structure)"]
    F --> S["Generate body of work sidecar JSON<br/>(hierarchical structure)"]
    
    R --> T["Save to sidecars/{book_id}_summaries.json"]
    S --> U["Save to sidecars/{work_id}_body_of_work.json"]
    
    T --> V["Save FAISS index to disk<br/>(booker.faiss + booker.pkl)"]
    U --> V
    V --> W["Commit DuckDB transactions"]
    
    style J fill:#e1f5fe
    style L fill:#e1f5fe
    style Q fill:#e1f5fe
    style K1 fill:#fff3e0
    style K2 fill:#fff3e0
    style N fill:#fff3e0
    style V fill:#fff3e0
```

## Key Components

### OpenAI API Calls (Light Blue)
1. **Embeddings**: `text-embedding-3-large` model converts text chunks into 3072-dimensional vectors
2. **Chunk Summaries**: LLM generates ≤2 sentence summaries for individual chunks with importance context
3. **Chapter/Work Summaries**: LLM creates ≤100 token summaries from combined chunk recaps

### Database Operations (Light Orange)
1. **FAISS Index**: Stores normalized L2 embeddings for semantic search
2. **DuckDB Tables**: 
   - `chunks` table: text, metadata, headings, source info
   - `summaries` table: LLM summaries, keywords, entities
3. **Sidecar JSONs**: Rich metadata files for UI presentation

### Text Processing
- **HeadingAwareTextSplitter**: Intelligently chunks text while preserving document structure
- **spaCy**: Extracts named entities and keywords
- **Importance Levels**: Hierarchical content prioritization (PRIMARY → QUATERNARY)

### Content Types
- **Books**: Files directly in source directory → flat structure
- **Bodies of Work**: Numbered subdirectories (0/, 1/, 2/) → hierarchical importance structure

## Output Artifacts

The ingestion process creates:
1. **FAISS Index** (`booker.faiss` + `booker.pkl`) - for semantic search
2. **DuckDB Database** (`booker.db`) - structured data storage
3. **Sidecar JSON files** - rich metadata for UI presentation

This creates a multi-modal search system combining semantic embeddings, structured metadata, and human-readable summaries for comprehensive document retrieval and understanding. 