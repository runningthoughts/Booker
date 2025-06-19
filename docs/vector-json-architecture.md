# Booker Vector Database & JSON Architecture

This diagram illustrates how the FAISS vector database and multi-level JSON "connective tissue" files work together to create a cohesive chat experience in the Booker system.

## Architecture Overview

```mermaid
graph TB
    subgraph "📚 Source Content"
        A1["📄 PDF/EPUB Files"]
        A2["📁 Hierarchical Directories<br/>0/ Primary<br/>1/ Secondary<br/>2/ Tertiary<br/>3/ Quaternary"]
    end
    
    subgraph "🔄 Ingestion Process"
        B1["HeadingAwareTextSplitter<br/>Creates structured chunks"]
        B2["OpenAI Embeddings<br/>text-embedding-3-large<br/>3072 dimensions"]
        B3["LLM Summarization<br/>Chunk summaries ≤2 sentences<br/>Directory summaries ≤100 tokens"]
        B4["spaCy NLP<br/>Extract entities & keywords"]
    end
    
    subgraph "🗄️ Raw Data Storage"
        C1["🔍 FAISS Vector Index<br/>booker.faiss<br/>Normalized L2 embeddings<br/>Semantic similarity search"]
        C2["🗃️ DuckDB Database<br/>booker.db<br/>chunks table: text, metadata<br/>summaries table: LLM summaries"]
        C3["📋 Pickle Metadata<br/>booker.pkl<br/>Chunk metadata & index mapping"]
    end
    
    subgraph "🎯 JSON Connective Tissue"
        D1["📑 Flat Book JSON<br/>{book_id}_summaries.json<br/>structure: 'flat'<br/>Single-level organization"]
        D2["🏗️ Hierarchical Work JSON<br/>{work_id}_body_of_work.json<br/>structure: 'hierarchical'<br/>Multi-level importance"]
        
        subgraph "Multi-Level Structure"
            D3["📋 Work Summary<br/>Overall work description"]
            D4["📂 Directory Level<br/>directories: {<br/>  '0': {PRIMARY}<br/>  '1': {SECONDARY}<br/>  '2': {TERTIARY}<br/>  '3': {QUATERNARY}<br/>}"]
            D5["🔖 Chunk Cards<br/>Per-chunk metadata:<br/>• summary<br/>• keywords<br/>• entities<br/>• headings<br/>• importance_level"]
        end
    end
    
    subgraph "💬 Chat Query Processing"
        E1["🎤 User Question"]
        E2["🔍 Vector Search<br/>Query embedding → FAISS<br/>MMR reranking for diversity"]
        E3["📖 Metadata Enrichment<br/>Join with JSON sidecars<br/>Add context & summaries"]
        E4["🧠 Intelligent Routing<br/>LOCAL: Use book chunks<br/>KNOWLEDGE: Use LLM + topics<br/>GLOBAL: Web search<br/>REJECT: Out of scope"]
        E5["💡 Enhanced Response<br/>Rich context from JSON<br/>+ Raw text from FAISS<br/>+ Metadata structure"]
    end
    
    subgraph "🎨 UI Presentation"
        F1["💬 Chat Interface<br/>Context-aware responses"]
        F2["📑 Source Citations<br/>Page ranges, summaries"]
        F3["🏷️ Metadata Tags<br/>Keywords, entities, headings"]
        F4["📊 Importance Indicators<br/>Primary/Secondary/etc."]
    end
    
    %% Data Flow Connections
    A1 --> B1
    A2 --> B1
    B1 --> B2
    B1 --> B3
    B1 --> B4
    
    B2 --> C1
    B1 --> C2
    B3 --> C2
    B4 --> C2
    B2 --> C3
    
    C2 --> D1
    C2 --> D2
    D2 --> D3
    D2 --> D4
    D4 --> D5
    
    E1 --> E2
    E2 --> C1
    C1 --> E3
    E3 --> D1
    E3 --> D2
    E3 --> E4
    E4 --> E5
    
    E5 --> F1
    D1 --> F2
    D2 --> F2
    D5 --> F3
    D4 --> F4
    
    %% Key Interaction Arrows
    C1 -.->|"Semantic<br/>Vector Search"| E2
    D1 -.->|"Flat Structure<br/>Context"| E3
    D2 -.->|"Hierarchical<br/>Context"| E3
    D5 -.->|"Rich Metadata<br/>Per Chunk"| E3
    
    %% Styling
    classDef sourceFiles fill:#e3f2fd
    classDef processing fill:#f3e5f5
    classDef rawData fill:#fff3e0
    classDef jsonData fill:#e8f5e8
    classDef chatFlow fill:#fce4ec
    classDef uiFlow fill:#f1f8e9
    
    class A1,A2 sourceFiles
    class B1,B2,B3,B4 processing
    class C1,C2,C3 rawData
    class D1,D2,D3,D4,D5 jsonData
    class E1,E2,E3,E4,E5 chatFlow
    class F1,F2,F3,F4 uiFlow
```

## Key Components

### FAISS Vector Index
- **Purpose**: Semantic similarity search using 3072-dimensional embeddings
- **Technology**: OpenAI's `text-embedding-3-large` model
- **Features**: Normalized L2 embeddings with MMR reranking for result diversity

### Multi-Level JSON Structure
The JSON "connective tissue" provides contextual intelligence in two formats:

#### Flat Structure (Books)
- Simple organization for traditional books
- Files directly in source directory
- Single-level chunk organization

#### Hierarchical Structure (Bodies of Work)
- Complex multi-level organization
- Numbered directories (0/, 1/, 2/, 3/) with importance levels
- Work-level, directory-level, and chunk-level metadata

### Integration Benefits
1. **Separation of Concerns**: FAISS handles semantic search, JSON provides context
2. **Rich Metadata**: Enhanced responses with summaries, keywords, entities
3. **Hierarchical Context**: Multi-level importance and structure
4. **UI-Ready Data**: Pre-formatted for citations and user presentation 