"""
Question-answering module that combines retrieval with LLM generation.
Supports intelligent routing between local and global search based on metadata.
"""

import json
import logging
import os
from pathlib import Path, PurePath
from typing import Dict, List, Any, Optional

import openai

from . import settings
from .retriever import BookerRetriever
from .memory import get_chat_memory
from .web_search import answer_from_web

logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai.api_key = settings.OPENAI_API_KEY

# Out of scope response for rejected questions
OUT_OF_SCOPE_RESPONSE = {
    "answer": "I'm sorry, but your question appears to be outside the scope of the available book content. Please ask questions related to the topics covered in the book.",
    "sources": []
}


def load_book_meta(book_id: str, library_root: Path = Path("library")) -> Optional[Dict[str, Any]]:
    """
    Load book metadata from book_meta.json if it exists.
    
    Args:
        book_id: Book identifier
        library_root: Root directory for books
        
    Returns:
        Book metadata dictionary or None if not found
    """
    # Handle both relative paths from root and from api directory
    possible_paths = [
        library_root / book_id / "book_meta.json",  # From root directory
        Path("..") / library_root / book_id / "book_meta.json"  # From api directory
    ]
    
    for meta_path in possible_paths:
        if meta_path.exists():
            try:
                logger.info(f"Loading book metadata from: {meta_path.absolute()}")
                with open(meta_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load book metadata from {meta_path}: {e}")
    
    logger.warning(f"Book metadata not found for {book_id}. Tried paths: {[str(p) for p in possible_paths]}")
    return None


def call_llm_router(question: str, top_chunks: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    """
    Use LLM to determine routing decision: LOCAL, KNOWLEDGE, GLOBAL, or REJECT.
    
    Args:
        question: User's question
        top_chunks: Top retrieved chunks from the book
        meta: Book metadata
        
    Returns:
        Routing decision: "LOCAL", "KNOWLEDGE", "GLOBAL", or "REJECT"
    """
    try:
        # Build context from top chunks
        chunk_summaries = []
        for i, chunk in enumerate(top_chunks[:3], 1):
            summary = chunk.get("summary", chunk.get("text", "")[:200])
            chunk_summaries.append(f"Chunk {i}: {summary}")
        
        chunks_context = "\n".join(chunk_summaries)
        
        # Build metadata context
        topics = ", ".join(meta.get("topics", [])[:10])  # Limit topics
        year_range = f"{meta.get('min_year', 'Unknown')}-{meta.get('max_year', 'Unknown')}"
        
        system_prompt = """You are a routing assistant for a book Q&A system. Given a user question, book metadata, and the most relevant chunks from the book, decide:

LOCAL: Answer can be found directly in the book content (chunks are highly relevant)
KNOWLEDGE: Question relates to book topics but needs broader knowledge (organizations, techniques, equipment mentioned in book but not fully explained)
GLOBAL: Question needs current/live information (recent events, current prices, latest versions)
REJECT: Question is off-topic, nonsensical, or inappropriate

Consider:
- If chunks directly answer the question → LOCAL
- If question asks about entities/concepts mentioned in book topics but not fully covered → KNOWLEDGE
- If question asks about current events, live data, or recent developments → GLOBAL
- If question is completely unrelated to book topics → REJECT

Examples:
- "What's in Chapter 3?" → LOCAL
- "Tell me about the National Stereoscopic Association" (if mentioned in topics) → KNOWLEDGE  
- "What's the current price of a Canon camera?" → GLOBAL
- "What's the weather today?" → REJECT

Respond with exactly one word: LOCAL, KNOWLEDGE, GLOBAL, or REJECT"""

        user_content = f"""Question: {question}

Book Metadata:
- Title: {meta.get('title', 'Unknown')}
- Topics: {topics}
- Year range covered: {year_range}
- Abstract: {meta.get('abstract', 'No abstract available')[:300]}

Most Relevant Chunks:
{chunks_context}

Routing Decision:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = openai.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=messages,
            max_tokens=10,
            temperature=0.1
        )

        decision = response.choices[0].message.content.strip().upper()
        
        # Validate decision
        if decision in ["LOCAL", "KNOWLEDGE", "GLOBAL", "REJECT"]:
            logger.info(f"Router decision for '{question[:50]}...': {decision}")
            return decision
        else:
            logger.warning(f"Invalid router decision '{decision}', defaulting to LOCAL")
            return "LOCAL"
            
    except Exception as e:
        logger.error(f"Router failed: {e}, defaulting to LOCAL")
        return "LOCAL"


def answer_from_chunks(question: str, chunks: List[Dict[str, Any]], session_id: str = None) -> Dict[str, Any]:
    """
    Answer a question using only the provided book chunks.
    
    Args:
        question: User's question
        chunks: Retrieved book chunks
        session_id: Optional session ID for memory
        
    Returns:
        Answer dictionary with local sources
    """
    if not chunks:
        return {
            "answer": "I couldn't find any relevant information in the books to answer your question.",
            "sources": []
        }
    
    # Build context from chunks
    context_parts = []
    sources = []
    
    for i, chunk in enumerate(chunks, 1):
        # Add chunk text to context
        context_parts.append(f"[Source {i}] {chunk['text']}")
        
        # Add to sources list
        sources.append({
            "source_id": i,
            "file_name": chunk["file_name"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
            "summary": chunk.get("summary", "")
        })
    
    context = "\n\n".join(context_parts)
    
    # Get conversation memory if session_id provided
    memory_context = ""
    if session_id:
        memory = get_chat_memory(session_id)
        memory_parts = []
        
        if memory.summary:
            memory_parts.append(f"Conversation summary:\n{memory.summary}")
        
        recent_turns = memory.format_recent_turns()
        if recent_turns:
            memory_parts.append(f"Recent turns:\n{recent_turns}")
        
        if memory_parts:
            memory_context = "\n\n".join(memory_parts) + "\n\n"
    
    # Build prompt for LLM
    system_prompt = ("You are an AI assistant who strictly answers from the provided book excerpts. "
                    "Base your answer only on the information given in the context. "
                    "If the context doesn't contain enough information to answer the question, "
                    "say so clearly. When referencing information, mention the source number in brackets.")
    
    if memory_context:
        system_prompt += (" Use the conversation summary and recent turns to maintain context, "
                        "but always prioritize information from the book excerpts.")
    
    user_content = f"{memory_context}Relevant book excerpts:\n{context}\n\nUser: {question}"
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user", 
            "content": user_content
        }
    ]
    
    # Generate answer using LLM
    response = openai.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        max_tokens=500,
        temperature=0.3
    )
    
    answer = response.choices[0].message.content.strip()
    
    # Add this turn to memory if session_id provided
    if session_id:
        memory = get_chat_memory(session_id)
        memory.add_turn(question, answer)
    
    return {
        "answer": answer,
        "sources": sources,
        "search_type": "local"
    }


def answer_from_knowledge(question: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Answer a question using LLM's training knowledge, guided by book context.
    
    Args:
        question: User's question
        meta: Book metadata for context
        
    Returns:
        Answer dictionary with knowledge-based response
    """
    try:
        # Build context from book metadata
        topics = ", ".join(meta.get("topics", [])[:15])
        year_range = f"{meta.get('min_year', 'Unknown')}-{meta.get('max_year', 'Unknown')}"
        
        system_prompt = f"""You are an expert assistant answering questions about topics covered in a book on {meta.get('title', 'the subject')}. 

The book covers these topics: {topics}
Time period: {year_range}
Book summary: {meta.get('abstract', 'No summary available')[:400]}

Use your knowledge to provide comprehensive, accurate information about the user's question as it relates to these topics. Since this relates to the book's subject matter, provide detailed explanations that would help someone learning about these topics.

Be informative and educational, drawing on your training knowledge. If you mention specific organizations, techniques, or equipment, explain what they are and why they're relevant to the field."""

        user_content = f"Question about {meta.get('title', 'the book topics')}: {question}"
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        response = openai.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=messages,
            max_tokens=600,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content.strip()
        
        return {
            "answer": answer,
            "sources": [],  # No specific sources for knowledge-based answers
            "search_type": "knowledge"
        }
        
    except Exception as e:
        logger.error(f"Knowledge-based answer failed: {e}")
        return {
            "answer": "I encountered an error while accessing my knowledge about this topic. Please try again.",
            "sources": [],
            "search_type": "knowledge"
        }


def answer_question(question: str, retriever: BookerRetriever, k: int = 5, session_id: str = None, book_id: str = None) -> Dict[str, Any]:
    """
    Answer a question using intelligent routing between local and global search.
    
    Args:
        question: The user's question
        retriever: BookerRetriever instance
        k: Number of chunks to retrieve for context
        session_id: Optional session ID for conversation memory
        book_id: Book identifier for metadata loading
    
    Returns:
        Dictionary containing the answer and source information
    """
    try:
        # Retrieve relevant chunks
        chunks = retriever.similar_chunks(question, k=k)
        
        # Extract book_id from the first chunk if not provided
        if not book_id and chunks:
            book_id = chunks[0].get("book_id")
        
        # Load metadata if available
        meta = None
        if book_id:
            try:
                meta = load_book_meta(book_id)
                logger.info(f"Loaded metadata for book {book_id}: {meta is not None}")
            except Exception as e:
                logger.warning(f"Failed to load metadata for book {book_id}: {e}")
        
        # Routing logic: automatically enable smart routing when metadata is present
        if meta:
            try:
                # Smart routing with metadata - metadata presence enables intelligent routing
                route = call_llm_router(question, chunks[:3], meta)
                logger.info(f"Router decision: {route}")
                
                if route == "LOCAL":
                    return answer_from_chunks(question, chunks, session_id)
                elif route == "KNOWLEDGE":
                    return answer_from_knowledge(question, meta)
                elif route == "GLOBAL":
                    return answer_from_web(question)
                else:  # REJECT
                    return OUT_OF_SCOPE_RESPONSE
            except Exception as e:
                logger.error(f"Error in routing logic: {e}, falling back to local search")
                return answer_from_chunks(question, chunks, session_id)
        else:
            # No metadata available - fall back to local-only mode (current behavior)
            logger.info("No metadata available, using local-only mode")
            if not chunks:
                return OUT_OF_SCOPE_RESPONSE
            
            return answer_from_chunks(question, chunks, session_id)
    
    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        # Return a safe fallback response instead of raising
        return {
            "answer": f"I encountered an error while processing your question: {str(e)}. Please try again.",
            "sources": []
        }


def answer_question_stream(question: str, retriever: BookerRetriever, k: int = 5):
    """
    Answer a question with streaming response.
    
    Args:
        question: The user's question
        retriever: BookerRetriever instance
        k: Number of chunks to retrieve for context
    
    Yields:
        Streaming response chunks
    """
    try:
        # Retrieve relevant chunks
        chunks = retriever.similar_chunks(question, k=k)
        
        if not chunks:
            yield {
                "type": "answer",
                "content": "I couldn't find any relevant information in the books to answer your question."
            }
            yield {
                "type": "sources",
                "content": []
            }
            return
        
        # Build context from chunks
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(chunks, 1):
            # Add chunk text to context
            context_parts.append(f"[Source {i}] {chunk['text']}")
            
            # Add to sources list
            sources.append({
                "source_id": i,
                "file_name": chunk["file_name"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "summary": chunk.get("summary", "")
            })
        
        context = "\n\n".join(context_parts)
        
        # Build prompt for LLM
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant who strictly answers from the provided book excerpts. "
                          "Base your answer only on the information given in the context. "
                          "If the context doesn't contain enough information to answer the question, "
                          "say so clearly. When referencing information, mention the source number in brackets."
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext:\n{context}"
            }
        ]
        
        # Generate streaming answer
        stream = openai.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.3,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield {
                    "type": "answer_chunk",
                    "content": chunk.choices[0].delta.content
                }
        
        # Send sources at the end
        yield {
            "type": "sources",
            "content": sources
        }
    
    except Exception as e:
        raise e 