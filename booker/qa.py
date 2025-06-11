"""
Question-answering module that combines retrieval with LLM generation.
"""

from typing import Dict, List, Any

import openai

from . import settings
from .retriever import BookerRetriever
from .memory import get_chat_memory

# Initialize OpenAI client
openai.api_key = settings.OPENAI_API_KEY


def answer_question(question: str, retriever: BookerRetriever, k: int = 5, session_id: str = None) -> Dict[str, Any]:
    """
    Answer a question using retrieved book chunks and LLM generation.
    
    Args:
        question: The user's question
        retriever: BookerRetriever instance
        k: Number of chunks to retrieve for context
        session_id: Optional session ID for conversation memory
    
    Returns:
        Dictionary containing the answer and source information
    """
    try:
        # Retrieve relevant chunks
        chunks = retriever.similar_chunks(question, k=k)
        
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
            "sources": sources
        }
    
    except Exception as e:
        raise e


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