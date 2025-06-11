"""
Conversation memory module for maintaining chat history and summaries.
"""

from collections import deque
from typing import Dict, List, Tuple, Optional

import openai

from . import settings


class ChatMemory:
    """
    Manages conversation history with a rolling window of recent Q-A pairs
    and a summary of older content.
    """
    
    def __init__(self):
        """Initialize the memory with empty deque and summary."""
        self.recent_turns = deque(maxlen=3)  # Last 3 Q-A pairs
        self.running_summary = ""  # Summary of older content
    
    def add_turn(self, user_question: str, assistant_answer: str) -> None:
        """
        Add a new Q-A turn to the memory.
        
        Args:
            user_question: The user's question
            assistant_answer: The assistant's answer
        """
        # Check if we need to summarize before adding new turn
        if len(self.recent_turns) == 3:  # About to overflow
            # Pop the oldest turn and update summary
            old_turn = self.recent_turns[0]
            self._update_summary(old_turn[0], old_turn[1])
        
        # Add the new turn
        self.recent_turns.append((user_question, assistant_answer))
    
    def _update_summary(self, old_question: str, old_answer: str) -> None:
        """
        Update the running summary with the outgoing turn.
        
        Args:
            old_question: The question being removed from recent turns
            old_answer: The answer being removed from recent turns
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "Rewrite the running summary and the outgoing turn so the result "
                              "is ≤250 tokens, preserves factual detail, no invented content."
                },
                {
                    "role": "user",
                    "content": f"Current summary:\n{self.running_summary}\n\n"
                              f"Outgoing turn:\n"
                              f"User: {old_question}\n"
                              f"Assistant: {old_answer}"
                }
            ]
            
            response = openai.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                max_tokens=250,
                temperature=0
            )
            
            self.running_summary = response.choices[0].message.content.strip()
            
        except Exception as e:
            # Fallback: simple truncation if OpenAI call fails
            combined = f"{self.running_summary}\nUser: {old_question}\nAssistant: {old_answer}"
            # Simple truncation to approximately 250 tokens (rough estimate: 1 token ≈ 4 chars)
            if len(combined) > 1000:
                self.running_summary = combined[:1000] + "..."
            else:
                self.running_summary = combined
    
    def format_recent_turns(self) -> str:
        """
        Format the recent turns for inclusion in prompt.
        
        Returns:
            Formatted string of recent Q-A pairs
        """
        if not self.recent_turns:
            return ""
        
        formatted = []
        for user_q, assistant_a in self.recent_turns:
            formatted.append(f"User: {user_q}\nAssistant: {assistant_a}")
        
        return "\n\n".join(formatted)
    
    @property
    def summary(self) -> str:
        """Get the current running summary."""
        return self.running_summary
    
    def clear(self) -> None:
        """Clear all memory."""
        self.recent_turns.clear()
        self.running_summary = ""
    
    def is_empty(self) -> bool:
        """Check if memory is empty."""
        return len(self.recent_turns) == 0 and not self.running_summary


# Global in-memory storage for chat sessions
# In production, this should be replaced with a proper session store
_chat_memories: Dict[str, ChatMemory] = {}


def get_chat_memory(session_id: str) -> ChatMemory:
    """
    Get or create a ChatMemory instance for the given session.
    
    Args:
        session_id: The session identifier
    
    Returns:
        ChatMemory instance for the session
    """
    if session_id not in _chat_memories:
        _chat_memories[session_id] = ChatMemory()
    return _chat_memories[session_id]


def reset_chat_memory(session_id: str) -> None:
    """
    Reset the chat memory for the given session.
    
    Args:
        session_id: The session identifier
    """
    if session_id in _chat_memories:
        del _chat_memories[session_id]


def cleanup_old_sessions(max_sessions: int = 100) -> None:
    """
    Simple cleanup to prevent unbounded memory growth.
    In production, implement proper session expiration.
    
    Args:
        max_sessions: Maximum number of sessions to keep
    """
    if len(_chat_memories) > max_sessions:
        # Remove oldest sessions (simple FIFO)
        oldest_keys = list(_chat_memories.keys())[:-max_sessions]
        for key in oldest_keys:
            del _chat_memories[key] 