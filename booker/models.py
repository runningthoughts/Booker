"""
Pydantic models for the Booker application.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class BookMeta(BaseModel):
    """Metadata model for books to support intelligent routing."""
    
    title: str = Field(description="Book title")
    pub_year: Optional[int] = Field(default=None, description="Publication year")
    abstract: str = Field(description="Book abstract/summary (up to 120 words)")
    topics: List[str] = Field(default_factory=list, description="Key topics extracted from the book")
    not_covered: List[str] = Field(default_factory=list, description="Topics explicitly not covered (for future use)")
    min_year: Optional[int] = Field(default=None, description="Earliest year mentioned in the book content")
    max_year: Optional[int] = Field(default=None, description="Latest year mentioned in the book content")
    pages: Optional[int] = Field(default=None, description="Total number of pages")
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"
        validate_assignment = True 