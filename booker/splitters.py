"""
Text splitting utilities for the Booker application.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import tiktoken


class Document:
    """Simple document class to hold text and metadata."""
    
    def __init__(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        self.text = text
        self.metadata = metadata or {}


class HeadingAwareTextSplitter:
    """
    Text splitter that first attempts to split on headings, then falls back
    to token-based splitting for sections that are too large.
    """
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 80):
        """
        Initialize the splitter.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Heading patterns
        self.markdown_pattern = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
        self.numbered_pattern = re.compile(r'^\d+(\.\d+)*\s+(.+)$', re.MULTILINE)
        self.caps_pattern = re.compile(r'^[A-Z][A-Z\s]{19,}$', re.MULTILINE)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def _detect_heading_type(self, line: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Detect if a line is a heading and return its type and level.
        
        Returns:
            Tuple of (heading_type, level) or (None, None) if not a heading
        """
        line = line.strip()
        
        # Markdown headings
        markdown_match = re.match(r'^(#{1,3})\s+(.+)$', line)
        if markdown_match:
            level = len(markdown_match.group(1))
            return "markdown", level
        
        # Numbered headings (e.g., "3.2 Clinical Outcomes")
        numbered_match = re.match(r'^(\d+(?:\.\d+)*)\s+(.+)$', line)
        if numbered_match:
            # Count the number of dot-separated parts (1 = level 1, 1.2 = level 2, 1.2.3 = level 3)
            number_part = numbered_match.group(1)
            level = len(number_part.split('.'))
            return "numbered", level
        
        # ALL CAPS headings (≥20 chars)
        if re.match(r'^[A-Z][A-Z\s]{19,}$', line):
            return "caps", 1
        
        return None, None
    
    def _split_into_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into sections based on headings.
        
        Returns:
            List of sections with metadata
        """
        lines = text.split('\n')
        sections = []
        current_section = []
        current_heading = None
        current_heading_level = None
        current_heading_type = None
        
        for line in lines:
            heading_type, heading_level = self._detect_heading_type(line)
            
            if heading_type:
                # Save previous section if it exists
                if current_section:
                    section_text = '\n'.join(current_section).strip()
                    if section_text:
                        sections.append({
                            'text': section_text,
                            'heading': current_heading,
                            'heading_level': current_heading_level,
                            'heading_type': current_heading_type
                        })
                
                # Start new section
                current_heading = line.strip()
                current_heading_level = heading_level
                current_heading_type = heading_type
                current_section = [line]
            else:
                current_section.append(line)
        
        # Add final section
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if section_text:
                sections.append({
                    'text': section_text,
                    'heading': current_heading,
                    'heading_level': current_heading_level,
                    'heading_type': current_heading_type
                })
        
        # If no sections were found (no headings), treat entire text as one section
        if not sections:
            sections.append({
                'text': text.strip(),
                'heading': None,
                'heading_level': None,
                'heading_type': None
            })
        
        return sections
    
    def _split_large_section(self, section: Dict[str, Any]) -> List[Document]:
        """
        Split a large section using token-based splitting.
        
        Args:
            section: Section dictionary with text and metadata
            
        Returns:
            List of Document objects
        """
        text = section['text']
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        chunk_num = 0
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Create metadata for this chunk
            metadata = {
                'heading': section['heading'],
                'heading_level': section['heading_level'],
                'heading_type': section['heading_type'],
                'chunk_num': chunk_num,
                'is_continuation': chunk_num > 0
            }
            
            chunks.append(Document(chunk_text, metadata))
            
            if end >= len(tokens):
                break
            
            start = end - self.chunk_overlap
            chunk_num += 1
        
        return chunks
    
    def split_text(self, text: str) -> List[Document]:
        """
        Split text into documents using heading-aware splitting.
        
        Args:
            text: Input text to split
            
        Returns:
            List of Document objects with text and metadata
        """
        # First, split into sections based on headings
        sections = self._split_into_sections(text)
        
        documents = []
        
        for section in sections:
            section_tokens = self._count_tokens(section['text'])
            
            if section_tokens <= self.chunk_size:
                # Section is small enough, use as-is
                metadata = {
                    'heading': section['heading'],
                    'heading_level': section['heading_level'],
                    'heading_type': section['heading_type'],
                    'chunk_num': 0,
                    'is_continuation': False
                }
                documents.append(Document(section['text'], metadata))
            else:
                # Section is too large, split it further
                sub_documents = self._split_large_section(section)
                documents.extend(sub_documents)
        
        return documents 