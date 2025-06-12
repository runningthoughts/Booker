#!/usr/bin/env python3
"""
Standalone script to generate book_meta.json from existing ingested data.
This script runs after ingestion to create metadata for intelligent routing.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import duckdb
import openai
from keybert import KeyBERT

# Add parent directory to path to import booker modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from booker import settings
from booker.models import BookMeta

# Initialize OpenAI client
openai.api_key = settings.OPENAI_API_KEY

# Initialize KeyBERT
kw_model = KeyBERT()


def extract_years_from_text(text: str) -> List[int]:
    """
    Extract 4-digit years from text using regex.
    
    Args:
        text: Text to search for years
        
    Returns:
        List of years found in the text
    """
    # Pattern for 4-digit years (1000-2999 range)
    year_pattern = r'\b(1[0-9]{3}|2[0-9]{3})\b'
    matches = re.findall(year_pattern, text)
    return [int(year) for year in matches]


def extract_technical_terms(texts: List[str]) -> List[str]:
    """
    Extract technical terms using regex patterns for photography/3D content.
    """
    combined_text = " ".join(texts)
    technical_terms = []
    
    # Camera brands and specific models
    camera_brands = re.findall(r'\b(Canon|Nikon|Sony|Fuji|FUJI|Panasonic|Olympus|Pentax|Leica|Sigma|Tamron|Zeiss)\b', combined_text, re.IGNORECASE)
    camera_models = re.findall(r'\b(Canon\s+EOS\s*\w*|Nikon\s+D\d+|Sony\s+\w+|Fuji\s*W\d+|Stereo\s*Realist|Nimslo|View[- ]?Master)\b', combined_text, re.IGNORECASE)
    
    # Software and applications (more specific)
    software = re.findall(r'\b(Adobe\s*Photoshop|Lightroom|GIMP|StereoPhoto\s*Maker|SPM|3D\s*Vision|Anaglyph\s*Maker|StereoData\s*Maker|Owl\s*Viewer|Pop[- ]?up\s*3D|iPad\s*app)\b', combined_text, re.IGNORECASE)
    
    # Equipment and technical terms
    equipment = re.findall(r'\b(lens|lenses|tripod|flash|shutter|aperture|focal|f/\d+|mm|ISO|megapixel|MP|pixels|LCD|viewfinder|exposure|depth of field|DOF)\b', combined_text, re.IGNORECASE)
    
    # 3D specific terms
    threeD_terms = re.findall(r'\b(stereoscopic|parallax|convergence|interocular|baseline|anaglyph|polarized|cross[- ]?eyed?|parallel|MPO|JPS|red[- ]?cyan|amber[- ]?blue)\b', combined_text, re.IGNORECASE)
    
    # File formats and technical specs
    formats = re.findall(r'\b(JPEG|JPG|PNG|TIFF|RAW|MPO|JPS|GIF|BMP)\b', combined_text, re.IGNORECASE)
    
    # Organizations and publications  
    organizations = re.findall(r'\b(National\s*Stereoscopic\s*Association|NSA|International\s*Stereoscopic\s*Union|ISU|StereoWorld|London\s*Stereoscopic\s*Company)\b', combined_text, re.IGNORECASE)
    
    # Combine all and deduplicate, cleaning up whitespace
    all_terms = camera_brands + camera_models + software + equipment + threeD_terms + formats + organizations
    cleaned_terms = []
    for term in all_terms:
        # Clean up whitespace, newlines, and normalize
        cleaned = ' '.join(term.strip().split()).lower()
        if len(cleaned) > 2:
            cleaned_terms.append(cleaned)
    
    return list(set(cleaned_terms))


def extract_topics_with_keybert(texts: List[str], top_k: int = 20) -> List[str]:
    """
    Extract key topics from text using KeyBERT, focusing on technical terms and proper nouns.
    
    Args:
        texts: List of text chunks to analyze
        top_k: Number of top topics to extract
        
    Returns:
        List of key topics (technical terms, brands, methodologies, etc.)
    """
    try:
        # Combine all texts for topic extraction
        combined_text = " ".join(texts)
        
        # Extract both single words and short phrases
        keywords_1 = kw_model.extract_keywords(
            combined_text, 
            keyphrase_ngram_range=(1, 1),  # Single words for technical terms
            stop_words='english',
            use_mmr=True,
            diversity=0.3
        )
        
        keywords_2 = kw_model.extract_keywords(
            combined_text,
            keyphrase_ngram_range=(2, 2),  # Two-word technical terms
            stop_words='english', 
            use_mmr=True,
            diversity=0.3
        )
        
        # Combine and deduplicate
        all_keywords = keywords_1 + keywords_2
        
        # Filter for meaningful technical terms
        filtered_topics = []
        seen_topics = set()
        
        for keyword, score in all_keywords:
            keyword_clean = keyword.lower().strip()
            
            # Skip duplicates
            if keyword_clean in seen_topics:
                continue
                
            # Skip vague/meaningless phrases
            skip_patterns = [
                'people', 'time', 'way', 'thing', 'things', 'much', 'many', 'very',
                'really', 'quite', 'rather', 'somewhat', 'usually', 'normally',
                'currently', 'generally', 'typically', 'basically', 'simply',
                'actually', 'probably', 'likely', 'possible', 'different',
                'various', 'several', 'certain', 'particular', 'specific',
                'important', 'interesting', 'useful', 'good', 'bad', 'better',
                'best', 'great', 'small', 'large', 'big', 'little', 'new', 'old',
                'make', 'take', 'get', 'using', 'used', 'use', 'work', 'works',
                'see', 'look', 'show', 'shown', 'need', 'want', 'like', 'first',
                'second', 'third', 'last', 'next', 'page', 'chapter', 'book',
                'example', 'figure', 'image', 'picture', 'photo'
            ]
            
            # Skip if it's just a vague word
            if any(pattern in keyword_clean for pattern in skip_patterns):
                continue
            
            # Look for technical indicators
            is_technical = (
                keyword[0].isupper() or  # Proper noun (brands, software)
                any(char.isdigit() for char in keyword) or  # Model numbers
                keyword_clean.endswith(('mm', 'px', 'mp', 'iso', 'fps')) or  # Technical units
                keyword_clean in ['canon', 'nikon', 'sony', 'fuji', 'panasonic', 'olympus', 'pentax'] or  # Camera brands
                keyword_clean in ['photoshop', 'lightroom', 'gimp', 'stereo', 'photo', 'maker'] or  # Software
                keyword_clean in ['lens', 'camera', 'tripod', 'flash', 'shutter', 'aperture', 'focal'] or  # Equipment
                keyword_clean in ['jpeg', 'raw', 'tiff', 'png', 'mpo', 'jps'] or  # File formats
                keyword_clean in ['parallel', 'crosseyed', 'anaglyph', 'polarized'] or  # 3D techniques
                len(keyword) > 6 and keyword.isalpha()  # Longer technical terms
            )
            
            # Add if it seems technical and meaningful
            if is_technical and len(keyword) > 2:
                filtered_topics.append(keyword)
                seen_topics.add(keyword_clean)
                
            if len(filtered_topics) >= top_k:
                break
        
        return filtered_topics[:top_k] if filtered_topics else []
        
    except Exception as e:
        print(f"Warning: KeyBERT extraction failed: {e}")
        return []


def extract_topics_with_llm(texts: List[str], top_k: int = 20) -> List[str]:
    """
    Extract technical topics using GPT-4o-mini for better context understanding.
    
    Args:
        texts: List of text chunks to analyze
        top_k: Number of top topics to extract
        
    Returns:
        List of technical topics extracted by LLM
    """
    try:
        # Combine representative chunks
        combined_text = " ".join(texts[:30])  # Use more chunks but limit total length
        sample_text = combined_text[:6000]  # Limit to avoid token limits
        
        response = openai.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """Extract technical topics from this text. Focus ONLY on:
- Specific camera brands and models (e.g., "Canon EOS", "Nikon D750")
- Equipment and accessories (e.g., "tripod", "polarizing filter") 
- Software names (e.g., "Photoshop", "StereoPhoto Maker")
- Technical photography terms (e.g., "aperture", "depth of field", "ISO")
- File formats (e.g., "JPEG", "MPO", "RAW")
- 3D/stereoscopic techniques (e.g., "anaglyph", "parallax", "convergence")
- Specific methodologies or workflows

Return ONLY a JSON list of strings. No explanations. Avoid vague terms like "photography", "images", "camera" alone. Be specific."""
                },
                {
                    "role": "user",
                    "content": f"Extract technical topics from this content:\n\n{sample_text}"
                }
            ],
            max_tokens=300,
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response (handle code blocks)
        import json
        try:
            # Remove code block markers if present
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.startswith("```"):
                content = content[3:]   # Remove ```
            if content.endswith("```"):
                content = content[:-3]  # Remove ending ```
            
            content = content.strip()
            topics = json.loads(content)
            if isinstance(topics, list):
                return topics[:top_k]
        except json.JSONDecodeError:
            # Fallback: try to extract from text if JSON parsing fails
            print(f"JSON parsing failed, raw response: {content}")
            
        return []
        
    except Exception as e:
        print(f"Warning: LLM topic extraction failed: {e}")
        return []


def extract_combined_topics(texts: List[str], top_k: int = 20) -> List[str]:
    """
    Combine regex, KeyBERT, and LLM-based topic extraction for comprehensive coverage.
    
    Args:
        texts: List of text chunks to analyze
        top_k: Number of top topics to extract
        
    Returns:
        List of combined technical topics
    """
    # Get technical terms using different approaches
    regex_terms = extract_technical_terms(texts)
    llm_topics = extract_topics_with_llm(texts, top_k=12)
    keybert_topics = extract_topics_with_keybert(texts, top_k=8)
    
    # Prioritize specific products/models from regex (these are most precise)
    high_priority_regex = []
    normal_regex = []
    
    for term in regex_terms:
        # High priority: specific models, software names, organizations
        if any(pattern in term.lower() for pattern in ['w1', 'w3', 'eos', 'd7', 'maker', 'app', 'nsa', 'isu']):
            high_priority_regex.append(term)
        else:
            normal_regex.append(term)
    
    # Combine with prioritization: High-priority regex first, then LLM, then normal regex, then KeyBERT
    all_topics = high_priority_regex + llm_topics + normal_regex + keybert_topics
    
    # Deduplicate while preserving order and cleaning whitespace
    seen = set()
    final_topics = []
    for topic in all_topics:
        # Clean up whitespace, newlines, and normalize
        topic_clean = ' '.join(topic.strip().split()).lower()
        if topic_clean not in seen and len(topic_clean) > 2:
            # Use the cleaned version for output
            final_topics.append(' '.join(topic.strip().split()))
            seen.add(topic_clean)
            
        if len(final_topics) >= top_k:
            break
    
    return final_topics


def generate_not_covered_topics(texts: List[str], covered_topics: List[str]) -> List[str]:
    """
    Use LLM to identify related topics that readers might expect but aren't covered in the book.
    
    Args:
        texts: Sample text chunks from the book
        covered_topics: Topics that are covered in the book
        
    Returns:
        List of related but not-covered topics
    """
    try:
        # Use a sample of text to understand book scope
        sample_text = " ".join(texts[:20])[:4000]  # Limit to avoid token limits
        covered_list = ", ".join(covered_topics[:15])
        
        response = openai.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """Analyze this book content and identify related topics that readers might reasonably expect to find but are NOT actually covered. 

Focus on:
- Adjacent technologies or techniques in the same field
- Common equipment or software not mentioned
- Related workflows or methodologies missing
- Complementary topics a complete guide might include

Return ONLY a JSON list of 5-8 specific topics that are:
1. Related to the book's subject matter
2. NOT actually covered in the content provided  
3. Within reasonable expectations for this type of book

Example: For a 3D photography book missing VR content: ["VR photography", "360-degree cameras", "photogrammetry"]"""
                },
                {
                    "role": "user",
                    "content": f"Book content sample:\n{sample_text}\n\nTopics covered: {covered_list}\n\nWhat related topics are NOT covered?"
                }
            ],
            max_tokens=200,
            temperature=0.2
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response (handle code blocks)
        import json
        try:
            # Remove code block markers if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            content = content.strip()
            topics = json.loads(content)
            if isinstance(topics, list):
                return topics[:8]  # Limit to 8 topics
        except json.JSONDecodeError:
            print(f"JSON parsing failed for not_covered, raw response: {content}")
            
        return []
        
    except Exception as e:
        print(f"Warning: Not-covered topic generation failed: {e}")
        return []


def generate_llm_summary(text: str, max_words: int = 120) -> str:
    """
    Generate a summary using LLM.
    
    Args:
        text: Text to summarize
        max_words: Maximum words in summary
        
    Returns:
        Generated summary
    """
    try:
        response = openai.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"Create a concise abstract of this book content in ≤{max_words} words. "
                               "Focus on the main themes, key concepts, and scope of the material."
                },
                {
                    "role": "user", 
                    "content": text[:8000]  # Limit input to avoid token limits
                }
            ],
            max_tokens=max_words * 2,  # Buffer for safety
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Warning: LLM summary generation failed: {e}")
        return text[:500] + "..." if len(text) > 500 else text


def get_book_chunks(db_path: Path, book_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve all chunks for a given book from the database.
    
    Args:
        db_path: Path to the DuckDB database
        book_id: Book identifier
        
    Returns:
        List of chunk dictionaries
    """
    conn = duckdb.connect(str(db_path))
    try:
        # First try to find chunks with exact book_id match
        query = """
        SELECT chunk_id, book_id, file_name, page_start, page_end, text, 
               chapter_no, chapter_title, heading, heading_level
        FROM chunks 
        WHERE book_id = ?
        ORDER BY chunk_id
        """
        results = conn.execute(query, (book_id,)).fetchall()
        
        # If no exact match, try to get all chunks (for cases where chapters are separate book_ids)
        if not results:
            print(f"No exact match for '{book_id}', checking for chapter-based ingestion...")
            query = """
            SELECT chunk_id, book_id, file_name, page_start, page_end, text, 
                   chapter_no, chapter_title, heading, heading_level
            FROM chunks 
            ORDER BY chunk_id
            """
            results = conn.execute(query).fetchall()
            
            if results:
                print(f"Found {len(results)} chunks from chapter-based ingestion, treating as unified book")
        
        chunks = []
        for row in results:
            chunks.append({
                "chunk_id": row[0],
                "book_id": row[1],  
                "file_name": row[2],
                "page_start": row[3],
                "page_end": row[4],
                "text": row[5],
                "chapter_no": row[6],
                "chapter_title": row[7],
                "heading": row[8],
                "heading_level": row[9]
            })
        
        return chunks
    finally:
        conn.close()


def estimate_page_count(chunks: List[Dict[str, Any]]) -> Optional[int]:
    """
    Estimate total page count from chunk page ranges.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Estimated page count or None if unable to determine
    """
    if not chunks:
        return None
        
    max_page = 0
    for chunk in chunks:
        if chunk.get("page_end"):
            max_page = max(max_page, chunk["page_end"])
    
    return max_page if max_page > 0 else None


def profile_book(book_id: str, db_path: Path, use_llm_summary: bool = False) -> BookMeta:
    """
    Generate book profile metadata from existing chunks.
    
    Args:
        book_id: Book identifier
        db_path: Path to DuckDB database
        use_llm_summary: Whether to use LLM for summary generation
        
    Returns:
        BookMeta object with extracted metadata
    """
    print(f"Profiling book: {book_id}")
    
    # Get all chunks for the book
    chunks = get_book_chunks(db_path, book_id)
    if not chunks:
        raise ValueError(f"No chunks found for book_id: {book_id}")
    
    print(f"Found {len(chunks)} chunks")
    
    # Extract all text content
    all_text = " ".join(chunk["text"] for chunk in chunks)
    chunk_texts = [chunk["text"] for chunk in chunks]
    
    # Extract title (use book_id as fallback, clean up underscores)
    title = book_id.replace("_", " ").title()
    
    # Extract years from content
    all_years = []
    for text in chunk_texts[:10]:  # Check first 10 chunks for efficiency
        years = extract_years_from_text(text)
        all_years.extend(years)
    
    # Filter reasonable years and find range
    reasonable_years = [y for y in all_years if 1400 <= y <= 2030]
    min_year = min(reasonable_years) if reasonable_years else None
    max_year = max(reasonable_years) if reasonable_years else None
    
    # Estimate publication year (assume recent if no years found)
    pub_year = max_year if max_year and max_year >= 1800 else None
    
    # Extract topics using combined approach (LLM + regex + KeyBERT)
    print("Extracting topics with LLM + regex + KeyBERT...")
    topics = extract_combined_topics(chunk_texts[:50])  # Use more chunks for better topic coverage
    
    # Generate abstract
    if use_llm_summary:
        print("Generating LLM summary...")
        abstract = generate_llm_summary(all_text[:4000])  # Use first 4000 chars
    else:
        # Fallback to first 500 characters
        abstract = all_text[:500].strip()
        if len(all_text) > 500:
            # Find the last complete sentence
            last_period = abstract.rfind('.')
            if last_period > 200:  # Ensure we have reasonable content
                abstract = abstract[:last_period + 1]
            else:
                abstract += "..."
    
    # Generate "not_covered" topics using LLM
    print("Identifying topics not covered...")
    not_covered = generate_not_covered_topics(chunk_texts[:30], topics)
    
    # Estimate page count
    pages = estimate_page_count(chunks)
    
    # Create BookMeta object
    book_meta = BookMeta(
        title=title,
        pub_year=pub_year,
        abstract=abstract,
        topics=topics,
        not_covered=not_covered,
        min_year=min_year,
        max_year=max_year,
        pages=pages
    )
    
    return book_meta


def main():
    """Main entry point for the profiling script."""
    parser = argparse.ArgumentParser(
        description="Generate book metadata profile from existing ingested data"
    )
    parser.add_argument(
        "--book-id", 
        required=True,
        help="Book identifier to profile"
    )
    parser.add_argument(
        "--db-path", 
        type=Path,
        help="Path to the DuckDB database file (auto-detected if not provided)"
    )
    parser.add_argument(
        "--no-llm-summary",
        action="store_true",
        help=argparse.SUPPRESS  # Hidden flag for subprocess use only
    )
    
    args = parser.parse_args()
    
    # Auto-detect database path if not provided
    if not args.db_path:
        args.db_path = Path("library") / args.book_id / "build" / "db" / "booker.db"
        print(f"Auto-detected database path: {args.db_path}")
    
    # Validate inputs
    if not args.db_path.exists():
        print(f"Error: Database file not found: {args.db_path}")
        print("Hint: Make sure the book has been ingested first, or provide --db-path explicitly")
        sys.exit(1)
    
    try:
        # Generate book profile (LLM summary by default, unless explicitly disabled for speed)
        use_llm = not args.no_llm_summary
        book_meta = profile_book(args.book_id, args.db_path, use_llm_summary=use_llm)
        
        # Determine output path
        # Follow the pattern: library/<book_id>/book_meta.json
        library_root = Path("library")
        book_dir = library_root / args.book_id
        book_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = book_dir / "book_meta.json"
        
        # Write JSON output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(book_meta.model_dump(), f, indent=2, ensure_ascii=False)
        
        print(f"✅ Book profile written to: {output_path}")
        print(f"Topics found: {', '.join(book_meta.topics[:5])}{'...' if len(book_meta.topics) > 5 else ''}")
        print(f"Year range: {book_meta.min_year}-{book_meta.max_year}")
        print(f"Estimated pages: {book_meta.pages}")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 