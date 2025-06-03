"""
Utility functions for handling bi-operational file paths and URLs.
"""

from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

from . import settings


def get_book_asset_url(book_id: str, asset_path: str) -> str:
    """
    Get the URL for a book asset, handling both local and S3 environments.
    
    Args:
        book_id: The book identifier
        asset_path: Relative path to the asset (e.g., "assets/title.json", "assets/cover.png")
    
    Returns:
        Full URL to the asset
    """
    if settings.IS_PRODUCTION:
        # Running on Render - construct S3 URL
        base_url = settings.DATA_BASE_URL
        if not base_url.endswith('/'):
            base_url += '/'
        return urljoin(base_url, f"{book_id}/{asset_path}")
    else:
        # Running locally - use local file path via API
        return f"/library/{book_id}/{asset_path}"


def get_book_source_path(book_id: str) -> Path:
    """
    Get the local path to book source files for ingestion.
    This is always local, even in production, as ingestion happens offline.
    
    Args:
        book_id: The book identifier
    
    Returns:
        Path to the source directory
    """
    return settings.BOOKS_ROOT / book_id / "source"


def get_book_build_path(book_id: str) -> Path:
    """
    Get the local path to book build artifacts (db, indexes).
    This is always local as the processed data stays on the server.
    
    Args:
        book_id: The book identifier
    
    Returns:
        Path to the build directory
    """
    return settings.BOOKS_ROOT / book_id / "build"


def resolve_book_paths(book_id: str) -> dict:
    """
    Resolve all paths needed for a book, handling both environments.
    
    Args:
        book_id: The book identifier
    
    Returns:
        Dictionary with all relevant paths
    """
    if settings.IS_PRODUCTION:
        # In production, use S3 URLs for database files (no /library/ prefix)
        base_url = settings.DATA_BASE_URL
        if not base_url.endswith('/'):
            base_url += '/'
        
        paths = {
            "db": f"{base_url}{book_id}/build/db/booker.db",
            "index": f"{base_url}{book_id}/build/indexes/booker.faiss",
            "meta": f"{base_url}{book_id}/build/indexes/booker.pkl",
            "title_url": get_book_asset_url(book_id, "assets/title.json"),
            "cover_url": get_book_asset_url(book_id, "assets/cover.png")
        }
    else:
        # Local development - use local file paths
        build_base = get_book_build_path(book_id)
        
        paths = {
            "db": build_base / "db" / "booker.db",
            "index": build_base / "indexes" / "booker.faiss",
            "meta": build_base / "indexes" / "booker.pkl",
        }
        
        # Add asset URLs for local environment
        local_assets = settings.BOOKS_ROOT / book_id / "assets"
        paths["title_url"] = f"/library/{book_id}/assets/title.json"
        paths["cover_url"] = f"/library/{book_id}/assets/cover.png"
        paths["title_path"] = local_assets / "title.json"
        paths["cover_path"] = local_assets / "cover.png"
    
    return paths 