"""
Utility functions for handling bi-operational file paths and URLs.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urljoin
import urllib.request
import urllib.error

from . import settings


def download_file_from_url(url: str, local_path: Optional[Path] = None) -> Path:
    """
    Download a file from a URL to a local path.
    
    Args:
        url: The URL to download from
        local_path: Optional local path to save to. If None, uses a temporary file.
    
    Returns:
        Path to the downloaded file
    
    Raises:
        Exception: If download fails
    """
    if local_path is None:
        # Create a temporary file
        fd, temp_path = tempfile.mkstemp()
        os.close(fd)  # Close the file descriptor
        local_path = Path(temp_path)
    
    try:
        print(f"Downloading {url} to {local_path}")
        urllib.request.urlretrieve(url, local_path)
        print(f"Successfully downloaded {url}")
        return local_path
    except urllib.error.URLError as e:
        raise Exception(f"Failed to download {url}: {e}")


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
    In production, this will download required files from S3 to temporary local files.
    
    Args:
        book_id: The book identifier
    
    Returns:
        Dictionary with all relevant paths (local paths, even in production)
    """
    build_base = get_book_build_path(book_id)
    
    if settings.IS_PRODUCTION:
        # In production, download files from S3 to temporary locations
        base_url = settings.DATA_BASE_URL
        if not base_url.endswith('/'):
            base_url += '/'
        
        # Download required files to temporary locations
        db_url = f"{base_url}{book_id}/build/db/booker.db"
        index_url = f"{base_url}{book_id}/build/indexes/booker.faiss"
        meta_url = f"{base_url}{book_id}/build/indexes/booker.pkl"
        
        # Create temporary directory for this book's files
        temp_dir = Path(tempfile.mkdtemp(prefix=f"booker_{book_id}_"))
        
        try:
            db_path = download_file_from_url(db_url, temp_dir / "booker.db")
            index_path = download_file_from_url(index_url, temp_dir / "booker.faiss")
            meta_path = download_file_from_url(meta_url, temp_dir / "booker.pkl")
        except Exception as e:
            # Clean up temp directory on failure
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise e
        
        paths = {
            "db": db_path,
            "index": index_path,
            "meta": meta_path,
            "temp_dir": temp_dir,  # Include temp dir for cleanup
            "title_url": get_book_asset_url(book_id, "assets/title.json"),
            "cover_url": get_book_asset_url(book_id, "assets/cover.png")
        }
    else:
        # Local development - use local file paths
        paths = {
            "db": build_base / "db" / "booker.db",
            "index": build_base / "indexes" / "booker.faiss",
            "meta": build_base / "indexes" / "booker.pkl",
        }
        
        # Add local asset paths
        local_assets = settings.BOOKS_ROOT / book_id / "assets"
        paths["title_url"] = f"/library/{book_id}/assets/title.json"
        paths["cover_url"] = f"/library/{book_id}/assets/cover.png"
        paths["title_path"] = local_assets / "title.json"
        paths["cover_path"] = local_assets / "cover.png"
    
    return paths 