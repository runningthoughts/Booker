"""
Loader module for Booker visualization package.
Handles loading FAISS vectors and metadata from disk.
"""

import pickle
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import faiss

def find_project_root() -> Path:
    """
    Find the project root directory by looking for setup.py.
    
    Returns:
        Path to project root directory
        
    Raises:
        FileNotFoundError: If setup.py not found
    """
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "setup.py").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root (setup.py)")

def add_kmeans_labels(df: pd.DataFrame, 
                      vectors: np.ndarray, 
                      n_clusters: int | None = None) -> pd.DataFrame:
    """
    Add a k-means cluster label column (category dtype).
    
    Args:
        df: DataFrame to add cluster labels to
        vectors: Vector embeddings for clustering
        n_clusters: Number of clusters (auto-calculated if None)
        
    Returns:
        DataFrame with added 'kmeans' column
    """
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        raise ImportError("Please install scikit-learn: pip install -r requirements-viz.txt")
    
    if n_clusters is None:
        n_clusters = max(3, len(df) // 25)   # ~25 pts/cluster
    
    km = MiniBatchKMeans(n_clusters, random_state=42)
    labels = km.fit_predict(vectors)
    df["kmeans"] = pd.Categorical(labels)
    return df

def merge_duckdb_metadata(df: pd.DataFrame, root_path: Path) -> pd.DataFrame:
    """
    Merge additional metadata from DuckDB if available.
    
    Args:
        df: DataFrame to merge metadata into
        root_path: Root path of the publication
        
    Returns:
        DataFrame with merged metadata (unchanged if DuckDB not available)
    """
    try:
        import duckdb
    except ImportError:
        # DuckDB not available, return df unchanged
        return df
    
    duckdb_path = root_path / "build" / "booker.duckdb"
    if not duckdb_path.exists():
        # DuckDB file doesn't exist, return df unchanged
        return df
    
    try:
        # Connect to DuckDB and query metadata
        conn = duckdb.connect(str(duckdb_path))
        metadata_df = conn.execute(
            "SELECT chunk_id, heading_level, importance_name FROM chunks"
        ).fetchdf()
        conn.close()
        
        # Left join onto df by chunk_id
        if 'chunk_id' in df.columns:
            df = df.merge(metadata_df, on='chunk_id', how='left')
        
        return df
        
    except Exception as e:
        # Any error with DuckDB operations, return df unchanged
        print(f"[Booker] Warning: Could not load DuckDB metadata: {e}")
        return df

def load_vectors(pub_id: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load FAISS vectors and metadata for a publication.
    
    Args:
        pub_id: Publication ID (folder name under /library)
        
    Returns:
        Tuple of (vectors array, metadata DataFrame)
        
    Raises:
        FileNotFoundError: If required files not found
        ImportError: If FAISS not installed
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("Please run 'pip install -r requirements.txt' to install FAISS")
    
    # Find project root and construct paths
    root = find_project_root()
    pub_path = root / "library" / pub_id
    index_path = pub_path / "build" / "indexes" / "booker.faiss"
    meta_path = pub_path / "build" / "indexes" / "booker.pkl"
    
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found at {meta_path}")
    
    # Load FAISS index
    index = faiss.read_index(str(index_path))
    
    # Get vectors from index
    vectors = np.zeros((index.ntotal, index.d), dtype=np.float32)
    index.reconstruct_n(0, index.ntotal, vectors)
    
    # Load metadata
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(metadata)
    
    # Add embedding column
    df['embedding'] = list(vectors)
    
    return vectors, df 