"""
CLI module for Booker visualization package.
Provides command-line interface for launching Spotlight visualizations.
"""

import sys
from pathlib import Path

import typer
import renumics.spotlight as spl
from rich.console import Console
from rich.panel import Panel

from .loader import find_project_root, load_vectors, add_kmeans_labels, merge_duckdb_metadata

app = typer.Typer(
    name="booker-viz",
    help="Launch Spotlight visualization for a Booker publication",
    add_completion=False
)

console = Console()

@app.command()
def main(
    pub_id: str = typer.Argument(
        ...,
        help="Publication ID (folder name under /library)"
    ),
    k: int = typer.Option(
        None,
        "--k",
        help="Manual cluster count (default: auto-calculated)"
    ),
    no_cluster: bool = typer.Option(
        False,
        "--no-cluster",
        help="Skip k-means clustering entirely"
    ),
    no_extra_metadata: bool = typer.Option(
        False,
        "--no-extra-metadata", 
        help="Skip DuckDB metadata merge step"
    )
) -> None:
    """
    Launch Spotlight visualization for a publication.
    
    Args:
        pub_id: Publication ID (folder name under /library)
        k: Manual cluster count (auto-calculated if not provided)
        no_cluster: Skip k-means clustering entirely
        no_extra_metadata: Skip DuckDB metadata merge step
    """
    try:
        # Find project root
        root = find_project_root()
        pub_path = root / "library" / pub_id
        
        if not pub_path.exists():
            console.print(f"[red]Error:[/] Publication directory not found at {pub_path}")
            sys.exit(1)
        
        # Load vectors and metadata
        try:
            vectors, df = load_vectors(pub_id)
        except ImportError as e:
            console.print(f"[red]Error:[/] {e}")
            sys.exit(1)
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/] {e}")
            sys.exit(1)
        
        # Merge DuckDB metadata if requested
        if not no_extra_metadata:
            df = merge_duckdb_metadata(df, pub_path)
        
        # Add k-means labels if requested
        if not no_cluster:
            try:
                df = add_kmeans_labels(df, vectors, k)
                console.print(f"[green]Added k-means clustering with {df['kmeans'].nunique()} clusters[/]")
            except ImportError as e:
                console.print(f"[yellow]Warning:[/] {e}")
                console.print("[yellow]Proceeding without clustering...[/]")
        
        # Configure layout - Spotlight will handle UMAP automatically
        layout = None
        
        # Determine color column
        color_by = None
        if "kmeans" in df.columns:
            color_by = "kmeans"
        elif "heading" in df.columns:
            color_by = "heading"
        elif "heading_level" in df.columns:
            color_by = "heading_level"
        
        # Print info
        info_text = (
            f"Loaded {len(df)} vectors from {pub_id}\n"
            f"Vector dimension: {vectors.shape[1]}\n"
            f"Available columns: {', '.join(df.columns)}"
        )
        
        if color_by:
            info_text += f"\nColoring by: {color_by}"
        
        console.print(Panel(
            info_text,
            title="Booker Visualization",
            border_style="green"
        ))
        
        # Launch Spotlight
        spotlight_kwargs = {
            "dtype": {"embedding": spl.Embedding}
        }
        
        # Add layout if we have one
        if layout:
            spotlight_kwargs["layout"] = layout
        
        spl.show(df, **spotlight_kwargs)
        
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)

if __name__ == "__main__":
    app() 