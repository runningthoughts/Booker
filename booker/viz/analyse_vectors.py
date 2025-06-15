"""
Backwards-compatible script that just calls the CLI.
This is kept for compatibility with existing code.
"""

from .cli import main

if __name__ == "__main__":
    import sys
    import typer
    typer.run(main) 