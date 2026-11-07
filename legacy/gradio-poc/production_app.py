"""Convenient local and container launcher for the production package."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from scientific_rag.main import main


if __name__ == "__main__":
    main()
