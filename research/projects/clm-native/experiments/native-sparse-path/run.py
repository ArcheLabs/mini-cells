"""Canonical experiment entrypoint; works from any working directory."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from native_sparse_path.cli import main
if __name__ == '__main__':
    main()
