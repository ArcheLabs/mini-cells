"""Experiment 006 canonical research entrypoint."""
from pathlib import Path
import sys
ROOT=next(parent for parent in Path(__file__).resolve().parents if (parent/'pyproject.toml').is_file())
sys.path[:0]=[str(ROOT),str(ROOT/'research/projects/cellular-language-models/experiments/006-path-first-clm/implementation')]
from exp006.cli import main
if __name__=='__main__':main()
