"""Experiment 006 canonical research entrypoint."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research/08-hybrid-clm/experiments/006-path-first-clm/implementation')]
from exp006.publish import main
if __name__=='__main__':main()
