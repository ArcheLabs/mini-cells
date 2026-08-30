#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from minicells.config import load_config
from minicells.search import run_search
p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); run_search(load_config(a.config))
