"""Reuse frozen repository primitives without copying historical models."""
import sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'epclm-x4c-001'))
import common as shared
ROOT=shared.ROOT
