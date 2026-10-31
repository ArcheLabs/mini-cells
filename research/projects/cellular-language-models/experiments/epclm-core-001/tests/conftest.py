import sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(BASE/"implementation"),str(BASE.parent/"native-clm-v0/implementation")]
