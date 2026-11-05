import sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];CORE=BASE.parent
sys.path[:0]=[str(BASE),str(CORE/'implementation'),str(CORE.parent/'native-clm-v0/implementation')]
