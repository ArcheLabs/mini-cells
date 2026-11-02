import sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];CORE=BASE.parent
sys.path[:0]=[str(BASE),str(CORE/'gc001-baseline-repair'),str(CORE/'implementation'),str(CORE/'gc002-fair-architecture'),str(CORE.parent/'native-clm-v0/implementation')]
