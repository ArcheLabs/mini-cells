"""Explicit dependency boundary around frozen Native X4 and EPCLM-001."""
import json
import sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=next(p for p in BASE.parents if (p/'pyproject.toml').exists())
NATIVE=BASE.parent/'native-clm-v0/implementation'
INCREMENTAL=BASE.parent/'epclm-001-incremental-execution/implementation'
for path in (NATIVE,INCREMENTAL,ROOT,ROOT/'shared/python'):
    if str(path) not in sys.path:sys.path.append(str(path))
from epclm001.common import digest,model_hash,sha,write
import native_clm_runtime as native
import native_clm_phase_c as phase_c
from native_clm_phase_b import PhaseBLM,N1_ALPHA
from epclm001.incremental import IncrementalLM,rope_at


def protocol():return json.loads((BASE/'protocol.json').read_text())


from contextlib import contextmanager
import signal
import time


@contextmanager
def bounded(seconds):
    previous=signal.getitimer(signal.ITIMER_REAL);began=time.monotonic()
    old=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('X4C execution budget exhausted')))
    signal.setitimer(signal.ITIMER_REAL,max(1e-6,min(seconds,previous[0]) if previous[0]>0 else seconds))
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        if previous[0]>0:signal.setitimer(signal.ITIMER_REAL,max(1e-6,previous[0]-(time.monotonic()-began)),previous[1])
