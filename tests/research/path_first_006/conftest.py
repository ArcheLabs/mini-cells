import sys,json
from pathlib import Path
import pytest,torch
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'research/projects/cellular-language-models/experiments/006-path-first-clm/implementation')]
from exp006.config import load
from exp006.preflight import tiny_config
@pytest.fixture
def c():
    torch.set_num_threads(1);return tiny_config(load()[0])
