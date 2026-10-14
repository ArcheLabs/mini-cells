from pathlib import Path
import sys
import torch
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'src'))
torch.set_num_threads(1)
