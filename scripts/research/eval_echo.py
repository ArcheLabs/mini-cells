#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from minicells.data import fixed_dataset
from minicells.evaluate import evaluate
from minicells.train import load_checkpoint
from minicells.vocab import CharVocab

p=argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--device", default="cpu"); a=p.parse_args()
model,payload=load_checkpoint(a.checkpoint,a.device); c=payload["config"]; v=CharVocab()
b=fixed_dataset(v,c["validation"]["seed"],c["validation"]["examples"],min_length=c["data"]["min_length"],max_length=c["data"]["max_length"],num_cells=c["model"]["num_cells"],random_fraction=c["data"]["random_fraction"]).to(a.device)
print(evaluate(model,b))
