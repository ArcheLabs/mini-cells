#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from minicells.sample import predict_text
from minicells.train import load_checkpoint
from minicells.vocab import CharVocab
p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--text",required=True); p.add_argument("--device",default="cpu"); a=p.parse_args()
m,_=load_checkpoint(a.checkpoint,a.device); r=predict_text(m,CharVocab(),a.text,a.device)
print(f"input:      {r['input']}\nprediction: {r['prediction']}\nsimilarity: {r['similarity']:.2%}")
