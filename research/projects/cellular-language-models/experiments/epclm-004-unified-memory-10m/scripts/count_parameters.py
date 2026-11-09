"""Instantiate E0; count unique static parameters without training."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e0 import E0,Config,parameters
parser=argparse.ArgumentParser()
parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/model-e0.json')
args=parser.parse_args()
print(json.dumps(parameters(E0(Config(**json.loads(args.config.read_text())))),indent=2))
