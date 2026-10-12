#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(next(_root for _root in Path(__file__).resolve().parents if (_root / ".git").exists() or (_root / "pyproject.toml").exists()) / "src"))
from minicells.config import load_config
from minicells.train import train

parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--seed", type=int); parser.add_argument("--steps", type=int); parser.add_argument("--device")
args = parser.parse_args(); config = load_config(args.config)
if args.seed is not None: config["train"]["seed"] = args.seed
if args.steps is not None: config["train"]["steps"] = args.steps
train(config, args.device)
