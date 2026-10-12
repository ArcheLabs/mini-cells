#!/usr/bin/env python3
"""Publish curated Core Validation 004 formal results."""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
from publish_experiment_results import DEFAULT_SECRET_NAME,push_results,repo_root
SOURCE=Path("results/core-validation-004-growth-restored-plasticity")
DEST=Path("artifacts/experiments/core-validation-004-growth-restored-plasticity")
BRANCH="kaggle/core-validation-004-growth-restored-plasticity-results"
FORMAT="minicells.core-validation.growth-restored-plasticity.v1"
FILES=("raw.json","decision.json","transaction-records.csv","seed-summary.csv","gate-summary.csv","stability-plasticity-frontier.png","growth-recovery.png","final-state-quality.png","plasticity-recovery-by-seed.png")
def main()->int:
 p=argparse.ArgumentParser(); p.add_argument("--push",action="store_true"); p.add_argument("--branch",default=BRANCH); p.add_argument("--secret-name",default=DEFAULT_SECRET_NAME); args=p.parse_args(); root=repo_root(); source=root/SOURCE; destination=root/DEST; raw=json.loads((source/"raw.json").read_text()); decision=json.loads((source/"decision.json").read_text())
 if raw.get("format")!=FORMAT or decision.get("format")!=FORMAT: raise RuntimeError("unexpected Core Validation 004 format")
 if raw.get("mode")!="formal" or decision.get("scientific_decision") is not True: raise RuntimeError("refusing non-formal result")
 if raw.get("provenance",{}).get("tracked_tree_dirty") is not False: raise RuntimeError("refusing dirty tracked provenance")
 if decision.get("status") not in {"GROWTH_RESTORED_PLASTICITY_SUPPORTED","GROWTH_RESTORED_PLASTICITY_NOT_SUPPORTED"}: raise RuntimeError("unexpected Core Validation 004 status")
 parent=raw.get("parent_experiment",{})
 if parent.get("experiment_id")!="core-validation-003" or parent.get("frozen_outcome")!="DEPENDENCY_SCOPED_TRANSACTIONAL_LEARNING_NOT_SUPPORTED": raise RuntimeError("004 must preserve frozen 003 outcome")
 missing=[n for n in FILES if not (source/n).is_file()]
 if missing: raise FileNotFoundError(f"missing Core Validation 004 outputs: {missing}")
 if destination.exists(): shutil.rmtree(destination)
 destination.mkdir(parents=True); [shutil.copy2(source/n,destination/n) for n in FILES]; shutil.copy2(root/"research/validations/core-004-growth-restored-plasticity/protocol.json",destination/"protocol.json")
 (destination/"RESULTS.md").write_text("# Core Validation 004 Results\n\n" f"- Status: `{decision.get('status')}`\n" f"- Passed seeds: `{decision.get('passed_seeds')}/{decision.get('total_seeds')}`\n" "- Parent 003 remains `DEPENDENCY_SCOPED_TRANSACTIONAL_LEARNING_NOT_SUPPORTED`.\n")
 print(f"Prepared {destination.relative_to(root)}")
 if args.push: push_results(root,destination,"core-validation-004",args.branch,args.secret_name)
 return 0
if __name__=="__main__": raise SystemExit(main())
