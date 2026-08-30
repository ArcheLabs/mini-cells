#!/usr/bin/env python3
"""Report Core Validation 004 formal/smoke outputs."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
DEFAULT_OUT=ROOT/"results"/"core-validation-004-growth-restored-plasticity"
def parse_args():
 p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,default=DEFAULT_OUT); return p.parse_args()
def _records(payload:dict[str,Any])->pd.DataFrame:
 rows=[]
 for run in payload["runs"]:
  for variant,vrun in run["variants"].items():
   for record in vrun["records"]:
    row={"seed":run["seed"],"variant":variant,**record}; row["attempts"]=json.dumps(row["attempts"]); rows.append(row)
 return pd.DataFrame(rows)
def _summary(payload:dict[str,Any])->pd.DataFrame:
 rows=[]
 for run in payload["runs"]:
  pre=run["gate_summary"]["pretraining"]
  for variant,vrun in run["variants"].items(): rows.append({"seed":run["seed"],"variant":variant,**pre,**vrun["summary"]})
 return pd.DataFrame(rows)
def _gates(payload:dict[str,Any])->pd.DataFrame:
 rows=[]
 for run in payload["runs"]:
  g=run["gate_summary"]; growth=g["variant_summaries"]["local_tx_growth"]
  rows.append({"seed":run["seed"],"pass":g["pass"],"regression_damage_ratio_vs_local_always":g["regression_damage_ratio_vs_local_always"],"committed_gain_ratio_vs_local_always":g["committed_gain_ratio_vs_local_always"],"final_mutable_nrmse_ratio_vs_local_always":g["final_mutable_nrmse_ratio_vs_local_always"],**{f"gate_{k}":v for k,v in g["gates"].items()},**{f"growth_{k}":v for k,v in growth.items()}})
 return pd.DataFrame(rows)
def _plot_stability_plasticity(summary:pd.DataFrame,dest:Path):
 if summary.empty:return
 grouped=summary.groupby("variant",as_index=False).agg(gain=("cumulative_committed_new_gain","mean"),damage=("cumulative_positive_global_regression","mean"),accept=("effective_acceptance_rate","mean"))
 fig,ax=plt.subplots(figsize=(8,6)); ax.scatter(grouped["damage"],grouped["gain"],s=70)
 for _,r in grouped.iterrows(): ax.annotate(r["variant"],(r["damage"],r["gain"]),xytext=(5,5),textcoords="offset points")
 ax.set_xlabel("Cumulative positive global regression"); ax.set_ylabel("Cumulative committed new-learning gain"); ax.set_title("Core Validation 004: stability–plasticity frontier"); fig.tight_layout(); fig.savefig(dest,dpi=180); plt.close(fig)
def _plot_growth(summary:pd.DataFrame,dest:Path):
 rows=summary[summary["variant"]=="local_tx_growth"]
 if rows.empty:return
 metrics=["effective_acceptance_rate","growth_rescue_rate","private_cell_reuse_acceptance_rate","spawned_cells_per_effective_commit"]
 vals=[rows[m].mean() for m in metrics]
 fig,ax=plt.subplots(figsize=(9,5)); ax.bar(metrics,vals); ax.tick_params(axis="x",rotation=20); ax.set_ylabel("Fraction / ratio"); ax.set_title("Growth recovery and boundedness"); fig.tight_layout(); fig.savefig(dest,dpi=180); plt.close(fig)
def _plot_final(summary:pd.DataFrame,dest:Path):
 if summary.empty:return
 grouped=summary.groupby("variant",as_index=False).agg(anchor=("anchor_normalized_mse","mean"),mutable=("mutable_normalized_mse","mean"))
 fig,ax=plt.subplots(figsize=(8,5)); x=range(len(grouped)); w=.35; ax.bar([i-w/2 for i in x],grouped["anchor"],width=w,label="anchor"); ax.bar([i+w/2 for i in x],grouped["mutable"],width=w,label="mutable"); ax.set_xticks(list(x),grouped["variant"],rotation=15); ax.set_ylabel("Normalized MSE"); ax.set_title("Final-state retention and plasticity"); ax.legend(); fig.tight_layout(); fig.savefig(dest,dpi=180); plt.close(fig)
def _plot_gain_ratio(gates:pd.DataFrame,dest:Path):
 if gates.empty:return
 fig,ax=plt.subplots(figsize=(8,5)); ax.bar(gates["seed"].astype(str),gates["committed_gain_ratio_vs_local_always"]); ax.axhline(.80,linestyle="--",linewidth=1,label="plasticity gate"); ax.set_xlabel("Seed"); ax.set_ylabel("Growth / local-always committed gain"); ax.set_title("Plasticity recovery by formal seed"); ax.legend(); fig.tight_layout(); fig.savefig(dest,dpi=180); plt.close(fig)
def main()->int:
 args=parse_args(); raw=args.out/"raw.json"
 if not raw.is_file(): raise FileNotFoundError(raw)
 payload=json.loads(raw.read_text()); args.out.mkdir(parents=True,exist_ok=True); rec=_records(payload); summary=_summary(payload); gates=_gates(payload); rec.to_csv(args.out/"transaction-records.csv",index=False); summary.to_csv(args.out/"seed-summary.csv",index=False); gates.to_csv(args.out/"gate-summary.csv",index=False)
 decision={**payload["decision"],"format":payload["format"],"experiment_id":payload["experiment_id"],"mode":payload["mode"],"protocol_sha256":payload["protocol_sha256"],"parent_experiment":payload["parent_experiment"],"elapsed_seconds":payload["elapsed_seconds"],"provenance":payload["provenance"]}; (args.out/"decision.json").write_text(json.dumps(decision,indent=2,sort_keys=True)+"\n")
 _plot_stability_plasticity(summary,args.out/"stability-plasticity-frontier.png"); _plot_growth(summary,args.out/"growth-recovery.png"); _plot_final(summary,args.out/"final-state-quality.png"); _plot_gain_ratio(gates,args.out/"plasticity-recovery-by-seed.png"); print(json.dumps(decision,indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
