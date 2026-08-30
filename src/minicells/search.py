from __future__ import annotations

import csv
import itertools
from copy import deepcopy
from pathlib import Path

from .config import load_config
from .train import train


def pareto_efficient(rows):
    for row in rows:
        row["pareto_efficient"] = not any(
            other["final_token_accuracy"] >= row["final_token_accuracy"]
            and other["final_exact_accuracy"] >= row["final_exact_accuracy"]
            and other["parameter_count"] <= row["parameter_count"]
            and other["estimated_macs"] <= row["estimated_macs"]
            and other != row for other in rows)
    return rows


def run_search(search_config):
    base = load_config(search_config["base_config"]); rows = []
    space = search_config["space"]
    keys = ("num_cells", "hidden_dim", "radius", "iterations")
    for values in itertools.product(*(space[key] for key in keys)):
        for seed in search_config["seeds"]:
            config = deepcopy(base); config["model"].update(dict(zip(keys, values)))
            config["model"]["mlp_width"] = 2 * config["model"]["hidden_dim"]
            config["model"]["max_seq_len"] = min(32, config["model"]["num_cells"])
            config["train"].update(seed=seed, steps=search_config["steps"])
            name = "-".join(f"{key}{value}" for key, value in zip(keys, values))
            config["output"]["root"] = f"results/search/{name}/seed-{seed}"
            report = train(config)
            rows.append({**dict(zip(keys, values)), "seed": seed,
                         "final_token_accuracy": report["metrics"]["final"]["token_accuracy"],
                         "final_exact_accuracy": report["metrics"]["final"]["exact_sequence_accuracy"],
                         "parameter_count": report["model"]["parameter_count"],
                         "estimated_macs": report["model"]["estimated_macs"]})
    pareto_efficient(rows); output = Path("results/search/results.csv"); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    return rows
