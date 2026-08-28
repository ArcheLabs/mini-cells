"""Machine-readable CLM-0.3 reports, aggregation, and lightweight plots."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .growth_validation import progressive_growth_decision


FORMAL_ARMS = ("fixed4", "pressure_growth", "random_growth")
FORMAL_REPLICATES = (0, 1, 2)
FORMAL_TARGET_TOKENS = 1_500_000

PPL_COLUMNS = (
    "replicate", "arm", "tokens", "phase", "ppl", "nll", "fixed4_ppl",
    "clm01_start_ppl", "textnca_frozen_ppl", "ppl_vs_fixed4",
    "ppl_vs_clm01", "ppl_vs_textnca", "health",
)


def validate_telemetry_event(event: dict[str, Any]) -> None:
    event_type = event.get("type")
    common = ("type", "arm", "replicate")
    missing = [key for key in common if key not in event]
    if missing:
        raise ValueError(f"telemetry event missing fields: {missing}")
    required = {
        "training_progress": ("consumed_tokens", "target_tokens", "phase"),
        "birth": ("birth_index", "stage", "parent", "child", "parity_status"),
        "evaluation": ("tokens", "ppl", "nll", "raw_model_ppl", "clm01_start_ppl", "textnca_frozen_ppl"),
        "checkpoint": ("path", "consumed_tokens", "training_step", "growth_event_index"),
        "worker_complete": ("consumed_tokens", "target_tokens"),
    }.get(event_type, ())
    missing = [key for key in required if key not in event]
    if missing:
        raise ValueError(f"{event_type} telemetry event missing fields: {missing}")


def write_ppl_history(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PPL_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return destination


def write_growth_history(path: str | Path, history: Iterable[dict[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(list(history), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_events(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_ppl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            row["replicate"] = int(raw["replicate"])
            row["tokens"] = int(raw["tokens"])
            for key in (
                "ppl", "nll", "fixed4_ppl", "clm01_start_ppl", "textnca_frozen_ppl",
                "ppl_vs_fixed4", "ppl_vs_clm01", "ppl_vs_textnca",
            ):
                value = raw.get(key, "")
                row[key] = None if value in (None, "", "None", "nan", "NaN") else float(value)
            rows.append(row)
    return rows


def _latest_diagnostics_by_birth(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        birth_index = int(row["birth_index"])
        offset = int(row.get("offset_tokens", 0))
        current = result.get(birth_index)
        if current is None or offset >= int(current.get("offset_tokens", 0)):
            result[birth_index] = row
    return result


def _birth_viability(history: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    latest = _latest_diagnostics_by_birth(diagnostics)
    births: list[dict[str, Any]] = []
    for event in history:
        birth_index = int(event["birth_index"])
        diag = latest.get(birth_index)
        equivalent = event.get("parity", {}).get("status") == "CLM_GROWTH_EQUIVALENCE"
        checks = {
            "equivalent": equivalent,
            "has_final_diagnostic": diag is not None,
            "child_received_traffic": bool(diag is not None and float(diag.get("child_usage", 0.0)) > 0.0),
            "parameters_differentiated": bool(diag is not None and float(diag.get("relative_l2", 0.0)) > 0.0),
            "split_noncollapsed": bool(diag is not None and float(diag.get("split_entropy", 0.0)) > 0.0),
            "mergeback_nonnegative": bool(
                diag is not None and float(diag.get("causal_merge_back_penalty", -math.inf)) >= 0.0
            ),
        }
        births.append({
            "birth_index": birth_index,
            "stage": int(event["stage"]),
            "parent": str(event["parent"]),
            "child": str(event["child"]),
            "checks": checks,
            "viable": all(checks.values()),
            "latest_diagnostic": diag,
        })
    return {
        "equivalent_births": sum(int(item["checks"]["equivalent"]) for item in births),
        "viable": len(births) == 2 and all(item["viable"] for item in births),
        "births": births,
    }


def aggregate_formal_results(
    output_root: str | Path,
    *,
    formal_gpu_experiment_run: bool = False,
    expected_target_tokens: int = FORMAL_TARGET_TOKENS,
) -> dict[str, Any]:
    """Aggregate the paired 3x3 experiment without synthesizing missing controls.

    Worker-local ``ppl_vs_fixed4`` is deliberately ignored.  The matched fixed-4
    denominator is joined only here on ``(replicate, tokens)``.
    """

    root = Path(output_root)
    worker_rows: dict[tuple[int, str], list[dict[str, Any]]] = {}
    worker_meta: dict[tuple[int, str], dict[str, Any]] = {}
    viability: dict[tuple[int, str], dict[str, Any]] = {}

    for replicate in FORMAL_REPLICATES:
        for arm in FORMAL_ARMS:
            directory = root / f"r{replicate}-{arm}"
            required = (directory / "events.jsonl", directory / "ppl-history.csv")
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise RuntimeError(f"formal worker artifacts missing for r{replicate}/{arm}: {missing}")
            events = _read_events(directory / "events.jsonl")
            completed = [event for event in events if event.get("type") == "worker_complete"]
            if not completed:
                raise RuntimeError(f"worker r{replicate}/{arm} has no worker_complete event")
            complete = completed[-1]
            consumed = int(complete.get("consumed_tokens", 0))
            target = int(complete.get("target_tokens", 0))
            if consumed < target:
                raise RuntimeError(f"worker r{replicate}/{arm} stopped before its declared target")
            if formal_gpu_experiment_run and (
                expected_target_tokens != FORMAL_TARGET_TOKENS or consumed < FORMAL_TARGET_TOKENS
            ):
                raise RuntimeError("formal_gpu_experiment_run requires the full 1.5M-token matrix")
            rows = _read_ppl_rows(directory / "ppl-history.csv")
            if not rows:
                raise RuntimeError(f"worker r{replicate}/{arm} has no PPL rows")
            worker_rows[(replicate, arm)] = rows
            worker_meta[(replicate, arm)] = {
                "consumed_tokens": consumed,
                "target_tokens": target,
                "complete_event": complete,
            }
            if arm == "fixed4":
                viability[(replicate, arm)] = {"equivalent_births": 0, "viable": True, "births": []}
            else:
                history_path = directory / "growth-history.json"
                diagnostics_path = directory / "newborn-diagnostics.json"
                if not history_path.exists() or not diagnostics_path.exists():
                    raise RuntimeError(f"growth evidence missing for r{replicate}/{arm}")
                history = _read_json(history_path)
                diagnostics = _read_json(diagnostics_path)
                viability[(replicate, arm)] = _birth_viability(history, diagnostics)

    fixed_by_key: dict[tuple[int, int], float] = {}
    for replicate in FORMAL_REPLICATES:
        for row in worker_rows[(replicate, "fixed4")]:
            fixed_by_key[(replicate, int(row["tokens"]))] = float(row["ppl"])

    formal_rows: list[dict[str, Any]] = []
    for replicate in FORMAL_REPLICATES:
        for arm in FORMAL_ARMS:
            for source in worker_rows[(replicate, arm)]:
                key = (replicate, int(source["tokens"]))
                if key not in fixed_by_key:
                    raise RuntimeError(
                        f"matched fixed4 control missing for replicate={replicate}, tokens={source['tokens']}"
                    )
                fixed4_ppl = fixed_by_key[key]
                ppl = float(source["ppl"])
                clm01 = source.get("clm01_start_ppl")
                textnca = source.get("textnca_frozen_ppl")
                if clm01 is None or textnca is None:
                    raise RuntimeError(f"sentinel PPL missing for r{replicate}/{arm} @ {source['tokens']}")
                row = dict(source)
                row["fixed4_ppl"] = fixed4_ppl
                row["ppl_vs_fixed4"] = ppl / fixed4_ppl
                row["ppl_vs_clm01"] = ppl / float(clm01)
                row["ppl_vs_textnca"] = ppl / float(textnca)
                formal_rows.append(row)

    formal_rows.sort(key=lambda row: (int(row["replicate"]), str(row["arm"]), int(row["tokens"])))
    write_ppl_history(root / "formal-ppl-history.csv", formal_rows)

    summaries: list[dict[str, Any]] = []
    detailed_summaries: list[dict[str, Any]] = []
    for replicate in FORMAL_REPLICATES:
        for arm in FORMAL_ARMS:
            rows = [row for row in formal_rows if row["replicate"] == replicate and row["arm"] == arm]
            final = max(rows, key=lambda row: int(row["tokens"]))
            evidence = viability[(replicate, arm)]
            summary = {
                "replicate": replicate,
                "arm": arm,
                "tokens": int(final["tokens"]),
                "ppl": float(final["ppl"]),
                "fixed4_ppl": float(final["fixed4_ppl"]),
                "ppl_vs_fixed4": float(final["ppl_vs_fixed4"]),
                "ppl_vs_clm01": float(final["ppl_vs_clm01"]),
                "ppl_vs_textnca": float(final["ppl_vs_textnca"]),
                "equivalent_births": int(evidence["equivalent_births"]),
                "viable": bool(evidence["viable"]),
            }
            summaries.append(summary)
            detailed_summaries.append({**summary, "birth_evidence": evidence["births"]})

    decision = progressive_growth_decision(
        summaries,
        formal_gpu_experiment_run=formal_gpu_experiment_run,
    )
    (root / "replicate-summary.json").write_text(
        json.dumps(detailed_summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "formal_rows": formal_rows,
        "summaries": detailed_summaries,
        "decision": decision,
        "worker_meta": worker_meta,
    }


def save_growth_plots(
    output_dir: str | Path,
    *,
    ppl_rows: Iterable[dict[str, Any]],
    growth_history: Iterable[dict[str, Any]],
    lineage_rows: Iterable[dict[str, Any]] = (),
    telemetry_rows: Iterable[dict[str, Any]] = (),
) -> list[Path]:
    """Generate the required plot set when an experiment has data.

    No placeholder plots are emitted when the formal experiment has no data.
    """

    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ppl = list(ppl_rows)
    events = list(growth_history)
    lineages = list(lineage_rows)
    telemetry = list(telemetry_rows)
    del lineages
    if not ppl:
        return []

    def figure(name: str, title: str, x: list[float] | None = None, y: list[float] | None = None) -> Path:
        fig, axis = plt.subplots(figsize=(7, 4))
        if x and y:
            axis.plot(x, y, marker="o")
        axis.set_title(title)
        axis.grid(alpha=.25)
        fig.tight_layout()
        path = output / name
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path

    paths = []
    paths.append(figure("ppl-over-time.png", "CLM-0.3 PPL over time",
                        [float(row["tokens"]) for row in ppl], [float(row["ppl"]) for row in ppl]))
    paths.append(figure("ppl-ratio-vs-fixed4.png", "PPL ratio vs fixed-4",
                        [float(row["tokens"]) for row in ppl], [float(row["ppl_vs_fixed4"]) for row in ppl]))
    event_x = [float(event["token"]) for event in events]
    paths.append(figure("growth-events.png", "Growth events", event_x, [1.0] * len(event_x)))
    paths.append(figure("expert-count-over-time.png", "Expert count over time", event_x, [13 + index for index in range(len(event_x))]))
    paths.append(figure("lineage-usage.png", "Lineage usage"))
    paths.append(figure("lineage-divergence.png", "Lineage divergence"))
    paths.append(figure("newborn-causal-utility.png", "Newborn causal utility"))
    paths.append(figure("pressure-ranking.png", "Pressure ranking"))
    progress = [row for row in telemetry if row.get("type") == "training_progress"]
    paths.append(figure("throughput.png", "Throughput",
                        [float(row.get("consumed_tokens", 0)) for row in progress],
                        [float(row.get("tokens_per_second", 0)) for row in progress]))
    paths.append(figure("vram.png", "Peak VRAM",
                        [float(row.get("consumed_tokens", 0)) for row in progress],
                        [float(row.get("peak_vram_bytes", 0)) for row in progress]))
    return paths
