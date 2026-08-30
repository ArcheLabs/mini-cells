"""CLM-0.4 Preview: product-oriented continual learning with public telemetry.

Preview deliberately separates product iteration from the frozen M1-v1/v2
validation history. Capability metrics are observable targets, not admission
gates. Runtime safety still uses dependency-scoped transactional validation.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Iterator

import torch

from .curriculum import build_curriculum, materialize_transaction, transaction_specs
from .data import BaseShardWriter
from .engine import VariantHarness
from .examples import tokenize_examples
from .model import MiniCLMConfig, TinyCLMDecoder
from .performance import install_runtime_patches, resolve_cuda_devices, train_base_model_parallel
from .protocol import CandidateOptimizerConfig, canonical_json_hash, file_sha256, load_protocol, m1_thresholds
from .state import model_state_hash
from .tokenizer import DigitAwareTokenizerBundle, train_digit_aware_tokenizer
from .training import BaseCorpusDataset, BaseTrainConfig, base_cell_activation_counts
from .v2 import (
    teacher_forced_answer_exact_accuracy,
    v2_math_eval_examples,
    v2_math_stream,
    v2_story_eval_examples,
    v2_story_stream,
)


PREVIEW_FORMAT = "minicells.clm-0.4-preview.v1"
PREVIEW_ASSET_FORMAT = "minicells.clm-0.4-preview.assets.v1"
PREVIEW_ROUTING_SALT = "clm-0.4-preview-v1"
PREVIEW_BASE_CORPUS_VERSION = "clm-0.4-preview-base-corpus-v1"
PREVIEW_MIXTURE = {
    "language_carrier": 0.60,
    "controlled_base_math": 0.30,
    "controlled_base_story": 0.10,
}
PREVIEW_SEED = 90500
PREVIEW_BASE_CELLS = 64  # 32 Cells in each of the two sparse layers.
PREVIEW_SHARED_CELL_FF_HIDDEN = 256
DEFAULT_DIRECT = CandidateOptimizerConfig("AdamW", 32, 0.003, 32, 0.0)
DEFAULT_GROWTH = CandidateOptimizerConfig("AdamW", 32, 0.003, 64, 0.0)


def preview_model_config() -> MiniCLMConfig:
    return MiniCLMConfig(
        vocab_size=8192,
        max_seq_len=256,
        num_layers=4,
        d_model=256,
        n_heads=8,
        dense_ff_hidden=768,
        base_cells=32,
        cell_hidden=32,
        routing_salt=PREVIEW_ROUTING_SALT,
        shared_cell_ff_hidden=PREVIEW_SHARED_CELL_FF_HIDDEN,
    )


def preview_math_stream(seed: int = 6101) -> Iterator[str]:
    yield from v2_math_stream(seed)


def preview_story_stream(seed: int = 6201) -> Iterator[str]:
    yield from v2_story_stream(seed)


def preview_tokenizer_training_texts(
    carrier_texts: Iterable[str],
    *,
    math_examples: int = 4096,
    story_examples: int = 2048,
) -> list[str]:
    values = list(carrier_texts)
    values.extend(itertools.islice(preview_math_stream(), int(math_examples)))
    values.extend(itertools.islice(preview_story_stream(), int(story_examples)))
    return values


class PreviewBaseShardWriter(BaseShardWriter):
    def build(
        self,
        *,
        carrier_texts: Iterable[str],
        carrier_source: dict,
        math_seed: int = 6101,
        story_seed: int = 6201,
    ) -> dict:
        targets = {
            category: int(round(self.target_tokens * fraction))
            for category, fraction in self.mixture.items()
        }
        self._fill_category("language_carrier", carrier_texts, targets["language_carrier"])
        self._fill_category("controlled_base_math", preview_math_stream(math_seed), targets["controlled_base_math"])
        self._fill_category("controlled_base_story", preview_story_stream(story_seed), targets["controlled_base_story"])
        self._flush()
        address_path = self.out_dir / "address-table.json"
        address_path.write_text(json.dumps(self.address_pool, indent=2) + "\n", encoding="utf-8")
        actual_total = sum(self.category_tokens.values())
        manifest = {
            "format": "minicells.clm-0.4-preview.base-corpus-manifest.v1",
            "generator_version": PREVIEW_BASE_CORPUS_VERSION,
            "target_tokens": self.target_tokens,
            "actual_tokens": actual_total,
            "mixture_target": self.mixture,
            "category_tokens": self.category_tokens,
            "category_fractions": {
                key: value / float(max(1, actual_total))
                for key, value in self.category_tokens.items()
            },
            "model_sequence_length": self.cfg.max_seq_len,
            "tokenizer_vocab_size": self.tokenizer.vocab_size,
            "routing_salt": self.cfg.routing_salt,
            "base_address_pool_size": len(self.address_pool),
            "address_table": {"path": address_path.name, "sha256": file_sha256(address_path)},
            "carrier_source": dict(carrier_source),
            "controlled_seeds": {"math": int(math_seed), "story": int(story_seed)},
            "controlled_task_alignment": {
                "math": "question-answer",
                "story": "context-conditioned-retrieval-qa",
            },
            "digit_policy": "individual-decimal-digits-before-bpe",
            "shards": self._shards,
        }
        manifest["manifest_sha256"] = canonical_json_hash(manifest)
        (self.out_dir / "base-corpus-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest


def prepare_preview_data_assets(
    *,
    out_dir: str | Path,
    tokenizer_training_texts: Iterable[str],
    carrier_texts: Iterable[str],
    carrier_source: dict,
    target_tokens: int = 30_000_000,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_manifest = train_digit_aware_tokenizer(
        tokenizer_training_texts,
        out_dir=out / "tokenizer",
        vocab_size=8192,
        source_manifest={**carrier_source, "preview_controlled_templates_included": True},
    )
    tokenizer = DigitAwareTokenizerBundle.load(out / "tokenizer" / "tokenizer.json")
    cfg = preview_model_config()
    writer = PreviewBaseShardWriter(
        tokenizer=tokenizer,
        model_config=cfg,
        out_dir=out / "base-corpus",
        target_tokens=int(target_tokens),
        mixture=dict(PREVIEW_MIXTURE),
    )
    base_manifest = writer.build(carrier_texts=carrier_texts, carrier_source=carrier_source)
    curriculum = build_curriculum()
    (out / "curriculum-manifest.json").write_text(
        json.dumps(curriculum, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "format": PREVIEW_ASSET_FORMAT,
        "routing_salt": PREVIEW_ROUTING_SALT,
        "dataset_revision": str(carrier_source.get("revision")),
        "target_tokens": int(target_tokens),
        "base_tokens": int(base_manifest["actual_tokens"]),
        "mixture": dict(PREVIEW_MIXTURE),
        "tokenizer_version": tokenizer_manifest["tokenizer_version"],
        "tokenizer_hash": tokenizer_manifest["tokenizer_sha256"],
        "tokenizer_manifest_hash": tokenizer_manifest["manifest_sha256"],
        "base_corpus_manifest_hash": base_manifest["manifest_sha256"],
        "curriculum_manifest_hash": curriculum["manifest_sha256"],
        "base_generator_version": PREVIEW_BASE_CORPUS_VERSION,
    }
    (out / "asset-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _capability_metrics(
    model: TinyCLMDecoder,
    tokenizer: DigitAwareTokenizerBundle,
    device: torch.device,
) -> dict[str, float]:
    cfg = model.cfg
    math_examples = v2_math_eval_examples(cfg, 64, seed=6301)
    story_examples = v2_story_eval_examples(cfg, 64, seed=6401)
    return {
        "math_teacher_forced_answer_exact": teacher_forced_answer_exact_accuracy(
            model, math_examples, tokenizer=tokenizer, max_seq_len=cfg.max_seq_len, device=device
        ),
        "story_teacher_forced_answer_exact": teacher_forced_answer_exact_accuracy(
            model, story_examples, tokenizer=tokenizer, max_seq_len=cfg.max_seq_len, device=device
        ),
    }


def _base_probes(model: TinyCLMDecoder, tokenizer: DigitAwareTokenizerBundle) -> list:
    texts = [
        *v2_math_eval_examples(model.cfg, 64, seed=6301),
        *v2_story_eval_examples(model.cfg, 64, seed=6401),
    ]
    return tokenize_examples(texts, tokenizer, max_seq_len=model.cfg.max_seq_len)


def _tokenized_transaction(spec, tokenizer: DigitAwareTokenizerBundle, max_seq_len: int) -> dict:
    materialized = materialize_transaction(spec, smoke=False)
    return {
        split: tokenize_examples(items, tokenizer, max_seq_len=max_seq_len)
        for split, items in materialized.items()
    }


def _attempt_value(record: dict, key: str, default: float = 0.0) -> float:
    attempts = record.get("attempts", [])
    if not attempts:
        return float(default)
    return float(attempts[-1].get(key, default))


def _timeline_row(harness: VariantHarness, record: dict, capability: dict[str, float] | None) -> dict[str, Any]:
    records = harness.records
    commits = [r for r in records if r["final_decision"] != "rollback"]
    direct = sum(r["final_decision"] == "direct-commit" for r in records)
    growth = sum(r["final_decision"] == "growth-commit" for r in records)
    reuse = sum(r["final_decision"] == "private-reuse-commit" for r in records)
    rollbacks = len(records) - len(commits)
    growth_attempts = sum(bool(r.get("growth_attempted")) for r in records)
    bundles = len(harness.model.private_addresses())
    private_cells = bundles * 2
    base_parameters = sum(
        p.numel() for name, p in harness.model.named_parameters() if "private_cells" not in name
    )
    total_parameters = sum(p.numel() for p in harness.model.parameters())
    training_seconds = sum(
        float(a.get("candidate_wall_seconds", 0.0))
        for r in records for a in r.get("attempts", [])
    )
    validation_seconds = sum(
        float(a.get("validation_wall_seconds", 0.0))
        for r in records for a in r.get("attempts", [])
    )
    transaction_seconds = sum(float(r.get("transaction_wall_seconds", 0.0)) for r in records)
    local_pass_attempts = [
        a for r in records for a in r.get("attempts", []) if a.get("local_pass")
    ]
    false_safe = sum(bool(a.get("false_safe")) for a in local_pass_attempts)
    row: dict[str, Any] = {
        "transaction_id": int(record["transaction_id"]),
        "transactions": len(records),
        "final_decision": record["final_decision"],
        "commits": len(commits),
        "rollbacks": rollbacks,
        "direct_commits": direct,
        "growth_commits": growth,
        "private_reuse_commits": reuse,
        "growth_attempts": growth_attempts,
        "growth_rescue_rate": growth / float(max(1, growth_attempts)),
        "acceptance_rate": len(commits) / float(max(1, len(records))),
        "base_cells": PREVIEW_BASE_CELLS,
        "private_cells": private_cells,
        "total_cells": PREVIEW_BASE_CELLS + private_cells,
        "private_bundles": bundles,
        "base_parameters": base_parameters,
        "total_parameters": total_parameters,
        "growth_parameters": total_parameters - base_parameters,
        "parameter_growth_ratio": (total_parameters - base_parameters) / float(max(1, base_parameters)),
        "protected_probes": len(harness.probes),
        "new_gain": _attempt_value(record, "new_gain"),
        "local_regression": _attempt_value(record, "local_regression"),
        "global_regression": _attempt_value(record, "global_regression"),
        "dependency_coverage": _attempt_value(record, "local_dependency_coverage"),
        "touched_parameter_fraction": _attempt_value(record, "touched_parameter_fraction"),
        "structural_escape_rate": _attempt_value(record, "structural_escape_rate"),
        "false_safe_rate": false_safe / float(max(1, len(local_pass_attempts))),
        "training_wall_seconds_cumulative": training_seconds,
        "validation_wall_seconds_cumulative": validation_seconds,
        "transaction_wall_seconds_cumulative": transaction_seconds,
        "last_transaction_wall_seconds": float(record.get("transaction_wall_seconds", 0.0)),
        "math_teacher_forced_answer_exact": "",
        "story_teacher_forced_answer_exact": "",
    }
    if capability:
        row.update(capability)
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _cell_snapshot_rows(harness: VariantHarness, transaction_id: int) -> list[dict[str, Any]]:
    result = []
    for entry in harness.registry.snapshot(harness.model, harness.dependency_index):
        result.append({"snapshot_transaction_id": int(transaction_id), **entry})
    return result


def _dashboard(harness: VariantHarness, base_metrics: dict, timeline: list[dict[str, Any]], status: str) -> dict:
    summary = harness.summary()
    last = timeline[-1] if timeline else {}
    return {
        "format": "minicells.clm-0.4-preview.public-dashboard.v1",
        "status": status,
        "model": {
            "parameter_count": sum(p.numel() for p in harness.model.parameters()),
            "shared_cell_ffn_parameters": harness.model.shared_cell_ffn_parameters(),
            "base_cells": PREVIEW_BASE_CELLS,
            "private_cells": int(last.get("private_cells", 0) or 0),
            "total_cells": int(last.get("total_cells", PREVIEW_BASE_CELLS) or PREVIEW_BASE_CELLS),
        },
        "base_capability": base_metrics["capability"],
        "live_capability": {
            "math_teacher_forced_answer_exact": last.get("math_teacher_forced_answer_exact", ""),
            "story_teacher_forced_answer_exact": last.get("story_teacher_forced_answer_exact", ""),
        },
        "learning": {
            "transactions": summary["transactions"],
            "effective_commits": summary["effective_commits"],
            "acceptance_rate": summary["effective_acceptance_rate"],
            "growth_attempts": summary["growth_attempts"],
            "growth_commits": summary["growth_commits"],
            "growth_rescue_rate": summary["growth_rescue_rate"],
            "private_reuse_attempts": summary["private_reuse_attempts"],
            "private_reuse_acceptance_rate": summary["private_reuse_acceptance_rate"],
            "protected_probe_count": summary["protected_probe_count"],
        },
        "safety": {
            "false_safe_rate": summary["false_safe_rate"],
            "maximum_structural_escape_rate": summary["maximum_structural_escape_rate"],
            "positive_global_regression_damage": summary["positive_global_regression_damage"],
            "mean_direct_dependency_coverage": summary["mean_direct_dependency_coverage"],
            "final_protected_token_accuracy": summary["final_protected_token_accuracy"],
        },
        "growth": {
            "private_bundles": summary["spawned_bundles"],
            "growth_parameter_overhead_ratio": summary["growth_parameter_overhead_ratio"],
        },
        "cost": {
            "base_training_wall_seconds": base_metrics["base_train"]["wall_seconds"],
            "continual_training_wall_seconds": last.get("training_wall_seconds_cumulative", 0.0),
            "validation_wall_seconds": last.get("validation_wall_seconds_cumulative", 0.0),
            "transaction_wall_seconds": last.get("transaction_wall_seconds_cumulative", 0.0),
        },
    }


def render_preview_visualizations(out_dir: str | Path) -> list[str]:
    """Generate analysis-sized public PNGs from telemetry; no checkpoint required."""
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    timeline_path = out / "telemetry" / "timeline.csv"
    if not timeline_path.is_file():
        return []
    with timeline_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []

    def nums(key: str) -> list[float]:
        values = []
        for row in rows:
            raw = row.get(key, "")
            values.append(float(raw) if raw not in (None, "") else math.nan)
        return values

    x = nums("transactions")
    vis = out / "visualizations"
    vis.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    def save(name: str) -> None:
        path = vis / name
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        created.append(path.relative_to(out).as_posix())

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, nums("total_cells"), label="Total Cells")
    plt.plot(x, nums("private_cells"), label="Grown Cells")
    plt.xlabel("Transactions")
    plt.ylabel("Cells")
    plt.title("Cells growth over continual learning")
    plt.legend()
    save("cells-growth.png")

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, nums("commits"), label="Commits")
    plt.plot(x, nums("rollbacks"), label="Rollbacks")
    plt.plot(x, nums("growth_commits"), label="Growth commits")
    plt.plot(x, nums("private_reuse_commits"), label="Private reuse commits")
    plt.xlabel("Transactions")
    plt.ylabel("Cumulative events")
    plt.title("Learning decisions")
    plt.legend()
    save("learning-decisions.png")

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, nums("new_gain"), label="New gain")
    plt.plot(x, nums("local_regression"), label="Local regression")
    plt.plot(x, nums("global_regression"), label="Global regression")
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel("Transactions")
    plt.ylabel("Relative change")
    plt.title("Learning gain vs regression")
    plt.legend()
    save("learning-vs-regression.png")

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, nums("dependency_coverage"))
    plt.xlabel("Transactions")
    plt.ylabel("Protected history covered")
    plt.title("Dependency-scoped validation coverage")
    save("dependency-coverage.png")

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, nums("training_wall_seconds_cumulative"), label="Training")
    plt.plot(x, nums("validation_wall_seconds_cumulative"), label="Validation")
    plt.xlabel("Transactions")
    plt.ylabel("Cumulative seconds")
    plt.title("Continual-learning compute cost")
    plt.legend()
    save("compute-cost.png")

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, [100.0 * value for value in nums("parameter_growth_ratio")])
    plt.xlabel("Transactions")
    plt.ylabel("Growth parameters / base parameters (%)")
    plt.title("Parameter growth")
    save("parameter-growth.png")

    capability_rows = [
        row for row in rows
        if row.get("math_teacher_forced_answer_exact") not in (None, "")
    ]
    if capability_rows:
        cx = [float(row["transactions"]) for row in capability_rows]
        math_values = [100.0 * float(row["math_teacher_forced_answer_exact"]) for row in capability_rows]
        story_values = [100.0 * float(row["story_teacher_forced_answer_exact"]) for row in capability_rows]
        plt.figure(figsize=(8, 4.5))
        plt.plot(cx, math_values, marker="o", label="Math")
        plt.plot(cx, story_values, marker="o", label="Story")
        plt.xlabel("Transactions")
        plt.ylabel("Teacher-forced answer exact (%)")
        plt.ylim(0, 100)
        plt.title("Capability retention during continual learning")
        plt.legend()
        save("capability-over-time.png")

    registry_path = out / "telemetry" / "cell-registry-final.jsonl"
    if registry_path.is_file():
        entries = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines() if line]
        entries.sort(key=lambda item: int(item.get("activation_count", 0)), reverse=True)
        top = entries[:20]
        if top:
            plt.figure(figsize=(9, 5))
            plt.barh([item["cell_id"] for item in reversed(top)], [item["activation_count"] for item in reversed(top)])
            plt.xlabel("Activation count")
            plt.title("Most active Cells")
            save("cell-activity-top20.png")
    return created


def write_public_metrics(out_dir: str | Path, dashboard: dict, visualizations: list[str]) -> None:
    out = Path(out_dir)
    (out / "dashboard.json").write_text(
        json.dumps({**dashboard, "visualizations": visualizations}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    learning = dashboard["learning"]
    safety = dashboard["safety"]
    growth = dashboard["growth"]
    model = dashboard["model"]
    cost = dashboard["cost"]
    lines = [
        "# CLM-0.4 Preview — Public Metrics",
        "",
        f"> Status: **{dashboard['status']}**",
        "",
        "## Model",
        "",
        f"- Cells: **{model['total_cells']}** ({model['base_cells']} base + {model['private_cells']} grown)",
        f"- Parameters: **{model['parameter_count']:,}**",
        "",
        "## Learning",
        "",
        f"- Transactions: **{learning['transactions']}**",
        f"- Commits: **{learning['effective_commits']}** ({learning['acceptance_rate']:.1%})",
        f"- Growth rescue: **{learning['growth_rescue_rate']:.1%}** ({learning['growth_commits']}/{learning['growth_attempts']})",
        f"- Private reuse acceptance: **{learning['private_reuse_acceptance_rate']:.1%}**",
        f"- Protected probes: **{learning['protected_probe_count']}**",
        "",
        "## Safety / locality",
        "",
        f"- False-safe rate: **{safety['false_safe_rate']:.2%}**",
        f"- Maximum structural escape: **{safety['maximum_structural_escape_rate']:.2%}**",
        f"- Mean direct dependency coverage: **{safety['mean_direct_dependency_coverage']:.2%}**",
        f"- Final protected token accuracy: **{safety['final_protected_token_accuracy']:.2%}**",
        "",
        "## Growth / cost",
        "",
        f"- Private bundles: **{growth['private_bundles']}**",
        f"- Parameter overhead from growth: **{growth['growth_parameter_overhead_ratio']:.2%}**",
        f"- Base training: **{float(cost['base_training_wall_seconds']):.1f}s**",
        f"- Continual candidate training: **{float(cost['continual_training_wall_seconds']):.1f}s**",
        f"- Validation: **{float(cost['validation_wall_seconds']):.1f}s**",
        "",
        "## Visualizations",
        "",
        *[f"- `{path}`" for path in visualizations],
        "",
        "These are Preview telemetry metrics, not a formal scientific claim.",
    ]
    (out / "PUBLIC_METRICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_preview(
    *,
    protocol_path: str | Path,
    data_dir: str | Path,
    out_dir: str | Path,
    seed: int = PREVIEW_SEED,
    device: str | torch.device = "cuda",
    devices: str | None = None,
    max_transactions: int = 192,
    checkpoint_every: int = 8,
    capability_every: int = 16,
    resume: bool = True,
    direct_optimizer: CandidateOptimizerConfig = DEFAULT_DIRECT,
    growth_optimizer: CandidateOptimizerConfig = DEFAULT_GROWTH,
) -> dict[str, Any]:
    """Train/resume CLM-0.4 Preview and emit public longitudinal telemetry."""
    install_runtime_patches()
    protocol = load_protocol(protocol_path)
    thresholds = m1_thresholds(protocol)
    data = Path(data_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    asset_summary = json.loads((data / "asset-summary.json").read_text(encoding="utf-8"))
    if asset_summary.get("format") != PREVIEW_ASSET_FORMAT:
        raise RuntimeError("Preview requires CLM-0.4 Preview data assets")
    resolved_devices = resolve_cuda_devices(requested_device=device, requested_devices=devices)
    primary = resolved_devices[0]
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if primary.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    tokenizer = DigitAwareTokenizerBundle.load(data / "tokenizer" / "tokenizer.json")
    dataset = BaseCorpusDataset(data / "base-corpus")
    cfg = preview_model_config()
    base_dir = out / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    base_checkpoint = base_dir / "checkpoint.pt"
    model = TinyCLMDecoder(cfg).to(primary)
    base_source = "trained-once"
    if resume and base_checkpoint.is_file():
        saved = torch.load(base_checkpoint, map_location=primary, weights_only=False)
        if saved["model_config"] != cfg.to_dict():
            raise RuntimeError("Preview base checkpoint config mismatch")
        model.load_state_dict(saved["model_state"])
        base_train = saved["base_train"]
        base_source = "resumed"
    else:
        base_train = train_base_model_parallel(
            model,
            dataset=dataset,
            tokenizer=tokenizer,
            devices=resolved_devices,
            seed=int(seed),
            config=BaseTrainConfig(),
        )
        torch.save(
            {
                "format": PREVIEW_FORMAT,
                "seed": int(seed),
                "model_config": cfg.to_dict(),
                "model_state": model.state_dict(),
                "base_train": base_train,
                "asset_summary": asset_summary,
            },
            base_checkpoint,
        )
    activation = base_cell_activation_counts(model, dataset)
    capability = _capability_metrics(model, tokenizer, primary)
    base_metrics = {
        "format": "minicells.clm-0.4-preview.base-metrics.v1",
        "checkpoint_source": base_source,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "shared_cell_ffn_parameters": model.shared_cell_ffn_parameters(),
        "base_train": base_train,
        "capability": capability,
        "cell_activation": {
            "minimum": min(activation.values()) if activation else 0,
            "maximum": max(activation.values()) if activation else 0,
            "mean": sum(activation.values()) / float(max(1, len(activation))),
        },
        "asset_summary": asset_summary,
    }
    (base_dir / "base-metrics.json").write_text(
        json.dumps(base_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    checkpoints = out / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    latest = checkpoints / "latest.pt"
    if resume and latest.is_file():
        harness = VariantHarness.load_checkpoint(
            latest, tokenizer=tokenizer, device=primary, thresholds=thresholds
        )
        if harness.model.cfg.to_dict() != cfg.to_dict():
            raise RuntimeError("Preview continual checkpoint config mismatch")
    else:
        harness = VariantHarness(
            variant="local_tx_growth",
            model=model,
            tokenizer=tokenizer,
            device=primary,
            thresholds=thresholds,
        )
        harness.admit_probes(_base_probes(model, tokenizer))

    curriculum = json.loads((data / "curriculum-manifest.json").read_text(encoding="utf-8"))
    specs = transaction_specs(curriculum)[: max(0, int(max_transactions))]
    done = {int(record["transaction_id"]) for record in harness.records}
    timeline: list[dict[str, Any]] = []
    timeline_path = out / "telemetry" / "timeline.csv"
    if timeline_path.is_file():
        with timeline_path.open(encoding="utf-8") as handle:
            timeline = list(csv.DictReader(handle))
    cell_snapshots: list[dict[str, Any]] = []
    snapshots_path = out / "telemetry" / "cell-snapshots.jsonl"
    if snapshots_path.is_file():
        cell_snapshots = [json.loads(line) for line in snapshots_path.read_text(encoding="utf-8").splitlines() if line]

    for index, spec in enumerate(specs):
        if int(spec.transaction_id) in done:
            continue
        tx = _tokenized_transaction(spec, tokenizer, cfg.max_seq_len)
        transaction_seed = (int(seed) * 1_000_003 + int(spec.transaction_id) * 97) & 0x7FFFFFFF
        record = harness.execute(
            transaction_id=spec.transaction_id,
            operation=spec.operation,
            address_id=spec.address_id,
            knowledge_key=spec.knowledge_key,
            supersedes_key=spec.supersedes_key,
            train_examples=tx["train"],
            validation_examples=tx["validation"],
            probe_examples=tx["probe"],
            direct_optimizer=direct_optimizer,
            growth_optimizer=growth_optimizer,
            rng_seed=transaction_seed,
        )
        cap = None
        processed = len(harness.records)
        if processed % int(capability_every) == 0 or processed == len(specs):
            cap = _capability_metrics(harness.model, tokenizer, primary)
        row = _timeline_row(harness, record, cap)
        timeline.append(row)
        _write_csv(timeline_path, timeline)
        _write_jsonl(out / "telemetry" / "transactions.jsonl", harness.records)

        if processed % int(checkpoint_every) == 0 or processed == len(specs):
            cell_snapshots.extend(_cell_snapshot_rows(harness, int(spec.transaction_id)))
            _write_jsonl(snapshots_path, cell_snapshots)
            _write_jsonl(
                out / "telemetry" / "cell-registry-final.jsonl",
                harness.registry.snapshot(harness.model, harness.dependency_index),
            )
            harness.save_checkpoint(latest)
            harness.save_checkpoint(checkpoints / f"tx-{int(spec.transaction_id):03d}.pt")

    status = "PREVIEW_COMPLETE" if len(harness.records) >= len(specs) else "PREVIEW_PARTIAL"
    summary = harness.summary()
    (out / "continual-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    dashboard = _dashboard(harness, base_metrics, timeline, status)
    visualizations = render_preview_visualizations(out)
    write_public_metrics(out, dashboard, visualizations)
    decision = {
        "format": PREVIEW_FORMAT,
        "status": status,
        "release_track": "preview",
        "scientific_decision": False,
        "capability_metrics_block_execution": False,
        "seed": int(seed),
        "transactions": len(harness.records),
        "state_hash": model_state_hash(harness.model),
    }
    (out / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "summary.json").write_text(
        json.dumps({"decision": decision, "dashboard": dashboard, "base": base_metrics}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"decision": decision, "dashboard": dashboard, "visualizations": visualizations}
