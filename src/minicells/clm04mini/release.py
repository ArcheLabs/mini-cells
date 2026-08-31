"""CLM-0.4 release pipeline with a mandatory 1M end-to-end smoke before 30M.

The release promotes the validated Preview configuration without changing its
base model/data recipe. Both profiles execute the same CLM, dense comparison,
continual-learning, telemetry, visualization, and reporting code paths; only
the base token budget differs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping

import torch

from .curriculum import transaction_specs
from .examples import tokenize_examples
from .model import MiniCLMConfig
from .performance import install_runtime_patches, resolve_cuda_devices, train_base_model_parallel
from .preview import (
    DEFAULT_DIRECT,
    DEFAULT_GROWTH,
    PREVIEW_MIXTURE,
    _tokenized_transaction,
    preview_model_config,
    run_preview,
)
from .protocol import canonical_json_hash, load_protocol, m1_thresholds
from .training import BaseCorpusDataset, BaseTrainConfig
from .v2 import (
    DenseContinualHarness,
    DenseDecoder,
    teacher_forced_answer_exact_accuracy,
    v2_math_eval_examples,
    v2_story_eval_examples,
)
from .tokenizer import DigitAwareTokenizerBundle


RELEASE_FORMAT = "minicells.clm-0.4-release.v1"
READINESS_FORMAT = "minicells.clm-0.4-release-readiness.v1"
COMPARISON_FORMAT = "minicells.clm-0.4-release-comparison.v1"
RELEASE_VERSION = "clm-0.4-release-v1"
RELEASE_SEED = 90600
RELEASE_TRANSACTIONS = 192
RELEASE_DENSE_FFN_HIDDEN = 1032
RELEASE_DENSE_EXPECTED_PARAMETERS = 5_273_120
RELEASE_CLM_EXPECTED_PARAMETERS = 5_273_088
RELEASE_PROFILES = {
    "smoke-1m": 1_000_000,
    "release-30m": 30_000_000,
}

SOURCE_FINGERPRINT_PATHS = (
    "src/minicells/clm04mini/model.py",
    "src/minicells/clm04mini/preview.py",
    "src/minicells/clm04mini/release.py",
    "src/minicells/clm04mini/v2.py",
    "src/minicells/clm04mini/performance.py",
    "src/minicells/clm04mini/engine.py",
    "src/minicells/clm04mini/tokenizer.py",
    "src/minicells/clm04mini/training.py",
    "src/minicells/clm04mini/curriculum.py",
    "scripts/research/prepare_clm_0_4_release_data.py",
    "scripts/research/run_clm_0_4_release.py",
    "scripts/research/report_clm_0_4_release.py",
    "scripts/research/publish_clm_0_4_release_results.py",
)


def release_model_config() -> MiniCLMConfig:
    """The first release freezes the successful Preview model configuration."""
    return preview_model_config()


def release_dense_config() -> MiniCLMConfig:
    clm = release_model_config()
    return MiniCLMConfig(
        vocab_size=clm.vocab_size,
        max_seq_len=clm.max_seq_len,
        num_layers=clm.num_layers,
        d_model=clm.d_model,
        n_heads=clm.n_heads,
        dense_ff_hidden=RELEASE_DENSE_FFN_HIDDEN,
        base_cells=2,
        cell_hidden=1,
        routing_salt=f"{clm.routing_salt}/release-dense-equal-parameter",
        shared_cell_ff_hidden=0,
    )


def _optimizer_payload(config) -> dict[str, Any]:
    return {
        "optimizer": str(config.optimizer),
        "batch_size": int(config.batch_size),
        "learning_rate": float(config.learning_rate),
        "steps": int(config.steps),
        "weight_decay": float(config.weight_decay),
    }


def release_pipeline_identity() -> dict[str, Any]:
    return {
        "release_version": RELEASE_VERSION,
        "clm_model": release_model_config().to_dict(),
        "clm_expected_parameters": RELEASE_CLM_EXPECTED_PARAMETERS,
        "dense_model": release_dense_config().to_dict(),
        "dense_expected_parameters": RELEASE_DENSE_EXPECTED_PARAMETERS,
        "base_mixture": dict(PREVIEW_MIXTURE),
        "base_train": BaseTrainConfig().__dict__,
        "direct_optimizer": _optimizer_payload(DEFAULT_DIRECT),
        "growth_optimizer": _optimizer_payload(DEFAULT_GROWTH),
        "transactions": RELEASE_TRANSACTIONS,
        "checkpoint_every": 8,
        "capability_every": 16,
        "dense_continual_variant": "dense_full_always",
        "tokenizer_policy": "digit-aware-preview-v1",
    }


def release_pipeline_sha256() -> str:
    return canonical_json_hash(release_pipeline_identity())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_source_fingerprint(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    files: dict[str, str] = {}
    for relative in SOURCE_FINGERPRINT_PATHS:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"release source fingerprint missing {relative}")
        files[relative] = _sha256_file(path)
    return {
        "files": files,
        "sha256": canonical_json_hash(files),
    }


def expected_profile_tokens(profile: str) -> int:
    if profile not in RELEASE_PROFILES:
        raise ValueError(f"unknown release profile: {profile}")
    return int(RELEASE_PROFILES[profile])


def validate_release_assets(profile: str, data_dir: str | Path) -> dict[str, Any]:
    data = Path(data_dir)
    summary = json.loads((data / "asset-summary.json").read_text(encoding="utf-8"))
    target = expected_profile_tokens(profile)
    if int(summary.get("target_tokens", -1)) != target:
        raise RuntimeError(
            f"{profile} requires target_tokens={target}, got {summary.get('target_tokens')}"
        )
    actual = int(summary.get("base_tokens", 0))
    if abs(actual - target) > max(1, int(target * 0.01)):
        raise RuntimeError(f"{profile} base token count outside 1% tolerance")
    if dict(summary.get("mixture", {})) != dict(PREVIEW_MIXTURE):
        raise RuntimeError("release base mixture drifted from frozen Preview recipe")
    return summary


def _release_eval_examples() -> tuple[list, list]:
    cfg = release_model_config()
    return (
        v2_math_eval_examples(cfg, 64, seed=6301),
        v2_story_eval_examples(cfg, 64, seed=6401),
    )


def capability_metrics(model, tokenizer, device: torch.device) -> dict[str, float]:
    math_examples, story_examples = _release_eval_examples()
    cfg = release_model_config()
    return {
        "math_teacher_forced_answer_exact": teacher_forced_answer_exact_accuracy(
            model,
            math_examples,
            tokenizer=tokenizer,
            max_seq_len=cfg.max_seq_len,
            device=device,
        ),
        "story_teacher_forced_answer_exact": teacher_forced_answer_exact_accuracy(
            model,
            story_examples,
            tokenizer=tokenizer,
            max_seq_len=cfg.max_seq_len,
            device=device,
        ),
    }


def release_base_probes(tokenizer) -> list:
    math_examples, story_examples = _release_eval_examples()
    return tokenize_examples(
        [*math_examples, *story_examples],
        tokenizer,
        max_seq_len=release_model_config().max_seq_len,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_or_train_dense_base(
    *,
    data_dir: Path,
    out_dir: Path,
    tokenizer,
    devices: list[torch.device],
    seed: int,
    asset_summary: Mapping[str, Any],
) -> tuple[DenseDecoder, dict[str, Any]]:
    cfg = release_dense_config()
    model = DenseDecoder(cfg).to(devices[0])
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != RELEASE_DENSE_EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"release dense parameter count drift: {count} != {RELEASE_DENSE_EXPECTED_PARAMETERS}"
        )
    checkpoint = out_dir / "base" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.is_file():
        payload = torch.load(checkpoint, map_location=devices[0], weights_only=False)
        if payload.get("model_config") != cfg.to_dict():
            raise RuntimeError("release dense base checkpoint config mismatch")
        if payload.get("asset_summary") != dict(asset_summary):
            raise RuntimeError("release dense base checkpoint asset mismatch")
        model.load_state_dict(payload["model_state"])
        train_stats = dict(payload["base_train"])
        source = "resumed"
    else:
        random.seed(int(seed))
        torch.manual_seed(int(seed))
        if devices[0].type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        train_stats = train_base_model_parallel(
            model,
            dataset=BaseCorpusDataset(data_dir / "base-corpus"),
            tokenizer=tokenizer,
            devices=devices,
            seed=int(seed),
            config=BaseTrainConfig(),
        )
        torch.save(
            {
                "format": RELEASE_FORMAT,
                "kind": "dense-equal-parameter-base",
                "model_config": cfg.to_dict(),
                "model_state": model.state_dict(),
                "base_train": train_stats,
                "asset_summary": dict(asset_summary),
            },
            checkpoint,
        )
        source = "trained-once"
    metrics = {
        "format": RELEASE_FORMAT,
        "kind": "dense-equal-parameter",
        "checkpoint_source": source,
        "parameter_count": count,
        "base_train": train_stats,
        "capability": capability_metrics(model, tokenizer, devices[0]),
    }
    _write_json(out_dir / "base" / "metrics.json", metrics)
    return model, metrics


def _dense_checkpoint_payload(harness: DenseContinualHarness) -> dict[str, Any]:
    return {
        "format": RELEASE_FORMAT,
        "variant": harness.variant,
        "model_config": harness.model.cfg.to_dict(),
        "model_state": harness.model.state_dict(),
        "records": harness.records,
        "probes": harness.probes,
        "reference_accuracy": harness.reference_accuracy,
    }


def run_dense_release_baseline(
    *,
    protocol_path: str | Path,
    data_dir: str | Path,
    out_dir: str | Path,
    asset_summary: Mapping[str, Any],
    seed: int,
    device: str | torch.device,
    devices: str | None,
) -> dict[str, Any]:
    install_runtime_patches()
    resolved = resolve_cuda_devices(requested_device=device, requested_devices=devices)
    primary = resolved[0]
    data = Path(data_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = DigitAwareTokenizerBundle.load(data / "tokenizer" / "tokenizer.json")
    base_model, base_metrics = _load_or_train_dense_base(
        data_dir=data,
        out_dir=out,
        tokenizer=tokenizer,
        devices=resolved,
        seed=int(seed),
        asset_summary=asset_summary,
    )
    protocol = load_protocol(protocol_path)
    thresholds = m1_thresholds(protocol)
    curriculum = json.loads((data / "curriculum-manifest.json").read_text(encoding="utf-8"))
    specs = transaction_specs(curriculum)
    latest = out / "checkpoints" / "latest.pt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    if latest.is_file():
        payload = torch.load(latest, map_location=primary, weights_only=False)
        model = DenseDecoder(release_dense_config()).to(primary)
        model.load_state_dict(payload["model_state"])
        harness = DenseContinualHarness(
            variant="dense_full_always",
            model=model,
            tokenizer=tokenizer,
            device=primary,
            thresholds=thresholds,
        )
        harness.records = list(payload["records"])
        harness.probes = dict(payload["probes"])
        harness.reference_accuracy = dict(payload["reference_accuracy"])
    else:
        harness = DenseContinualHarness(
            variant="dense_full_always",
            model=base_model,
            tokenizer=tokenizer,
            device=primary,
            thresholds=thresholds,
        )
        harness.admit(release_base_probes(tokenizer))

    done = {int(row["transaction_id"]) for row in harness.records}
    timeline_path = out / "telemetry" / "timeline.csv"
    timeline: list[dict[str, Any]] = []
    if timeline_path.is_file():
        with timeline_path.open(encoding="utf-8") as handle:
            timeline = list(csv.DictReader(handle))
    for spec in specs:
        if int(spec.transaction_id) in done:
            continue
        tx = _tokenized_transaction(spec, tokenizer, release_model_config().max_seq_len)
        tx_seed = (int(seed) * 1_000_003 + int(spec.transaction_id) * 97 + 997) & 0x7FFFFFFF
        harness.execute(
            transaction_id=spec.transaction_id,
            operation=spec.operation,
            supersedes_key=spec.supersedes_key,
            train_examples=tx["train"],
            validation_examples=tx["validation"],
            probe_examples=tx["probe"],
            optimizer_config=DEFAULT_DIRECT,
            rng_seed=tx_seed,
        )
        processed = len(harness.records)
        cap = capability_metrics(harness.model, tokenizer, primary) if processed % 16 == 0 or processed == RELEASE_TRANSACTIONS else None
        row = {
            "transactions": processed,
            "commits": sum(int(item["commit"]) for item in harness.records),
            "global_regression": float(harness.records[-1]["global_regression"]),
            "new_gain": float(harness.records[-1]["new_gain"]),
            "training_wall_seconds_cumulative": sum(float(item["candidate_wall_seconds"]) for item in harness.records),
            "math_teacher_forced_answer_exact": "" if cap is None else cap["math_teacher_forced_answer_exact"],
            "story_teacher_forced_answer_exact": "" if cap is None else cap["story_teacher_forced_answer_exact"],
        }
        timeline.append(row)
        _write_csv(timeline_path, timeline)
        _write_jsonl(out / "telemetry" / "transactions.jsonl", harness.records)
        if processed % 8 == 0 or processed == RELEASE_TRANSACTIONS:
            torch.save(_dense_checkpoint_payload(harness), latest)
            torch.save(_dense_checkpoint_payload(harness), out / "checkpoints" / f"tx-{processed:03d}.pt")

    summary = harness.summary()
    summary.update(
        {
            "format": RELEASE_FORMAT,
            "base_capability": base_metrics["capability"],
            "final_capability": capability_metrics(harness.model, tokenizer, primary),
            "base_training_wall_seconds": float(base_metrics["base_train"]["wall_seconds"]),
            "continual_training_wall_seconds": sum(float(item["candidate_wall_seconds"]) for item in harness.records),
        }
    )
    _write_json(out / "summary.json", summary)
    return summary


def _read_clm_capability_timeline(clm_out: Path) -> list[dict[str, Any]]:
    path = clm_out / "telemetry" / "timeline.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def render_release_comparison(out_dir: str | Path, comparison: Mapping[str, Any]) -> list[str]:
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    vis = out / "visualizations"
    vis.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    def save(name: str) -> None:
        path = vis / name
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        created.append(path.relative_to(out).as_posix())

    clm_base = comparison["clm"]["base_capability"]
    dense_base = comparison["dense"]["base_capability"]
    labels = ["Math", "Story"]
    x = [0, 1]
    width = 0.36
    plt.figure(figsize=(7, 4.5))
    plt.bar([v - width / 2 for v in x], [100 * clm_base["math_teacher_forced_answer_exact"], 100 * clm_base["story_teacher_forced_answer_exact"]], width=width, label="CLM")
    plt.bar([v + width / 2 for v in x], [100 * dense_base["math_teacher_forced_answer_exact"], 100 * dense_base["story_teacher_forced_answer_exact"]], width=width, label="Dense")
    plt.xticks(x, labels)
    plt.ylim(0, 100)
    plt.ylabel("Teacher-forced answer exact (%)")
    plt.title("Base capability: CLM vs Dense")
    plt.legend()
    save("base-capability-clm-vs-dense.png")

    for domain, title in (("math", "Math capability retention"), ("story", "Story capability retention")):
        plt.figure(figsize=(8, 4.5))
        for label, rows in (("CLM", comparison["clm"]["capability_timeline"]), ("Dense", comparison["dense"]["capability_timeline"])):
            filtered = [row for row in rows if row.get(f"{domain}_teacher_forced_answer_exact") not in (None, "")]
            if filtered:
                plt.plot(
                    [float(row["transactions"]) for row in filtered],
                    [100 * float(row[f"{domain}_teacher_forced_answer_exact"]) for row in filtered],
                    marker="o",
                    label=label,
                )
        plt.ylim(0, 100)
        plt.xlabel("Continual transactions")
        plt.ylabel("Teacher-forced answer exact (%)")
        plt.title(title)
        plt.legend()
        save(f"{domain}-retention-clm-vs-dense.png")

    plt.figure(figsize=(7, 4.5))
    retention = [
        100 * float(comparison["clm"]["protected_retention_ratio"]),
        100 * float(comparison["dense"]["protected_retention_ratio"]),
    ]
    plt.bar(["CLM", "Dense"], retention)
    plt.ylabel("Protected retention ratio (%)")
    plt.title("Protected-history retention after 192 updates")
    save("protected-retention-clm-vs-dense.png")
    return created


def _load_dense_timeline(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def build_comparison(out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    clm_dashboard = json.loads((out / "clm" / "dashboard.json").read_text(encoding="utf-8"))
    clm_summary = json.loads((out / "clm" / "continual-summary.json").read_text(encoding="utf-8"))
    dense_summary = json.loads((out / "dense" / "summary.json").read_text(encoding="utf-8"))
    dense_timeline = _load_dense_timeline(out / "dense" / "telemetry" / "timeline.csv")
    payload = {
        "format": COMPARISON_FORMAT,
        "authority": "release-benchmark",
        "equal_parameter_difference": RELEASE_DENSE_EXPECTED_PARAMETERS - RELEASE_CLM_EXPECTED_PARAMETERS,
        "clm": {
            "parameter_count_base": RELEASE_CLM_EXPECTED_PARAMETERS,
            "parameter_count_final": int(clm_dashboard["model"]["parameter_count"]),
            "base_capability": clm_dashboard["base_capability"],
            "final_capability": clm_dashboard["live_capability"],
            "protected_retention_ratio": float(clm_summary["final_protected_retention_ratio"]),
            "effective_commits": int(clm_summary["effective_commits"]),
            "growth_parameter_overhead_ratio": float(clm_summary["growth_parameter_overhead_ratio"]),
            "capability_timeline": _read_clm_capability_timeline(out / "clm"),
        },
        "dense": {
            "parameter_count_base": int(dense_summary["parameter_count"]),
            "parameter_count_final": int(dense_summary["parameter_count"]),
            "base_capability": dense_summary["base_capability"],
            "final_capability": dense_summary["final_capability"],
            "protected_retention_ratio": float(dense_summary["final_protected_retention_ratio"]),
            "effective_commits": int(dense_summary["effective_commits"]),
            "growth_parameter_overhead_ratio": 0.0,
            "capability_timeline": dense_timeline,
        },
    }
    _write_json(out / "comparison.json", payload)
    payload["visualizations"] = render_release_comparison(out, payload)
    _write_json(out / "comparison.json", payload)
    return payload


def verify_smoke_readiness(
    readiness_path: str | Path,
    *,
    source_fingerprint: Mapping[str, Any],
    pipeline_sha256: str,
    tokenizer_hash: str | None = None,
) -> dict[str, Any]:
    path = Path(readiness_path)
    if not path.is_file():
        raise RuntimeError("30M release requires a completed 1M release-readiness.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != READINESS_FORMAT or payload.get("status") != "READY_FOR_30M":
        raise RuntimeError("1M readiness artifact is not READY_FOR_30M")
    if payload.get("pipeline_sha256") != pipeline_sha256:
        raise RuntimeError("release pipeline changed after 1M smoke; rerun 1M")
    if payload.get("source_fingerprint", {}).get("sha256") != source_fingerprint.get("sha256"):
        raise RuntimeError("release source files changed after 1M smoke; rerun 1M")
    if tokenizer_hash is not None and payload.get("tokenizer_hash") != tokenizer_hash:
        raise RuntimeError("30M tokenizer identity differs from 1M smoke")
    if int(payload.get("transactions", 0)) != RELEASE_TRANSACTIONS:
        raise RuntimeError("1M smoke did not complete the full 192-transaction path")
    return payload


def build_smoke_readiness(
    *,
    out_dir: str | Path,
    asset_summary: Mapping[str, Any],
    source_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    out = Path(out_dir)
    root_decision = json.loads((out / "decision.json").read_text(encoding="utf-8"))
    clm_decision = json.loads((out / "clm" / "decision.json").read_text(encoding="utf-8"))
    dense_summary = json.loads((out / "dense" / "summary.json").read_text(encoding="utf-8"))
    required = [
        out / "comparison.json",
        out / "clm" / "dashboard.json",
        out / "clm" / "telemetry" / "timeline.csv",
        out / "dense" / "telemetry" / "timeline.csv",
        out / "visualizations" / "base-capability-clm-vs-dense.png",
        out / "visualizations" / "math-retention-clm-vs-dense.png",
        out / "visualizations" / "protected-retention-clm-vs-dense.png",
    ]
    checks = {
        "profile_is_smoke_1m": root_decision.get("profile") == "smoke-1m",
        "clm_complete": clm_decision.get("status") == "PREVIEW_COMPLETE" and int(clm_decision.get("transactions", 0)) == RELEASE_TRANSACTIONS,
        "dense_complete": int(dense_summary.get("transactions", 0)) == RELEASE_TRANSACTIONS,
        "analysis_complete": all(path.is_file() and path.stat().st_size > 0 for path in required),
        "asset_budget_valid": abs(int(asset_summary["base_tokens"]) - 1_000_000) <= 10_000,
        "clm_parameter_identity": int(json.loads((out / "clm" / "base" / "base-metrics.json").read_text(encoding="utf-8"))["parameter_count"]) == RELEASE_CLM_EXPECTED_PARAMETERS,
        "dense_parameter_identity": int(dense_summary.get("parameter_count", 0)) == RELEASE_DENSE_EXPECTED_PARAMETERS,
    }
    ready = all(checks.values())
    payload = {
        "format": READINESS_FORMAT,
        "status": "READY_FOR_30M" if ready else "SMOKE_INCOMPLETE",
        "release_version": RELEASE_VERSION,
        "pipeline_sha256": release_pipeline_sha256(),
        "source_fingerprint": dict(source_fingerprint),
        "tokenizer_hash": str(asset_summary["tokenizer_hash"]),
        "curriculum_manifest_hash": str(asset_summary["curriculum_manifest_hash"]),
        "transactions": RELEASE_TRANSACTIONS if ready else int(clm_decision.get("transactions", 0)),
        "checks": checks,
    }
    _write_json(out / "release-readiness.json", payload)
    if not ready:
        raise RuntimeError("1M release smoke completed but readiness checks failed")
    return payload


def run_release(
    *,
    profile: str,
    protocol_path: str | Path,
    data_dir: str | Path,
    out_dir: str | Path,
    repo_root: str | Path,
    device: str | torch.device = "cuda",
    devices: str | None = None,
    smoke_readiness_path: str | Path | None = None,
) -> dict[str, Any]:
    install_runtime_patches()
    target = expected_profile_tokens(profile)
    data = Path(data_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    assets = validate_release_assets(profile, data)
    source_fingerprint = release_source_fingerprint(repo_root)
    pipeline_sha = release_pipeline_sha256()
    if profile == "release-30m":
        if smoke_readiness_path is None:
            raise RuntimeError("30M release is blocked until --smoke-readiness is supplied")
        verify_smoke_readiness(
            smoke_readiness_path,
            source_fingerprint=source_fingerprint,
            pipeline_sha256=pipeline_sha,
            tokenizer_hash=str(assets["tokenizer_hash"]),
        )

    started = time.perf_counter()
    clm_result = run_preview(
        protocol_path=protocol_path,
        data_dir=data,
        out_dir=out / "clm",
        seed=RELEASE_SEED,
        device=device,
        devices=devices,
        max_transactions=RELEASE_TRANSACTIONS,
        checkpoint_every=8,
        capability_every=16,
        resume=True,
        direct_optimizer=DEFAULT_DIRECT,
        growth_optimizer=DEFAULT_GROWTH,
    )
    dense_summary = run_dense_release_baseline(
        protocol_path=protocol_path,
        data_dir=data,
        out_dir=out / "dense",
        asset_summary=assets,
        seed=RELEASE_SEED,
        device=device,
        devices=devices,
    )
    comparison = build_comparison(out)
    decision = {
        "format": RELEASE_FORMAT,
        "release_track": "release",
        "release_version": RELEASE_VERSION,
        "profile": profile,
        "status": "RELEASE_SMOKE_COMPLETE" if profile == "smoke-1m" else "RELEASE_30M_COMPLETE",
        "target_tokens": target,
        "base_tokens": int(assets["base_tokens"]),
        "seed": RELEASE_SEED,
        "transactions": RELEASE_TRANSACTIONS,
        "pipeline_sha256": pipeline_sha,
        "source_fingerprint_sha256": source_fingerprint["sha256"],
        "clm_status": clm_result["decision"]["status"],
        "dense_transactions": int(dense_summary["transactions"]),
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json(out / "decision.json", decision)
    summary = {
        "decision": decision,
        "asset_summary": assets,
        "pipeline_identity": release_pipeline_identity(),
        "source_fingerprint": source_fingerprint,
        "comparison": comparison,
    }
    _write_json(out / "summary.json", summary)
    if profile == "smoke-1m":
        summary["readiness"] = build_smoke_readiness(
            out_dir=out,
            asset_summary=assets,
            source_fingerprint=source_fingerprint,
        )
        _write_json(out / "summary.json", summary)
    return summary
