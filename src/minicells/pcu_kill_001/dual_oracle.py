"""Two-GPU scheduling for the engineering context-oracle v2.

The positive-control definition is unchanged. Independent AB samples are split
between two byte-identical Granite replicas and merged back in canonical sample
order. Formal execution never installs this wrapper.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Mapping, Sequence

import torch

from .cellular import GraniteArchitectureInspector
from .model import load_granite
from .synthetic import (
    POSITIVE_CONTROL_CANDIDATES,
    POSITIVE_CONTROL_FLOOR,
    POSITIVE_CONTROL_VERSION,
    SyntheticWorld,
    _candidate_pool,
    _free_generation_diagnostic,
    _rank_candidate,
)


class DualGPUContextOracle:
    def __init__(
        self,
        original_oracle: Any,
        *,
        primary_model: Any,
        primary_tokenizer: Any,
        model_repo: str,
        model_revision: str,
        foundation_hash: str,
        inspector: GraniteArchitectureInspector,
        primary_device: str = "cuda:0",
        secondary_device: str = "cuda:1",
    ) -> None:
        self.original_oracle = original_oracle
        self.primary_model = primary_model
        self.primary_tokenizer = primary_tokenizer
        self.model_repo = str(model_repo)
        self.model_revision = str(model_revision)
        self.foundation_hash = str(foundation_hash)
        self.inspector = inspector
        self.primary_device = str(primary_device)
        self.secondary_device = str(secondary_device)

    def _rows_for_samples(
        self,
        world: SyntheticWorld,
        samples: Sequence[Any],
        *,
        model: Any,
        tokenizer: Any,
        device: str,
    ) -> list[dict[str, Any]]:
        vs = [item.v for item in world.triples]
        ws = [item.w for item in world.triples]
        pairs = [f"{item.v} {item.w}" for item in world.triples]
        rows: list[dict[str, Any]] = []
        for sample in samples:
            triple = world.triples[int(sample.pair_id)]
            prompt_a = (
                f"Mapping record:\n{triple.u} -> {triple.v}\n"
                f"Query:\n{triple.u} ->\nAnswer: "
            )
            prompt_b = (
                f"Mapping record:\n{triple.v} -> {triple.w}\n"
                f"Query:\n{triple.v} ->\nAnswer: "
            )
            prompt_ab = (
                f"Mapping records:\n{triple.u} -> {triple.v}\n{triple.v} -> {triple.w}\n"
                f"Query path:\n{triple.u} ->\nRelay and terminal: "
            )
            a_candidates = _candidate_pool(vs, triple.v, f"{sample.sample_id}:A")
            b_candidates = _candidate_pool(ws, triple.w, f"{sample.sample_id}:B")
            pair_correct = f"{triple.v} {triple.w}"
            pair_candidates = _candidate_pool(pairs, pair_correct, f"{sample.sample_id}:AB")
            rank_a = _rank_candidate(
                model, tokenizer, prompt_a, a_candidates, triple.v, device=device
            )
            rank_b = _rank_candidate(
                model, tokenizer, prompt_b, b_candidates, triple.w, device=device
            )
            rank_ab = _rank_candidate(
                model, tokenizer, prompt_ab, pair_candidates, pair_correct, device=device
            )
            predicted_pair = rank_ab["winner"].split()
            predicted_relay = predicted_pair[0] if predicted_pair else None
            predicted_terminal = predicted_pair[-1] if len(predicted_pair) >= 2 else None
            rows.append({
                "sample_id": sample.sample_id,
                "expected_relay": triple.v,
                "expected_terminal": triple.w,
                "retrieval_a": rank_a,
                "retrieval_b": rank_b,
                "composition": rank_ab,
                "retrieval_a_exact": bool(rank_a["exact"]),
                "retrieval_b_exact": bool(rank_b["exact"]),
                "relay_exact": predicted_relay == triple.v,
                "terminal_exact": predicted_terminal == triple.w,
                "both_exact": bool(rank_ab["exact"]),
            })
        return rows

    def __call__(
        self,
        world: Any,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        device: str = "cpu",
        max_new_tokens: int = 16,
    ) -> dict[str, Any]:
        if (
            model is not self.primary_model
            or tokenizer is not self.primary_tokenizer
            or not torch.cuda.is_available()
            or torch.cuda.device_count() < 2
        ):
            return self.original_oracle(
                world,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
            )
        if not isinstance(world, SyntheticWorld):
            # The real engineering worker passes SyntheticWorld. Keep the
            # mapping compatibility path delegated to the canonical oracle.
            return self.original_oracle(
                world,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
            )

        secondary_tokenizer, secondary_model, manifest = load_granite(
            self.model_repo,
            revision=self.model_revision,
            device=self.secondary_device,
        )
        try:
            secondary_inspector = GraniteArchitectureInspector.inspect(
                secondary_model, require_granite=True
            )
            if asdict(secondary_inspector) != asdict(self.inspector):
                raise RuntimeError("context-oracle secondary architecture mismatch")
            if str(manifest.get("foundation_tensor_sha256")) != self.foundation_hash:
                raise RuntimeError("context-oracle secondary foundation hash mismatch")

            samples = list(world.splits.get("AB_eval", []))
            left = samples[0::2]
            right = samples[1::2]
            with ThreadPoolExecutor(max_workers=2) as pool:
                future_left = pool.submit(
                    self._rows_for_samples,
                    world,
                    left,
                    model=self.primary_model,
                    tokenizer=self.primary_tokenizer,
                    device=self.primary_device,
                )
                future_right = pool.submit(
                    self._rows_for_samples,
                    world,
                    right,
                    model=secondary_model,
                    tokenizer=secondary_tokenizer,
                    device=self.secondary_device,
                )
                unordered = future_left.result() + future_right.result()

            by_id = {str(row["sample_id"]): row for row in unordered}
            rows = [by_id[str(sample.sample_id)] for sample in samples]
            n = max(1, len(rows))
            retrieval_a_accuracy = sum(bool(row["retrieval_a_exact"]) for row in rows) / n
            retrieval_b_accuracy = sum(bool(row["retrieval_b_exact"]) for row in rows) / n
            composition_accuracy = sum(bool(row["both_exact"]) for row in rows) / n
            relay_accuracy = sum(bool(row["relay_exact"]) for row in rows) / n
            terminal_accuracy = sum(bool(row["terminal_exact"]) for row in rows) / n
            passed = min(
                retrieval_a_accuracy,
                retrieval_b_accuracy,
                composition_accuracy,
            ) >= POSITIVE_CONTROL_FLOOR
            return {
                "schema": "minicells.pcu-kill-001.context-oracle.v2",
                "positive_control_version": POSITIVE_CONTROL_VERSION,
                "mode": "model_backed_teacher_forced_candidate_ranking",
                "execution_schedule": "dual_gpu_sample_partition",
                "candidate_pool_size": min(POSITIVE_CONTROL_CANDIDATES, len(world.triples)),
                "threshold": POSITIVE_CONTROL_FLOOR,
                "retrieval_a_accuracy": retrieval_a_accuracy,
                "retrieval_b_accuracy": retrieval_b_accuracy,
                "composition_accuracy": composition_accuracy,
                "accuracy": composition_accuracy,
                "relay_accuracy": relay_accuracy,
                "terminal_accuracy": terminal_accuracy,
                "rows": rows,
                "free_generation_diagnostic": _free_generation_diagnostic(
                    world,
                    model=self.primary_model,
                    tokenizer=self.primary_tokenizer,
                    device=self.primary_device,
                    max_new_tokens=max_new_tokens,
                ),
                "passed": passed,
                "scientific_evidence": False,
            }
        finally:
            del secondary_model
            del secondary_tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
