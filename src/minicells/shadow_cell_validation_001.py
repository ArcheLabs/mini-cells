"""Shadow Cell Validation 001: copy-on-write functional isolation kill test.

This module intentionally stays independent from the Native CLM M2/M3 evidence chain.
It reuses the NativeCLM architecture and CellularLayer implementation, but generates a
fresh controlled token-predictive world and trains a fresh base model for every seed.

The experiment isolates one variable: whether an exact-clone Shadow operator with a
conditional expression gate can dominate matched direct-write/interpolation controls.
No certificate projection, growth, replay-based weight training, natural Cell discovery,
or autonomous routing adaptation is used during B adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .native_clm_v0 import NativeCLM, NativeCLMConfig

Domain = Literal["A", "B"]


@dataclass(frozen=True)
class RuleWorldSplit:
    name: str
    domain: Domain
    tokens: Tensor
    answers: Tensor
    sha256: str

    @property
    def size(self) -> int:
        return int(self.tokens.shape[0])


@dataclass(frozen=True)
class EncodedSplit:
    name: str
    domain: Domain
    hidden: Tensor
    route_idx: Tensor
    answers: Tensor

    @property
    def size(self) -> int:
        return int(self.hidden.shape[0])


@dataclass(frozen=True)
class GateState:
    mean: Tensor
    scale: Tensor
    weight: Tensor
    bias: Tensor
    auc: float


@dataclass(frozen=True)
class ShadowValidationConfig:
    base_steps: int = 3000
    base_batch_size: int = 128
    base_lr: float = 3e-4
    base_weight_decay: float = 0.01
    base_warmup_steps: int = 100
    adapt_steps: int = 1200
    adapt_batch_size: int = 128
    adapt_lr: float = 8e-4
    adapt_warmup_steps: int = 40
    grad_clip: float = 1.0
    eval_batch_size: int = 512
    encode_batch_size: int = 512
    gate_steps: int = 500
    gate_batch_size: int = 1024
    gate_lr: float = 5e-2
    gate_weight_decay: float = 1e-4
    precision: str = "fp16"

    def validate(self) -> None:
        if self.base_steps < 1 or self.adapt_steps < 1:
            raise ValueError("training steps must be positive")
        if self.base_batch_size < 1 or self.adapt_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")


def shadow_model_config() -> NativeCLMConfig:
    return NativeCLMConfig(
        vocab_size=256,
        max_seq_len=13,
        d_model=384,
        n_layers=4,
        n_heads=6,
        d_ff=1024,
        dropout=0.0,
        initial_cells=1,
        active_cells=1,
        cellular_layer_index=3,
        route_temperature=0.7,
        certificate_max_rank=0,
        cell_init_scale=0.02,
        tie_embeddings=True,
    )


def _split_sha256(tokens: Tensor, answers: Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(tokens.detach().cpu().contiguous().numpy().tobytes())
    digest.update(answers.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _record(p: int, q: int, x: int, y: int) -> bytes:
    value = f"p{p:02d}q{q:02d}x{x:02d}y{y:02d}=".encode("ascii")
    if len(value) != 13:
        raise RuntimeError("Shadow Cell world prefix must be exactly 13 bytes")
    return value


def _valid_domain(p: int, q: int, domain: Domain) -> bool:
    return p < q if domain == "A" else p > q


def generate_rule_world_split(
    *,
    name: str,
    domain: Domain,
    count: int,
    seed: int,
    used: set[tuple[int, int, int, int]] | None = None,
) -> RuleWorldSplit:
    """Generate a deterministic unique split in the overlapping-rule byte world."""

    if domain not in {"A", "B"}:
        raise ValueError("domain must be A or B")
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(int(seed))
    occupied = used if used is not None else set()
    prefixes: list[list[int]] = []
    answers: list[int] = []
    attempts = 0
    max_attempts = count * 20 + 1000
    while len(prefixes) < count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"could not generate {count} unique {domain} samples")
        p = rng.randrange(100)
        q = rng.randrange(100)
        if not _valid_domain(p, q, domain):
            continue
        x = rng.randrange(100)
        y = rng.randrange(100)
        key = (p, q, x, y)
        if key in occupied:
            continue
        occupied.add(key)
        prefix = list(_record(p, q, x, y))
        target_digit = x % 10 if domain == "A" else y % 10
        prefixes.append(prefix)
        answers.append(ord(str(target_digit)))

    token_tensor = torch.tensor(prefixes, dtype=torch.long)
    answer_tensor = torch.tensor(answers, dtype=torch.long)
    return RuleWorldSplit(
        name=name,
        domain=domain,
        tokens=token_tensor,
        answers=answer_tensor,
        sha256=_split_sha256(token_tensor, answer_tensor),
    )


def build_seed_world(seed: int, counts: dict[str, int]) -> dict[str, RuleWorldSplit]:
    expected = {
        "A_train",
        "A_calibration",
        "A_eval",
        "B_train",
        "B_calibration",
        "B_eval",
    }
    if set(counts) != expected:
        raise ValueError(f"counts must contain exactly {sorted(expected)}")
    worlds: dict[str, RuleWorldSplit] = {}
    used_a: set[tuple[int, int, int, int]] = set()
    used_b: set[tuple[int, int, int, int]] = set()
    order = [
        ("A_train", "A", used_a),
        ("A_calibration", "A", used_a),
        ("A_eval", "A", used_a),
        ("B_train", "B", used_b),
        ("B_calibration", "B", used_b),
        ("B_eval", "B", used_b),
    ]
    for index, (name, domain, used) in enumerate(order):
        worlds[name] = generate_rule_world_split(
            name=name,
            domain=domain,
            count=int(counts[name]),
            seed=int(seed) * 100 + index + 1,
            used=used,
        )
    return worlds


def world_manifest(seed: int, worlds: dict[str, RuleWorldSplit]) -> dict[str, Any]:
    return {
        "format": "minicells.shadow-cell-validation-001.data-manifest.v1",
        "seed": int(seed),
        "world": "overlapping-rule-byte-world-v1",
        "record_template": "p{p:02d}q{q:02d}x{x:02d}y{y:02d}=",
        "context_factor_A": "p < q",
        "context_factor_B": "p > q",
        "rule_A": "answer = x mod 10",
        "rule_B": "answer = y mod 10",
        "splits": {
            name: {
                "domain": split.domain,
                "samples": split.size,
                "sha256": split.sha256,
            }
            for name, split in sorted(worlds.items())
        },
    }


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _cosine_lr(step: int, total: int, warmup: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return max(1e-3, (step + 1) / max(1, warmup))
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_ratio + (1.0 - min_ratio) * cosine


def make_batch_schedule(
    *,
    samples: int,
    steps: int,
    batch_size: int,
    seed: int,
) -> Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randint(0, samples, (steps, batch_size), generator=generator)


def _shared_hidden_before_cell(model: NativeCLM, tokens: Tensor) -> Tensor:
    if model.config.cellular_layer_index != model.config.n_layers - 1:
        raise RuntimeError("Shadow validation requires the Cellular Layer after the last shared block")
    batch, seq_len = tokens.shape
    positions = torch.arange(seq_len, device=tokens.device)
    x = model.token_embedding(tokens) + model.position_embedding(positions)[None, :, :]
    x = model.dropout(x)
    for block in model.blocks:
        x = block(x)
    return x[:, -1, :]


def _route_final(model: NativeCLM, hidden: Tensor) -> tuple[Tensor, Tensor]:
    route_input = model.cellular.norm(hidden)
    query = F.normalize(model.cellular.query_proj(route_input), dim=-1)
    keys = torch.stack(
        [F.normalize(cell.route_key, dim=0) for cell in model.cellular.cells],
        dim=0,
    )
    scores = query.matmul(keys.transpose(0, 1)) / model.config.route_temperature
    top_scores, top_idx = torch.topk(scores, k=1, dim=-1)
    return top_idx[:, 0], top_scores[:, 0]


def _answer_logits_from_state(
    model: NativeCLM,
    hidden: Tensor,
    route_idx: Tensor,
    *,
    parent_id: int | None = None,
    candidate_weight: Tensor | None = None,
    expression: float | Tensor = 1.0,
) -> Tensor:
    if hidden.ndim != 2 or hidden.shape[-1] != model.config.d_model:
        raise ValueError("hidden must have shape [batch, d_model]")
    if route_idx.ndim != 1 or route_idx.shape[0] != hidden.shape[0]:
        raise ValueError("route_idx must have shape [batch]")
    if candidate_weight is not None and parent_id is None:
        raise ValueError("candidate_weight requires parent_id")
    cell_out = torch.zeros_like(hidden)
    if isinstance(expression, Tensor):
        expr = expression.to(device=hidden.device, dtype=hidden.dtype).reshape(-1)
        if expr.numel() != hidden.shape[0]:
            raise ValueError("expression tensor must have one value per example")
    else:
        expr = hidden.new_full((hidden.shape[0],), float(expression))

    for cell_id, cell in enumerate(model.cellular.cells):
        mask = route_idx == cell_id
        if not bool(mask.any()):
            continue
        selected = hidden[mask]
        if cell_id == parent_id and candidate_weight is not None:
            parent_weight = cell.weight.detach()
            base = F.linear(selected, parent_weight)
            candidate = F.linear(selected, candidate_weight)
            alpha = expr[mask].unsqueeze(-1)
            value = base + alpha * (candidate - base)
        else:
            value = cell(selected)
        cell_out[mask] = value
    final_hidden = model.final_norm(hidden + cell_out)
    return model.lm_head(final_hidden)


def answer_logits(
    model: NativeCLM,
    tokens: Tensor,
    *,
    parent_id: int | None = None,
    candidate_weight: Tensor | None = None,
    expression: float | Tensor = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    hidden = _shared_hidden_before_cell(model, tokens)
    route_idx, _ = _route_final(model, hidden)
    logits = _answer_logits_from_state(
        model,
        hidden,
        route_idx,
        parent_id=parent_id,
        candidate_weight=candidate_weight,
        expression=expression,
    )
    return logits, hidden, route_idx


def validate_standard_forward_equivalence(model: NativeCLM, tokens: Tensor) -> float:
    """Verify the answer-only helper exactly matches NativeCLM at the final position."""

    model.eval()
    with torch.no_grad():
        reference = model(tokens)["logits"][:, -1, :]
        candidate, _, _ = answer_logits(model, tokens)
    return float((reference - candidate).abs().max().item())


def train_base_model(
    *,
    seed: int,
    split: RuleWorldSplit,
    device: torch.device,
    config: ShadowValidationConfig,
) -> tuple[NativeCLM, dict[str, Any]]:
    config.validate()
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = NativeCLM(shadow_model_config()).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.base_lr,
        betas=(0.9, 0.95),
        weight_decay=config.base_weight_decay,
    )
    schedule = make_batch_schedule(
        samples=split.size,
        steps=config.base_steps,
        batch_size=config.base_batch_size,
        seed=seed + 11,
    )
    scaler_enabled = device.type == "cuda" and config.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    losses: list[float] = []
    for step in range(config.base_steps):
        factor = _cosine_lr(step, config.base_steps, config.base_warmup_steps)
        optimizer.param_groups[0]["lr"] = config.base_lr * factor
        idx = schedule[step]
        tokens = split.tokens[idx].to(device)
        answers = split.answers[idx].to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, config.precision):
            logits = model(tokens)["logits"][:, -1, :]
            loss = F.cross_entropy(logits, answers)
        scaler.scale(loss).backward()
        if scaler_enabled:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return model, {
        "steps": config.base_steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_last_100_loss": float(sum(losses[-100:]) / min(100, len(losses))),
    }


@torch.no_grad()
def encode_split(
    model: NativeCLM,
    split: RuleWorldSplit,
    *,
    device: torch.device,
    batch_size: int,
) -> EncodedSplit:
    model.eval()
    hidden_values: list[Tensor] = []
    route_values: list[Tensor] = []
    for start in range(0, split.size, batch_size):
        end = min(split.size, start + batch_size)
        tokens = split.tokens[start:end].to(device)
        hidden = _shared_hidden_before_cell(model, tokens)
        route_idx, _ = _route_final(model, hidden)
        hidden_values.append(hidden.detach().float().cpu())
        route_values.append(route_idx.detach().cpu())
    return EncodedSplit(
        name=split.name,
        domain=split.domain,
        hidden=torch.cat(hidden_values, dim=0),
        route_idx=torch.cat(route_values, dim=0),
        answers=split.answers.clone(),
    )


def parent_route_share(encoded: EncodedSplit, parent_id: int) -> float:
    return float((encoded.route_idx == int(parent_id)).float().mean().item())


def select_mature_parent(encoded_a_eval: EncodedSplit, cell_count: int) -> tuple[int, list[float]]:
    counts = torch.bincount(encoded_a_eval.route_idx, minlength=cell_count).float()
    shares = counts / max(1, encoded_a_eval.size)
    parent = int(torch.argmax(shares).item())
    return parent, [float(v) for v in shares.tolist()]


def _evaluate_encoded(
    model: NativeCLM,
    encoded: EncodedSplit,
    *,
    device: torch.device,
    batch_size: int,
    parent_id: int | None = None,
    candidate_weight: Tensor | None = None,
    expressions: Tensor | None = None,
    global_expression: float = 1.0,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for start in range(0, encoded.size, batch_size):
        end = min(encoded.size, start + batch_size)
        hidden = encoded.hidden[start:end].to(device)
        route_idx = encoded.route_idx[start:end].to(device)
        answers = encoded.answers[start:end].to(device)
        if expressions is None:
            expression: float | Tensor = global_expression
        else:
            expression = expressions[start:end].to(device)
        with torch.no_grad():
            logits = _answer_logits_from_state(
                model,
                hidden,
                route_idx,
                parent_id=parent_id,
                candidate_weight=candidate_weight,
                expression=expression,
            )
            loss = F.cross_entropy(logits, answers, reduction="sum")
            prediction = torch.argmax(logits, dim=-1)
        total_loss += float(loss.cpu())
        total_correct += int((prediction == answers).sum().item())
        total += int(answers.numel())
    return {
        "nll": total_loss / max(1, total),
        "accuracy": total_correct / max(1, total),
    }


def evaluate_base(
    model: NativeCLM,
    encoded: EncodedSplit,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    return _evaluate_encoded(
        model,
        encoded,
        device=device,
        batch_size=batch_size,
    )


def _train_candidate_operator(
    *,
    model: NativeCLM,
    parent_id: int,
    encoded_b_train: EncodedSplit,
    schedule: Tensor,
    device: torch.device,
    config: ShadowValidationConfig,
) -> tuple[Tensor, dict[str, Any]]:
    parent = model.cellular.cells[parent_id].weight.detach().to(device)
    candidate = nn.Parameter(parent.clone())
    optimizer = torch.optim.AdamW(
        [candidate],
        lr=config.adapt_lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    scaler_enabled = device.type == "cuda" and config.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    losses: list[float] = []
    for step in range(config.adapt_steps):
        factor = _cosine_lr(step, config.adapt_steps, config.adapt_warmup_steps)
        optimizer.param_groups[0]["lr"] = config.adapt_lr * factor
        idx = schedule[step]
        hidden = encoded_b_train.hidden[idx].to(device)
        route_idx = encoded_b_train.route_idx[idx].to(device)
        answers = encoded_b_train.answers[idx].to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, config.precision):
            logits = _answer_logits_from_state(
                model,
                hidden,
                route_idx,
                parent_id=parent_id,
                candidate_weight=candidate,
                expression=1.0,
            )
            loss = F.cross_entropy(logits, answers)
        scaler.scale(loss).backward()
        if scaler_enabled:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([candidate], config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return candidate.detach(), {
        "steps": config.adapt_steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_last_100_loss": float(sum(losses[-100:]) / min(100, len(losses))),
    }


def train_matched_direct_and_shadow(
    *,
    model: NativeCLM,
    parent_id: int,
    encoded_b_train: EncodedSplit,
    seed: int,
    device: torch.device,
    config: ShadowValidationConfig,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    schedule = make_batch_schedule(
        samples=encoded_b_train.size,
        steps=config.adapt_steps,
        batch_size=config.adapt_batch_size,
        seed=seed + 313,
    )
    direct, direct_summary = _train_candidate_operator(
        model=model,
        parent_id=parent_id,
        encoded_b_train=encoded_b_train,
        schedule=schedule,
        device=device,
        config=config,
    )
    shadow, shadow_summary = _train_candidate_operator(
        model=model,
        parent_id=parent_id,
        encoded_b_train=encoded_b_train,
        schedule=schedule,
        device=device,
        config=config,
    )
    denom = float(torch.linalg.vector_norm(direct).item()) + 1e-12
    relative_error = float(torch.linalg.vector_norm(direct - shadow).item()) / denom
    return direct, shadow, {
        "direct": direct_summary,
        "shadow": shadow_summary,
        "operator_relative_error": relative_error,
    }


def _standardize_train(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    mean = x.mean(dim=0)
    scale = x.std(dim=0).clamp_min(1e-5)
    return (x - mean) / scale, mean, scale


def _roc_auc(scores: Tensor, labels: Tensor) -> float:
    scores = scores.detach().double().cpu().reshape(-1)
    labels = labels.detach().long().cpu().reshape(-1)
    positives = int((labels == 1).sum().item())
    negatives = int((labels == 0).sum().item())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.double)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.double)
    positive_rank_sum = float(ranks[labels == 1].sum().item())
    u = positive_rank_sum - positives * (positives + 1) / 2
    return float(u / (positives * negatives))


def train_gate_probe(
    *,
    a_cal: EncodedSplit,
    b_cal: EncodedSplit,
    a_eval: EncodedSplit,
    b_eval: EncodedSplit,
    seed: int,
    device: torch.device,
    config: ShadowValidationConfig,
) -> GateState:
    x = torch.cat([a_cal.hidden, b_cal.hidden], dim=0).float()
    y = torch.cat(
        [
            torch.zeros(a_cal.size, dtype=torch.float32),
            torch.ones(b_cal.size, dtype=torch.float32),
        ],
        dim=0,
    )
    x_std, mean, scale = _standardize_train(x)
    linear = nn.Linear(x.shape[1], 1).to(device)
    torch.manual_seed(seed + 701)
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.zero_()
    optimizer = torch.optim.AdamW(
        linear.parameters(),
        lr=config.gate_lr,
        weight_decay=config.gate_weight_decay,
    )
    generator = torch.Generator().manual_seed(seed + 702)
    for _ in range(config.gate_steps):
        idx = torch.randint(
            0,
            x_std.shape[0],
            (config.gate_batch_size,),
            generator=generator,
        )
        xb = x_std[idx].to(device)
        yb = y[idx].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = linear(xb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        optimizer.step()

    eval_x = torch.cat([a_eval.hidden, b_eval.hidden], dim=0).float()
    eval_y = torch.cat(
        [
            torch.zeros(a_eval.size, dtype=torch.long),
            torch.ones(b_eval.size, dtype=torch.long),
        ],
        dim=0,
    )
    with torch.no_grad():
        z = (eval_x - mean) / scale
        scores = torch.sigmoid(linear(z.to(device)).squeeze(-1)).cpu()
    return GateState(
        mean=mean.cpu(),
        scale=scale.cpu(),
        weight=linear.weight.detach().cpu().reshape(-1),
        bias=linear.bias.detach().cpu().reshape(()),
        auc=_roc_auc(scores, eval_y),
    )


def gate_values(gate: GateState, hidden: Tensor) -> Tensor:
    z = (hidden.float() - gate.mean) / gate.scale
    logits = z.matmul(gate.weight) + gate.bias
    return torch.sigmoid(logits)


def _derived_point(
    *,
    maturity: float,
    a_metrics: dict[str, float],
    b_metrics: dict[str, float],
    base_a: dict[str, float],
    base_b: dict[str, float],
    direct_b_gain: float,
) -> dict[str, float]:
    a_regression = max(0.0, base_a["accuracy"] - a_metrics["accuracy"])
    b_gain = max(0.0, b_metrics["accuracy"] - base_b["accuracy"])
    gain_fraction = b_gain / max(1e-12, direct_b_gain)
    return {
        "maturity": float(maturity),
        "A_regression": float(a_regression),
        "B_gain": float(b_gain),
        "B_gain_fraction_of_direct": float(gain_fraction),
        "A_accuracy": float(a_metrics["accuracy"]),
        "B_accuracy": float(b_metrics["accuracy"]),
        "A_nll": float(a_metrics["nll"]),
        "B_nll": float(b_metrics["nll"]),
    }


def pareto_hypervolume(points: list[dict[str, float]]) -> float:
    """Area dominated toward reference (damage=1, gain=0)."""

    xy = sorted(
        {
            (
                min(1.0, max(0.0, float(point["A_regression"]))),
                min(1.0, max(0.0, float(point["B_gain_fraction_of_direct"]))),
            )
            for point in points
        },
        key=lambda item: item[0],
    )
    if not xy:
        return 0.0
    area = 0.0
    best_y = 0.0
    for index, (x, y) in enumerate(xy):
        best_y = max(best_y, y)
        next_x = xy[index + 1][0] if index + 1 < len(xy) else 1.0
        if next_x > x:
            area += (next_x - x) * best_y
    return float(area)


def _curve(
    *,
    model: NativeCLM,
    parent_id: int,
    candidate_weight: Tensor,
    a_eval: EncodedSplit,
    b_eval: EncodedSplit,
    a_gate: Tensor | None,
    b_gate: Tensor | None,
    maturity_grid: list[float],
    base_a: dict[str, float],
    base_b: dict[str, float],
    direct_b_gain: float,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for maturity in maturity_grid:
        if a_gate is None:
            a_expr = None
            b_expr = None
            global_expression = float(maturity)
        else:
            a_expr = a_gate * float(maturity)
            b_expr = b_gate * float(maturity) if b_gate is not None else None
            global_expression = 1.0
        a_metrics = _evaluate_encoded(
            model,
            a_eval,
            device=device,
            batch_size=batch_size,
            parent_id=parent_id,
            candidate_weight=candidate_weight,
            expressions=a_expr,
            global_expression=global_expression,
        )
        b_metrics = _evaluate_encoded(
            model,
            b_eval,
            device=device,
            batch_size=batch_size,
            parent_id=parent_id,
            candidate_weight=candidate_weight,
            expressions=b_expr,
            global_expression=global_expression,
        )
        points.append(
            _derived_point(
                maturity=float(maturity),
                a_metrics=a_metrics,
                b_metrics=b_metrics,
                base_a=base_a,
                base_b=base_b,
                direct_b_gain=direct_b_gain,
            )
        )
    return points


def _primary_point(
    points: list[dict[str, float]],
    *,
    maximum_a_regression: float,
) -> dict[str, float] | None:
    eligible = [point for point in points if point["A_regression"] <= maximum_a_regression]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda point: (-point["B_gain_fraction_of_direct"], point["maturity"]),
    )[0]


def _metric_delta(a: dict[str, float], b: dict[str, float]) -> float:
    return max(abs(float(a[key]) - float(b[key])) for key in ("accuracy", "nll"))


def _tensor_sha256(tensor: Tensor) -> str:
    digest = hashlib.sha256()
    value = tensor.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _model_state_sha256(model: NativeCLM) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def run_shadow_validation_seed(
    *,
    seed: int,
    counts: dict[str, int],
    maturity_grid: list[float],
    thresholds: dict[str, float | bool],
    output_dir: str | Path,
    device: str = "cuda",
    config: ShadowValidationConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ShadowValidationConfig()
    cfg.validate()
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    worlds = build_seed_world(seed, counts)
    manifest = world_manifest(seed, worlds)
    (output / "data-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    model, base_train = train_base_model(
        seed=seed,
        split=worlds["A_train"],
        device=target_device,
        config=cfg,
    )
    sample_tokens = worlds["A_eval"].tokens[:32].to(target_device)
    helper_equivalence = validate_standard_forward_equivalence(model, sample_tokens)

    encoded = {
        name: encode_split(
            model,
            split,
            device=target_device,
            batch_size=cfg.encode_batch_size,
        )
        for name, split in worlds.items()
    }
    base_a = evaluate_base(
        model,
        encoded["A_eval"],
        device=target_device,
        batch_size=cfg.eval_batch_size,
    )
    base_b = evaluate_base(
        model,
        encoded["B_eval"],
        device=target_device,
        batch_size=cfg.eval_batch_size,
    )
    parent_id, a_usage = select_mature_parent(encoded["A_eval"], model.cell_count)
    parent_share_a = parent_route_share(encoded["A_eval"], parent_id)
    parent_share_b = parent_route_share(encoded["B_eval"], parent_id)

    parent_hash_before = _tensor_sha256(model.cellular.cells[parent_id].weight)
    model_hash_before = _model_state_sha256(model)
    parent_weight = model.cellular.cells[parent_id].weight.detach().clone()

    birth_metrics_a = _evaluate_encoded(
        model,
        encoded["A_eval"],
        device=target_device,
        batch_size=cfg.eval_batch_size,
        parent_id=parent_id,
        candidate_weight=parent_weight,
        global_expression=1.0,
    )
    birth_metrics_b = _evaluate_encoded(
        model,
        encoded["B_eval"],
        device=target_device,
        batch_size=cfg.eval_batch_size,
        parent_id=parent_id,
        candidate_weight=parent_weight,
        global_expression=1.0,
    )
    birth_metric_drift = max(
        _metric_delta(base_a, birth_metrics_a),
        _metric_delta(base_b, birth_metrics_b),
    )

    direct_weight, shadow_weight, adapt_summary = train_matched_direct_and_shadow(
        model=model,
        parent_id=parent_id,
        encoded_b_train=encoded["B_train"],
        seed=seed,
        device=target_device,
        config=cfg,
    )
    parent_hash_after = _tensor_sha256(model.cellular.cells[parent_id].weight)
    model_hash_after = _model_state_sha256(model)

    direct_a = _evaluate_encoded(
        model,
        encoded["A_eval"],
        device=target_device,
        batch_size=cfg.eval_batch_size,
        parent_id=parent_id,
        candidate_weight=direct_weight,
        global_expression=1.0,
    )
    direct_b = _evaluate_encoded(
        model,
        encoded["B_eval"],
        device=target_device,
        batch_size=cfg.eval_batch_size,
        parent_id=parent_id,
        candidate_weight=direct_weight,
        global_expression=1.0,
    )
    direct_b_gain = max(0.0, direct_b["accuracy"] - base_b["accuracy"])
    direct_point = _derived_point(
        maturity=1.0,
        a_metrics=direct_a,
        b_metrics=direct_b,
        base_a=base_a,
        base_b=base_b,
        direct_b_gain=direct_b_gain,
    )

    gate = train_gate_probe(
        a_cal=encoded["A_calibration"],
        b_cal=encoded["B_calibration"],
        a_eval=encoded["A_eval"],
        b_eval=encoded["B_eval"],
        seed=seed,
        device=target_device,
        config=cfg,
    )
    a_gate = gate_values(gate, encoded["A_eval"].hidden)
    b_gate = gate_values(gate, encoded["B_eval"].hidden)
    combined_gate = torch.cat([a_gate, b_gate], dim=0)
    generator = torch.Generator().manual_seed(seed + 909)
    perm = torch.randperm(combined_gate.numel(), generator=generator)
    shuffled = combined_gate[perm]
    shuffled_a = shuffled[: a_gate.numel()]
    shuffled_b = shuffled[a_gate.numel() :]

    direct_interp = _curve(
        model=model,
        parent_id=parent_id,
        candidate_weight=direct_weight,
        a_eval=encoded["A_eval"],
        b_eval=encoded["B_eval"],
        a_gate=None,
        b_gate=None,
        maturity_grid=maturity_grid,
        base_a=base_a,
        base_b=base_b,
        direct_b_gain=direct_b_gain,
        device=target_device,
        batch_size=cfg.eval_batch_size,
    )
    shadow_global = _curve(
        model=model,
        parent_id=parent_id,
        candidate_weight=shadow_weight,
        a_eval=encoded["A_eval"],
        b_eval=encoded["B_eval"],
        a_gate=None,
        b_gate=None,
        maturity_grid=maturity_grid,
        base_a=base_a,
        base_b=base_b,
        direct_b_gain=direct_b_gain,
        device=target_device,
        batch_size=cfg.eval_batch_size,
    )
    shadow_conditional = _curve(
        model=model,
        parent_id=parent_id,
        candidate_weight=shadow_weight,
        a_eval=encoded["A_eval"],
        b_eval=encoded["B_eval"],
        a_gate=a_gate,
        b_gate=b_gate,
        maturity_grid=maturity_grid,
        base_a=base_a,
        base_b=base_b,
        direct_b_gain=direct_b_gain,
        device=target_device,
        batch_size=cfg.eval_batch_size,
    )
    shadow_shuffled = _curve(
        model=model,
        parent_id=parent_id,
        candidate_weight=shadow_weight,
        a_eval=encoded["A_eval"],
        b_eval=encoded["B_eval"],
        a_gate=shuffled_a,
        b_gate=shuffled_b,
        maturity_grid=maturity_grid,
        base_a=base_a,
        base_b=base_b,
        direct_b_gain=direct_b_gain,
        device=target_device,
        batch_size=cfg.eval_batch_size,
    )

    maximum_a_regression = float(thresholds["maximum_primary_A_regression"])
    primary = _primary_point(
        shadow_conditional,
        maximum_a_regression=maximum_a_regression,
    )
    primary_shuffled = None
    if primary is not None:
        maturity = primary["maturity"]
        primary_shuffled = next(
            point for point in shadow_shuffled if point["maturity"] == maturity
        )

    hv_direct = pareto_hypervolume(direct_interp)
    hv_global = pareto_hypervolume(shadow_global)
    hv_conditional = pareto_hypervolume(shadow_conditional)
    hv_shuffled = pareto_hypervolume(shadow_shuffled)
    hv_improvement_direct = (hv_conditional - hv_direct) / max(1e-12, hv_direct)
    hv_improvement_global = (hv_conditional - hv_global) / max(1e-12, hv_global)

    identity_curve_delta = 0.0
    for direct_row, global_row in zip(direct_interp, shadow_global, strict=True):
        for key in (
            "A_regression",
            "B_gain_fraction_of_direct",
            "A_accuracy",
            "B_accuracy",
            "A_nll",
            "B_nll",
        ):
            identity_curve_delta = max(
                identity_curve_delta,
                abs(float(direct_row[key]) - float(global_row[key])),
            )
    immediate = shadow_global[-1]
    immediate_direct_delta = max(
        abs(float(immediate[key]) - float(direct_point[key]))
        for key in (
            "A_regression",
            "B_gain_fraction_of_direct",
            "A_accuracy",
            "B_accuracy",
            "A_nll",
            "B_nll",
        )
    )

    base_ok = base_a["accuracy"] >= 0.95 and base_a["nll"] <= 0.20
    conflict_ok = parent_share_a >= 0.80 and parent_share_b >= 0.80
    direct_ok = direct_b_gain >= float(thresholds["minimum_direct_B_gain"])
    gate_ok = gate.auc >= 0.90
    identity_ok = (
        adapt_summary["operator_relative_error"] <= 1e-6
        and identity_curve_delta <= 1e-5
        and immediate_direct_delta <= 1e-5
        and birth_metric_drift <= 1e-5
        and parent_hash_before == parent_hash_after
        and model_hash_before == model_hash_after
        and helper_equivalence <= 1e-5
    )
    primary_pass = (
        primary is not None
        and primary["B_gain_fraction_of_direct"]
        >= float(thresholds["minimum_primary_B_gain_fraction_of_direct"])
    )
    immediate_pass = (
        immediate["A_regression"] <= maximum_a_regression
        and immediate["B_gain_fraction_of_direct"]
        >= float(thresholds["minimum_primary_B_gain_fraction_of_direct"])
    )
    shuffled_a_advantage = None
    shuffled_b_gain_drop = None
    if primary is not None and primary_shuffled is not None:
        shuffled_a_advantage = primary_shuffled["A_regression"] - primary["A_regression"]
        shuffled_b_gain_drop = (
            primary["B_gain_fraction_of_direct"]
            - primary_shuffled["B_gain_fraction_of_direct"]
        )

    base_checkpoint = output / "base.pt"
    model.save_checkpoint(
        base_checkpoint,
        extra={
            "experiment": "Shadow Cell Validation 001",
            "seed": seed,
            "role": "fresh_A_base",
            "parent_id": parent_id,
            "data_manifest": manifest,
        },
    )
    candidate_checkpoint = output / "candidates.pt"
    torch.save(
        {
            "format": "minicells.shadow-cell-validation-001.candidates.v1",
            "seed": seed,
            "parent_id": parent_id,
            "direct_weight": direct_weight.detach().cpu(),
            "shadow_weight": shadow_weight.detach().cpu(),
            "gate": {
                "mean": gate.mean,
                "scale": gate.scale,
                "weight": gate.weight,
                "bias": gate.bias,
                "auc": gate.auc,
            },
        },
        candidate_checkpoint,
    )

    result = {
        "format": "minicells.shadow-cell-validation-001.seed-result.v1",
        "seed": int(seed),
        "scientific_decision": False,
        "independent_of_native_clm_m2_chain": True,
        "data_manifest": manifest,
        "base_training": base_train,
        "base_metrics": {"A": base_a, "B": base_b},
        "parent": {
            "id": parent_id,
            "A_usage_shares": a_usage,
            "top1_share_A": parent_share_a,
            "top1_share_B": parent_share_b,
            "operator_sha256_before": parent_hash_before,
            "operator_sha256_after": parent_hash_after,
        },
        "birth": {
            "metric_drift": birth_metric_drift,
            "helper_standard_forward_max_logit_drift": helper_equivalence,
        },
        "adaptation": adapt_summary,
        "direct_tx": direct_point,
        "gate": {
            "heldout_auc": gate.auc,
            "A_mean": float(a_gate.mean().item()),
            "B_mean": float(b_gate.mean().item()),
        },
        "curves": {
            "Direct-Interp": direct_interp,
            "Shadow-Global": shadow_global,
            "Shadow-Conditional": shadow_conditional,
            "Shadow-Conditional-Shuffled": shadow_shuffled,
        },
        "hypervolume": {
            "Direct-Interp": hv_direct,
            "Shadow-Global": hv_global,
            "Shadow-Conditional": hv_conditional,
            "Shadow-Conditional-Shuffled": hv_shuffled,
            "conditional_improvement_vs_direct_interp": hv_improvement_direct,
            "conditional_improvement_vs_shadow_global": hv_improvement_global,
        },
        "primary_conditional": primary,
        "primary_shuffled": primary_shuffled,
        "causal_control": {
            "correct_vs_shuffled_A_regression_advantage": shuffled_a_advantage,
            "correct_vs_shuffled_B_gain_fraction_drop": shuffled_b_gain_drop,
        },
        "identity": {
            "direct_shadow_operator_relative_error": adapt_summary["operator_relative_error"],
            "direct_interp_shadow_global_max_metric_delta": identity_curve_delta,
            "shadow_immediate_direct_tx_max_metric_delta": immediate_direct_delta,
            "base_model_state_unchanged_during_B_adaptation": model_hash_before == model_hash_after,
            "parent_operator_immutable_during_shadow_training": parent_hash_before == parent_hash_after,
        },
        "gates": {
            "base_training": base_ok,
            "parent_conflict": conflict_ok,
            "direct_plasticity": direct_ok,
            "gate_capacity": gate_ok,
            "identity_control": identity_ok,
            "conditional_primary": bool(primary_pass),
            "immediate_primary": bool(immediate_pass),
        },
        "artifacts": {
            "base_checkpoint": str(base_checkpoint),
            "candidate_checkpoint": str(candidate_checkpoint),
        },
    }
    (output / "seed-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def classify_shadow_validation(
    seed_results: list[dict[str, Any]],
    *,
    thresholds: dict[str, float | bool],
) -> str:
    if not seed_results:
        raise ValueError("at least one seed result is required")
    if any(not result["gates"]["base_training"] for result in seed_results):
        return "INCONCLUSIVE_BASE_TRAINING"
    if any(not result["gates"]["parent_conflict"] for result in seed_results):
        return "INCONCLUSIVE_PARENT_CONFLICT"
    if any(not result["gates"]["direct_plasticity"] for result in seed_results):
        return "INCONCLUSIVE_DIRECT_PLASTICITY"
    if any(not result["gates"]["gate_capacity"] for result in seed_results):
        return "INCONCLUSIVE_GATE_CAPACITY"
    if any(not result["gates"]["identity_control"] for result in seed_results):
        return "INCONCLUSIVE_IDENTITY_CONTROL"
    if any(not result["gates"]["conditional_primary"] for result in seed_results):
        return "SHADOW_CELL_NOT_SUPPORTED"

    minimum_hv = float(
        thresholds["minimum_conditional_hypervolume_improvement_vs_direct_interp"]
    )
    if any(
        result["hypervolume"]["conditional_improvement_vs_direct_interp"] < minimum_hv
        for result in seed_results
    ):
        return "ISOLATED_SHADOW_ADVANTAGE_NOT_SUPPORTED"

    minimum_global = float(
        thresholds["minimum_conditional_hypervolume_improvement_vs_shadow_global"]
    )
    minimum_shuffle = float(
        thresholds["minimum_correct_vs_shuffled_A_regression_advantage"]
    )
    if any(result["gates"]["immediate_primary"] for result in seed_results):
        return "SHADOW_ISOLATION_SUPPORTED_MATURATION_NOT_NECESSARY"
    if any(
        result["hypervolume"]["conditional_improvement_vs_shadow_global"] < minimum_global
        for result in seed_results
    ):
        return "SHADOW_ISOLATION_SUPPORTED_MATURATION_NOT_NECESSARY"
    if any(
        result["causal_control"]["correct_vs_shuffled_A_regression_advantage"] is None
        or result["causal_control"]["correct_vs_shuffled_A_regression_advantage"]
        < minimum_shuffle
        for result in seed_results
    ):
        return "SHADOW_ISOLATION_SUPPORTED_MATURATION_NOT_NECESSARY"
    return "SHADOW_CELL_CONTROLLED_MATURATION_SUPPORTED"


def aggregate_shadow_validation(
    seed_results: list[dict[str, Any]],
    *,
    thresholds: dict[str, float | bool],
    protocol_sha256: str,
    phase: str,
) -> dict[str, Any]:
    classification = classify_shadow_validation(seed_results, thresholds=thresholds)
    return {
        "format": "minicells.shadow-cell-validation-001.result.v1",
        "phase": phase,
        "classification": classification,
        "scientific_decision": phase == "formal",
        "independent_of_native_clm_m2_chain": True,
        "native_clm_m2_decision_modified": False,
        "protocol_sha256": protocol_sha256,
        "seeds": [int(result["seed"]) for result in seed_results],
        "seed_results": seed_results,
    }
