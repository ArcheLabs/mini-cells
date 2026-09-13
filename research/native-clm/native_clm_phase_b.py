from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import native_clm_runtime as base
from native_clm_candidates import C1_DIM, C1_FFN, C1_GATE_BIAS, C1_HEADS, VERIFIED_DATASET_REVISION

base.DATASET_REVISION = VERIFIED_DATASET_REVISION

M1_NAME = "M1-c1-rmsnorm"
M2_NAME = "M2-c1-swiglu"
M3_NAME = "M3-c1-rope"
M4_NAME = "M4-c1-modernized"
N1_NAME = "N1-residual-only"
N2_NAME = "N2-dual-gate-aru"
N3_NAME = "N3-step-conditioned-write"
N4_NAME = "N4-adaptive-communication"

M_NAMES = (M1_NAME, M2_NAME, M3_NAME, M4_NAME)
N_NAMES = (N1_NAME, N2_NAME, N3_NAME, N4_NAME)
PHASE_B_NAMES = M_NAMES + N_NAMES
PARAMETER_TOLERANCE = 0.01

M_SWIGLU_FFN = 504
N1_FFN = 932
N1_ALPHA = 1.0 / (1.0 + math.exp(2.0))
N2_GATE_RANK = C1_DIM // 2
N2_RETAIN_BIAS = 6.0
N2_WRITE_BIAS = -2.0
N4_COMM_BIAS = 6.0


def _init(module: nn.Module) -> None:
    base.HistoricalScaledCLM._init_weights(module)


def _norm(kind: str, dim: int) -> nn.Module:
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    if kind == "rmsnorm":
        return base.RMSNorm(dim)
    raise ValueError(kind)


class LocalRoPECausalSelfAttention(nn.Module):
    """Local causal attention with the same projections as C1, but RoPE on q/k."""

    def __init__(self, dim: int, heads: int, window: int):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.window = window
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = base._rope(q), base._rope(k)
        qi = torch.arange(t, device=x.device)[:, None]
        ki = torch.arange(t, device=x.device)[None, :]
        allowed = (ki <= qi) & (ki >= qi - self.window + 1)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed[None, None, :, :], dropout_p=0.0, is_causal=False
        )
        return self.out(y.transpose(1, 2).contiguous().view(b, t, self.dim))


class PhaseBStage(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        ffn_dim: int,
        window: int,
        *,
        norm_kind: str = "layernorm",
        activation: str = "gelu",
        use_rope: bool = False,
        update: str = "single-gate",
        step_conditioned_write: bool = False,
        adaptive_communication: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.window = window
        self.iterations = 4
        self.update = update
        self.step_conditioned_write = step_conditioned_write
        self.adaptive_communication = adaptive_communication
        self.n1 = _norm(norm_kind, dim)
        self.n2 = _norm(norm_kind, dim)
        attn_cls = LocalRoPECausalSelfAttention if use_rope else base.LocalCausalSelfAttention
        self.attn = attn_cls(dim, heads, window)
        if activation == "gelu":
            self.ff = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))
        elif activation == "swiglu":
            self.ff = base.SwiGLU(dim, ffn_dim)
        else:
            raise ValueError(activation)
        self.step = nn.Parameter(torch.empty(self.iterations, dim))
        nn.init.normal_(self.step, 0.0, 0.02)

        self.gate: nn.Linear | None = None
        self.gate_in: nn.Linear | None = None
        self.gate_out: nn.Linear | None = None
        self.step_gate_bias: nn.Parameter | None = None
        self.comm_gate: nn.Linear | None = None

        if update == "single-gate":
            self.gate = nn.Linear(2 * dim, dim)
        elif update == "residual-only":
            pass
        elif update == "dual-gate":
            self.gate_in = nn.Linear(2 * dim, N2_GATE_RANK, bias=False)
            self.gate_out = nn.Linear(N2_GATE_RANK, 2 * dim, bias=True)
        else:
            raise ValueError(update)

        if step_conditioned_write:
            if update != "single-gate":
                raise ValueError("step-conditioned write requires the C1 single gate")
            self.step_gate_bias = nn.Parameter(torch.zeros(self.iterations, dim))
        if adaptive_communication:
            self.comm_gate = nn.Linear(dim, 1)

    def reset_special_parameters(self) -> None:
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, C1_GATE_BIAS)
        if self.gate_out is not None:
            nn.init.zeros_(self.gate_out.weight)
            with torch.no_grad():
                self.gate_out.bias[: self.dim].fill_(N2_RETAIN_BIAS)
                self.gate_out.bias[self.dim :].fill_(N2_WRITE_BIAS)
        if self.step_gate_bias is not None:
            nn.init.zeros_(self.step_gate_bias)
        if self.comm_gate is not None:
            nn.init.zeros_(self.comm_gate.weight)
            nn.init.constant_(self.comm_gate.bias, N4_COMM_BIAS)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        b, t, d = state.shape
        for step in range(self.iterations):
            conditioned = state + self.step[step].view(1, 1, d)
            communication_input = self.n1(conditioned)
            attention_delta = self.attn(communication_input)
            if self.comm_gate is not None:
                # Soft state-conditioned communication. This probes whether adaptive
                # communication helps representation quality; it does NOT skip the
                # attention kernel and therefore claims no realized FLOP reduction.
                strength = torch.sigmoid(self.comm_gate(communication_input))
                attention_delta = strength * attention_delta
            candidate = state + attention_delta
            ffn_delta = self.ff(self.n2(candidate))
            proposal = attention_delta + ffn_delta

            if self.update == "residual-only":
                state = state + N1_ALPHA * proposal
            elif self.update == "single-gate":
                assert self.gate is not None
                gate_logits = self.gate(torch.cat((state, proposal), dim=-1))
                if self.step_gate_bias is not None:
                    gate_logits = gate_logits + self.step_gate_bias[step].view(1, 1, d)
                state = state + torch.sigmoid(gate_logits) * proposal
            else:
                assert self.gate_in is not None and self.gate_out is not None
                hidden = F.silu(self.gate_in(torch.cat((state, proposal), dim=-1)))
                retain_logits, write_logits = self.gate_out(hidden).chunk(2, dim=-1)
                retain = torch.sigmoid(retain_logits)
                write = torch.sigmoid(write_logits)
                state = retain * state + write * proposal
        return state


class PhaseBLM(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        ffn_dim: int,
        vocab_size: int,
        norm_kind: str = "layernorm",
        activation: str = "gelu",
        use_rope: bool = False,
        update: str = "single-gate",
        step_conditioned_write: bool = False,
        adaptive_communication: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.use_rope = use_rope
        self.token = nn.Embedding(vocab_size, dim)
        self.pos = None if use_rope else nn.Embedding(base.CONTEXT_LENGTH, dim)
        self.stages = nn.ModuleList(
            [
                PhaseBStage(
                    dim,
                    heads,
                    ffn_dim,
                    window,
                    norm_kind=norm_kind,
                    activation=activation,
                    use_rope=use_rope,
                    update=update,
                    step_conditioned_write=step_conditioned_write,
                    adaptive_communication=adaptive_communication,
                )
                for window in base.C0_WINDOWS
            ]
        )
        self.norm = _norm(norm_kind, dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.apply(_init)
        for stage in self.stages:
            stage.reset_special_parameters()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        t = ids.shape[1]
        if t > base.CONTEXT_LENGTH:
            raise ValueError("context exceeds protocol")
        x = self.token(ids)
        if self.pos is not None:
            pos = torch.arange(t, device=ids.device)
            x = x + self.pos(pos)[None]
        for stage in self.stages:
            x = stage(x)
        return self.head(self.norm(x))


def build_phase_b(name: str, vocab_size: int = base.VOCAB_SIZE) -> nn.Module:
    common = dict(dim=C1_DIM, heads=C1_HEADS, vocab_size=vocab_size, update="single-gate")
    if name == M1_NAME:
        return PhaseBLM(**common, ffn_dim=C1_FFN, norm_kind="rmsnorm")
    if name == M2_NAME:
        return PhaseBLM(**common, ffn_dim=M_SWIGLU_FFN, activation="swiglu")
    if name == M3_NAME:
        return PhaseBLM(**common, ffn_dim=C1_FFN, use_rope=True)
    if name == M4_NAME:
        return PhaseBLM(
            **common,
            ffn_dim=M_SWIGLU_FFN,
            norm_kind="rmsnorm",
            activation="swiglu",
            use_rope=True,
        )
    if name == N1_NAME:
        return PhaseBLM(
            dim=C1_DIM,
            heads=C1_HEADS,
            ffn_dim=N1_FFN,
            vocab_size=vocab_size,
            update="residual-only",
        )
    if name == N2_NAME:
        return PhaseBLM(
            dim=C1_DIM,
            heads=C1_HEADS,
            ffn_dim=C1_FFN,
            vocab_size=vocab_size,
            update="dual-gate",
        )
    if name == N3_NAME:
        return PhaseBLM(
            **common,
            ffn_dim=C1_FFN,
            step_conditioned_write=True,
        )
    if name == N4_NAME:
        return PhaseBLM(
            **common,
            ffn_dim=C1_FFN,
            adaptive_communication=True,
        )
    raise ValueError(name)


def phase_b_config(name: str) -> dict[str, object]:
    base_config: dict[str, object] = {
        "name": name,
        "parent": "C1-residual-gated-update",
        "search_phase": "B-two-track",
        "windows": list(base.C0_WINDOWS),
        "iterations": [4, 4, 4],
    }
    if name == M1_NAME:
        return {**base_config, "track": "modernization", "factor": "normalization", "normalization": "RMSNorm", "activation": "GELU", "position": "learned-absolute"}
    if name == M2_NAME:
        return {**base_config, "track": "modernization", "factor": "activation", "normalization": "LayerNorm", "activation": "SwiGLU", "swiglu_hidden": M_SWIGLU_FFN, "position": "learned-absolute"}
    if name == M3_NAME:
        return {**base_config, "track": "modernization", "factor": "position", "normalization": "LayerNorm", "activation": "GELU", "position": "RoPE", "learned_position_embedding": False}
    if name == M4_NAME:
        return {**base_config, "track": "modernization", "factor": "combined-modernization", "normalization": "RMSNorm", "activation": "SwiGLU", "swiglu_hidden": M_SWIGLU_FFN, "position": "RoPE", "learned_position_embedding": False}
    if name == N1_NAME:
        return {**base_config, "track": "native", "factor": "gate-ablation", "update": "fixed-scaled-residual", "residual_alpha": N1_ALPHA, "ffn_dim": N1_FFN}
    if name == N2_NAME:
        return {**base_config, "track": "native", "factor": "dual-gate-aru", "update": "retain*state+write*proposal", "gate_rank": N2_GATE_RANK, "retain_bias": N2_RETAIN_BIAS, "write_bias": N2_WRITE_BIAS}
    if name == N3_NAME:
        return {**base_config, "track": "native", "factor": "recurrent-step-conditioned-write", "update": "C1+per-step-gate-bias", "step_gate_bias_initial": 0.0}
    if name == N4_NAME:
        return {**base_config, "track": "native", "factor": "adaptive-communication", "communication": "state-conditioned-soft-gate", "communication_bias": N4_COMM_BIAS, "realized_compute_reduction_claim": False}
    raise ValueError(name)


def parameter_summary() -> dict[str, object]:
    t1 = base.count_parameters(base.build_model(base.T1_NAME))
    out: dict[str, object] = {"T1_anchor": t1, "tolerance": PARAMETER_TOLERANCE}
    for name in PHASE_B_NAMES:
        params = base.count_parameters(build_phase_b(name))
        out[name] = {
            "parameters": params,
            "over_t1": params / t1,
            "relative_error": abs(params / t1 - 1.0),
        }
    return out


def _attn_cost(t: int, d: int, window: int) -> int:
    return 8 * t * d * d + 4 * d * base._local_pairs(t, window)


def estimate_forward_flops(name: str, t: int = base.TRAIN_SEQUENCE_LENGTH) -> int:
    d = C1_DIM
    if name in (M1_NAME, M3_NAME, N3_NAME):
        ff_cost = 4 * t * d * C1_FFN
        update_cost = 4 * t * d * d
    elif name in (M2_NAME, M4_NAME):
        ff_cost = 6 * t * d * M_SWIGLU_FFN
        update_cost = 4 * t * d * d
    elif name == N1_NAME:
        ff_cost = 4 * t * d * N1_FFN
        update_cost = 0
    elif name == N2_NAME:
        ff_cost = 4 * t * d * C1_FFN
        # Low-rank dual gate: (2d -> d/2 -> 2d).
        update_cost = 4 * t * d * N2_GATE_RANK + 4 * t * N2_GATE_RANK * d
    elif name == N4_NAME:
        ff_cost = 4 * t * d * C1_FFN
        update_cost = 4 * t * d * d + 2 * t * d
    else:
        raise ValueError(name)
    total = 0
    for window in base.C0_WINDOWS:
        per_step = _attn_cost(t, d, window) + ff_cost + update_cost
        total += 4 * per_step
    total += 2 * t * d * base.VOCAB_SIZE
    return int(total)


def estimate_flops(name: str, tokens: int, t: int = base.TRAIN_SEQUENCE_LENGTH) -> dict[str, float]:
    per_token = estimate_forward_flops(name, t) / t
    return {
        "inference_forward_flops_per_token": per_token,
        "train_flops_estimate": per_token * tokens * 3.0,
        "convention": "matmul-heavy; multiply-add=2; training≈3x forward; RoPE/norm/elementwise gates omitted; N4 soft communication does not skip attention",
    }


def validation_log_token_auc(rows: list[dict[str, object]]) -> dict[str, float]:
    if len(rows) < 2:
        return {"auc": float("nan"), "mean_nll": float("nan")}
    x = np.log(np.asarray([float(row["consumed_tokens"]) for row in rows]))
    y = np.asarray([float(row["validation_nll"]) for row in rows])
    auc = float(np.trapezoid(y, x))
    width = float(x[-1] - x[0])
    return {"auc": auc, "mean_nll": auc / width}
