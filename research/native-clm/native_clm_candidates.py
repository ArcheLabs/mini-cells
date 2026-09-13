from __future__ import annotations

import numpy as np
import torch
from torch import nn

import native_clm_runtime as base

VERIFIED_DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
base.DATASET_REVISION = VERIFIED_DATASET_REVISION

C1_NAME = "C1-residual-gated-update"
C2_NAME = "C2-shared-wide-phase"
C3_NAME = "C3-progressive-receptive-field"
C4_NAME = "C4-sparse-communication"
CANDIDATE_NAMES = (C1_NAME, C2_NAME, C3_NAME, C4_NAME)
PARAMETER_TOLERANCE = 0.01

C1_DIM, C1_HEADS, C1_FFN = 188, 4, 752
C1_GATE_BIAS = -2.0
C2_DIM, C2_HEADS, C2_FFN = 270, 6, 1080
C2_WINDOWS = (8, 8, 8, 8, 32, 32, 32, 32, 128, 128, 128, 128)
C2_PHASES = (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2)
C3_SCHEDULE = ((8, 16, 32, 64), (16, 32, 64, 128), (32, 64, 128, 128))
C4_COMM = (True, False, False, True)


def _init(m: nn.Module) -> None:
    base.HistoricalScaledCLM._init_weights(m)


def _set_carry(gru: nn.GRUCell, bias: float = 2.0) -> None:
    h = gru.hidden_size
    with torch.no_grad():
        gru.bias_ih[h:2*h].fill_(bias / 2)
        gru.bias_hh[h:2*h].fill_(bias / 2)


class SearchStage(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, windows: tuple[int, ...], *, gated=False, comm=None):
        super().__init__()
        self.d, self.windows, self.comm, self.gated = d, windows, comm, gated
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = base.LocalCausalSelfAttention(d, heads, windows[0])
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.step = nn.Parameter(torch.empty(len(windows), d))
        nn.init.normal_(self.step, 0.0, 0.02)
        if gated:
            self.gate = nn.Linear(2*d, d)
            self.gru = None
        else:
            self.gru = nn.GRUCell(d, d)
            self.gate = None
            _set_carry(self.gru)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        b, t, d = state.shape
        for i, window in enumerate(self.windows):
            conditioned = state + self.step[i].view(1, 1, d)
            do_comm = self.comm is None or self.comm[i]
            if do_comm:
                self.attn.window = window
                a = self.attn(self.n1(conditioned))
            else:
                a = torch.zeros_like(state)
            f = self.ff(self.n2(state + a))
            proposal = a + f
            if self.gated:
                g = torch.sigmoid(self.gate(torch.cat((state, proposal), dim=-1)))
                state = state + g * proposal
            else:
                state = self.gru(proposal.reshape(b*t, d), state.reshape(b*t, d)).view(b, t, d)
        return state


class StageLM(nn.Module):
    def __init__(self, d: int, stages: list[nn.Module], vocab_size: int):
        super().__init__()
        self.d = d
        self.token = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(base.CONTEXT_LENGTH, d)
        self.stages = nn.ModuleList(stages)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.apply(_init)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        t = ids.shape[1]
        if t > base.CONTEXT_LENGTH:
            raise ValueError("context exceeds protocol")
        pos = torch.arange(t, device=ids.device)
        x = self.token(ids) + self.pos(pos)[None]
        for stage in self.stages:
            x = stage(x)
        return self.head(self.norm(x))


class SharedWideLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        d = C2_DIM
        self.token = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(base.CONTEXT_LENGTH, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = base.LocalCausalSelfAttention(d, C2_HEADS, 128)
        self.ff = nn.Sequential(nn.Linear(d, C2_FFN), nn.GELU(), nn.Linear(C2_FFN, d))
        self.gru = nn.GRUCell(d, d)
        self.step = nn.Parameter(torch.empty(12, d))
        self.phase_scale = nn.Parameter(torch.zeros(3, d))
        self.phase_bias = nn.Parameter(torch.zeros(3, d))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.apply(_init)
        nn.init.normal_(self.step, 0.0, 0.02)
        nn.init.zeros_(self.phase_scale)
        nn.init.zeros_(self.phase_bias)
        _set_carry(self.gru)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        t = ids.shape[1]
        if t > base.CONTEXT_LENGTH:
            raise ValueError("context exceeds protocol")
        d = C2_DIM
        pos = torch.arange(t, device=ids.device)
        x = self.token(ids) + self.pos(pos)[None]
        b = x.shape[0]
        for i, (window, phase) in enumerate(zip(C2_WINDOWS, C2_PHASES)):
            self.attn.window = window
            z = x * (1 + self.phase_scale[phase].view(1,1,d)) + self.phase_bias[phase].view(1,1,d) + self.step[i].view(1,1,d)
            a = self.attn(self.n1(z))
            f = self.ff(self.n2(x + a))
            x = self.gru((a + f).reshape(b*t, d), x.reshape(b*t, d)).view(b,t,d)
        return self.head(self.norm(x))


def build_candidate(name: str, vocab_size: int = base.VOCAB_SIZE) -> nn.Module:
    if name == C1_NAME:
        stages = [SearchStage(C1_DIM, C1_HEADS, C1_FFN, (w,)*4, gated=True) for w in base.C0_WINDOWS]
        model = StageLM(C1_DIM, stages, vocab_size)
        for s in model.stages:
            nn.init.zeros_(s.gate.weight); nn.init.constant_(s.gate.bias, C1_GATE_BIAS)
        return model
    if name == C2_NAME:
        return SharedWideLM(vocab_size)
    if name == C3_NAME:
        return StageLM(base.C0_DIM, [SearchStage(base.C0_DIM, base.C0_HEADS, base.C0_FFN, s) for s in C3_SCHEDULE], vocab_size)
    if name == C4_NAME:
        return StageLM(base.C0_DIM, [SearchStage(base.C0_DIM, base.C0_HEADS, base.C0_FFN, (w,)*4, comm=C4_COMM) for w in base.C0_WINDOWS], vocab_size)
    raise ValueError(name)


def candidate_config(name: str) -> dict[str, object]:
    if name == C1_NAME:
        return {"factor":"state_update","dim":C1_DIM,"heads":C1_HEADS,"ffn":C1_FFN,"update":"identity+sigmoid_write","gate_bias":C1_GATE_BIAS}
    if name == C2_NAME:
        return {"factor":"cross_phase_sharing","dim":C2_DIM,"heads":C2_HEADS,"ffn":C2_FFN,"shared_steps":12,"phase_film":True,"windows":list(C2_WINDOWS)}
    if name == C3_NAME:
        return {"factor":"receptive_field_schedule","schedule":[list(x) for x in C3_SCHEDULE],"update":"C0_GRU"}
    if name == C4_NAME:
        return {"factor":"communication_frequency","communicate":list(C4_COMM),"windows":list(base.C0_WINDOWS),"update":"C0_GRU"}
    raise ValueError(name)


def parameter_summary() -> dict[str, object]:
    t1 = base.count_parameters(base.build_model(base.T1_NAME))
    out = {"T1_anchor": t1, "tolerance": PARAMETER_TOLERANCE}
    for name in CANDIDATE_NAMES:
        p = base.count_parameters(build_candidate(name))
        out[name] = {"parameters":p,"over_t1":p/t1,"relative_error":abs(p/t1-1)}
    return out


def _attn_cost(t: int, d: int, window: int) -> int:
    return 8*t*d*d + 4*d*base._local_pairs(t, window)


def estimate_forward_flops(name: str, t: int = base.TRAIN_SEQUENCE_LENGTH) -> int:
    if name == C1_NAME:
        d, ff = C1_DIM, C1_FFN
        total = sum(4*(_attn_cost(t,d,w)+4*t*d*ff+4*t*d*d) for w in base.C0_WINDOWS)
    elif name == C2_NAME:
        d, ff = C2_DIM, C2_FFN
        total = sum(_attn_cost(t,d,w)+4*t*d*ff+12*t*d*d for w in C2_WINDOWS)
    elif name == C3_NAME:
        d, ff = base.C0_DIM, base.C0_FFN
        total = sum(_attn_cost(t,d,w)+4*t*d*ff+12*t*d*d for s in C3_SCHEDULE for w in s)
    elif name == C4_NAME:
        d, ff = base.C0_DIM, base.C0_FFN
        total = sum((_attn_cost(t,d,w) if c else 0)+4*t*d*ff+12*t*d*d for w in base.C0_WINDOWS for c in C4_COMM)
    else:
        raise ValueError(name)
    return int(total + 2*t*d*base.VOCAB_SIZE)


def estimate_flops(name: str, tokens: int, t: int = base.TRAIN_SEQUENCE_LENGTH) -> dict[str,float]:
    per_token = estimate_forward_flops(name,t)/t
    return {"inference_forward_flops_per_token":per_token,"train_flops_estimate":per_token*tokens*3.0,"convention":"matmul-heavy; multiply-add=2; training≈3x forward"}


def validation_log_token_auc(rows: list[dict[str, object]]) -> dict[str,float]:
    if len(rows) < 2:
        return {"auc":float("nan"),"mean_nll":float("nan")}
    x = np.log(np.array([float(r["consumed_tokens"]) for r in rows]))
    y = np.array([float(r["validation_nll"]) for r in rows])
    auc = float(np.trapezoid(y,x)); width=float(x[-1]-x[0])
    return {"auc":auc,"mean_nll":auc/width}
