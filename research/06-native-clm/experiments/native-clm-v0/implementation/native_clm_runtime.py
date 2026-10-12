from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


DATASET_ID = "roneneldan/TinyStories"
DATASET_REVISION = "f54c09fd23315a6f9c86f9e14f9f4f06615b22e3"
SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
VOCAB_SIZE = 2048
CONTEXT_LENGTH = 128
TRAIN_SEQUENCE_LENGTH = 125
BATCH_SIZE = 8
TOKENS_PER_STEP = BATCH_SIZE * TRAIN_SEQUENCE_LENGTH
BASE_LR = 3e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
TOKENIZER_STORIES = 20_000
VALIDATION_BATCHES = 48

C0_NAME = "C0-historical-scaled"
T1_NAME = "T1-modern-transformer"
MODEL_NAMES = (C0_NAME, T1_NAME)

# C0 is a scaled reconstruction of Experiment 007's TextNCA architecture:
# 3 independent stages; stage-local parameters shared across 4 recurrent steps;
# Local causal attention + GELU FFN + GRUCell; step embedding; LayerNorm;
# carry/update bias +2; windows 8/32/128.
C0_DIM = 168
C0_HEADS = 8
C0_FFN = 672
C0_WINDOWS = (8, 32, 128)
C0_ITERATIONS = (4, 4, 4)
C0_CARRY_BIAS = 2.0

# Modern small decoder baseline.
T1_DIM = 192
T1_LAYERS = 4
T1_HEADS = 6
T1_KV_HEADS = 2
T1_FFN = 480


@dataclass(frozen=True)
class Profile:
    name: str
    target_tokens: int
    train_stream_tokens: int
    validation_stream_tokens: int
    checkpoints: tuple[int, ...]
    warmup_steps: int


PROFILES: dict[str, Profile] = {
    "smoke": Profile("smoke", 100_000, 200_000, 50_000, (50_000, 100_000), 20),
    "dev": Profile("dev", 1_000_000, 1_200_000, 250_000, (250_000, 500_000, 750_000, 1_000_000), 100),
    "baseline": Profile(
        "baseline",
        10_000_000,
        12_000_000,
        1_000_000,
        (1_000_000, 2_500_000, 5_000_000, 7_500_000, 10_000_000),
        1_000,
    ),
}


@dataclass(frozen=True)
class Corpus:
    train_path: Path
    validation_path: Path
    tokenizer_path: Path
    manifest: dict[str, object]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hf():
    try:
        from datasets import load_dataset
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:
        raise RuntimeError("Install the repo LM extras: pip install -e '.[lm]'") from exc
    return load_dataset, Tokenizer, ByteLevelDecoder, BPE, ByteLevel, BpeTrainer


def iter_tinystories(split: str, *, token: str | None = None, max_stories: int | None = None) -> Iterator[str]:
    load_dataset, *_ = _require_hf()
    kwargs: dict[str, object] = {
        "path": DATASET_ID,
        "split": split,
        "revision": DATASET_REVISION,
        "streaming": True,
    }
    if token:
        kwargs["token"] = token
    dataset = load_dataset(**kwargs)
    yielded = 0
    for row in dataset:
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        yield text.strip()
        yielded += 1
        if max_stories is not None and yielded >= max_stories:
            break


def train_tokenizer(path: Path, *, token: str | None = None) -> object:
    _, Tokenizer, ByteLevelDecoder, BPE, ByteLevel, BpeTrainer = _require_hf()
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        iter_tinystories("train", token=token, max_stories=TOKENIZER_STORIES),
        trainer=trainer,
        length=TOKENIZER_STORIES,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return tokenizer


def load_tokenizer(path: Path) -> object:
    _, Tokenizer, *_ = _require_hf()
    return Tokenizer.from_file(str(path))


def _write_token_stream(
    tokenizer: object,
    *,
    split: str,
    output_path: Path,
    target_tokens: int,
    token: str | None = None,
) -> int:
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise RuntimeError("tokenizer does not contain <eos>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=(target_tokens,))
    cursor = 0
    stories = 0
    try:
        for text in iter_tinystories(split, token=token):
            ids = tokenizer.encode(text).ids
            if not ids:
                continue
            ids.append(eos_id)
            take = min(len(ids), target_tokens - cursor)
            values[cursor: cursor + take] = np.asarray(ids[:take], dtype=np.uint16)
            cursor += take
            stories += 1
            if cursor >= target_tokens:
                break
        if cursor != target_tokens:
            raise RuntimeError(f"{split} ended at {cursor:,}; required {target_tokens:,} tokens")
        values.flush()
    finally:
        del values
    return stories


def prepare_corpus(cache_root: Path, profile: Profile, *, hf_token: str | None = None) -> Corpus:
    cache = cache_root / f"tinystories-{VOCAB_SIZE}-{profile.name}-{DATASET_REVISION[:8]}"
    cache.mkdir(parents=True, exist_ok=True)
    tokenizer_path = cache / "tokenizer.json"
    train_path = cache / "train.u16"
    validation_path = cache / "validation.u16"
    manifest_path = cache / "manifest.json"

    expected = {
        "format": "minicells.native-clm-corpus.v1",
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "vocab_size_requested": VOCAB_SIZE,
        "tokenizer_training_stories": TOKENIZER_STORIES,
        "train_stream_tokens": profile.train_stream_tokens,
        "validation_stream_tokens": profile.validation_stream_tokens,
        "dtype": "uint16",
    }
    if all(path.exists() for path in (tokenizer_path, train_path, validation_path, manifest_path)):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shape_ok = (
            train_path.stat().st_size == profile.train_stream_tokens * 2
            and validation_path.stat().st_size == profile.validation_stream_tokens * 2
        )
        if (
            shape_ok
            and all(manifest.get(k) == v for k, v in expected.items())
            and sha256_file(tokenizer_path) == manifest.get("tokenizer_sha256")
            and sha256_file(train_path) == manifest.get("train_raw_sha256")
            and sha256_file(validation_path) == manifest.get("validation_raw_sha256")
        ):
            return Corpus(train_path, validation_path, tokenizer_path, manifest)

    for path in (tokenizer_path, train_path, validation_path, manifest_path):
        path.unlink(missing_ok=True)

    tokenizer = train_tokenizer(tokenizer_path, token=hf_token)
    if tokenizer.get_vocab_size() > np.iinfo(np.uint16).max:
        raise RuntimeError("uint16 token cache requires vocab <= 65535")
    train_stories = _write_token_stream(
        tokenizer,
        split="train",
        output_path=train_path,
        target_tokens=profile.train_stream_tokens,
        token=hf_token,
    )
    validation_stories = _write_token_stream(
        tokenizer,
        split="validation",
        output_path=validation_path,
        target_tokens=profile.validation_stream_tokens,
        token=hf_token,
    )
    manifest = {
        **expected,
        "vocab_size_actual": tokenizer.get_vocab_size(),
        "train_stories_consumed": train_stories,
        "validation_stories_consumed": validation_stories,
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "train_raw_sha256": sha256_file(train_path),
        "validation_raw_sha256": sha256_file(validation_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return Corpus(train_path, validation_path, tokenizer_path, manifest)


def open_memmap(path: Path) -> np.memmap:
    if path.stat().st_size % 2:
        raise RuntimeError(f"invalid uint16 stream: {path}")
    return np.memmap(path, dtype=np.uint16, mode="r")


def memmap_batch(stream: np.memmap, starts: tuple[int, ...], length: int, device: torch.device):
    rows = np.stack(
        [np.asarray(stream[s:s + length + 1], dtype=np.int64) for s in starts],
        axis=0,
    )
    packed = torch.from_numpy(rows).to(device, non_blocking=True)
    return packed[:, :-1], packed[:, 1:]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * scale.to(x.dtype) * self.weight


class LocalCausalSelfAttention(nn.Module):
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
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        qi = torch.arange(t, device=x.device)[:, None]
        ki = torch.arange(t, device=x.device)[None, :]
        allowed = (ki <= qi) & (ki >= qi - self.window + 1)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed[None, None, :, :], dropout_p=0.0, is_causal=False
        )
        return self.out(y.transpose(1, 2).contiguous().view(b, t, self.dim))


class HistoricalNCAStage(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, window: int, iterations: int, carry_bias: float):
        super().__init__()
        self.iterations = iterations
        self.norm_attention = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.attention = LocalCausalSelfAttention(dim, heads, window)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))
        self.step_embedding = nn.Parameter(torch.empty(iterations, dim))
        nn.init.normal_(self.step_embedding, mean=0.0, std=0.02)
        self.gru = nn.GRUCell(dim, dim)
        hidden = self.gru.hidden_size
        with torch.no_grad():
            self.gru.bias_ih[hidden:2 * hidden].fill_(carry_bias / 2.0)
            self.gru.bias_hh[hidden:2 * hidden].fill_(carry_bias / 2.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        b, t, d = state.shape
        for step in range(self.iterations):
            conditioned = state + self.step_embedding[step].view(1, 1, d)
            attn_delta = self.attention(self.norm_attention(conditioned))
            candidate = state + attn_delta
            ffn_delta = self.ffn(self.norm_ffn(candidate))
            proposal = attn_delta + ffn_delta
            state = self.gru(
                proposal.reshape(b * t, d), state.reshape(b * t, d)
            ).view(b, t, d)
        return state


class HistoricalScaledCLM(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, C0_DIM)
        self.position_embedding = nn.Embedding(CONTEXT_LENGTH, C0_DIM)
        self.stages = nn.ModuleList([
            HistoricalNCAStage(C0_DIM, C0_HEADS, C0_FFN, w, n, C0_CARRY_BIAS)
            for w, n in zip(C0_WINDOWS, C0_ITERATIONS)
        ])
        self.final_norm = nn.LayerNorm(C0_DIM)
        self.lm_head = nn.Linear(C0_DIM, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, 0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        t = ids.shape[1]
        if t > CONTEXT_LENGTH:
            raise ValueError("context exceeds protocol")
        pos = torch.arange(t, device=ids.device)
        h = self.token_embedding(ids) + self.position_embedding(pos)[None, :, :]
        for stage in self.stages:
            h = stage(h)
        return self.lm_head(self.final_norm(h))


def _rope(x: torch.Tensor) -> torch.Tensor:
    _, _, t, d = x.shape
    if d % 2:
        even = d - 1
        return torch.cat([_rope(x[..., :even]), x[..., even:]], dim=-1)
    half = d // 2
    inv = 1.0 / (10000 ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    pos = torch.arange(t, device=x.device, dtype=torch.float32)
    ang = torch.einsum("t,d->td", pos, inv)
    sin = ang.sin().to(x.dtype)[None, None]
    cos = ang.cos().to(x.dtype)[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class GQAAttention(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int):
        super().__init__()
        if dim % heads or heads % kv_heads:
            raise ValueError("invalid GQA dimensions")
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        kv_dim = kv_heads * self.head_dim
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, kv_dim, bias=False)
        self.v = nn.Linear(dim, kv_dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q = self.q(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = _rope(q), _rope(k)
        rep = self.heads // self.kv_heads
        k = k.repeat_interleave(rep, 1)
        v = v.repeat_interleave(rep, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.o(y.transpose(1, 2).contiguous().view(b, t, d))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ModernBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = RMSNorm(T1_DIM)
        self.attn = GQAAttention(T1_DIM, T1_HEADS, T1_KV_HEADS)
        self.n2 = RMSNorm(T1_DIM)
        self.ffn = SwiGLU(T1_DIM, T1_FFN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))


class ModernTransformerLM(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, T1_DIM)
        self.blocks = nn.ModuleList([ModernBlock() for _ in range(T1_LAYERS)])
        self.final_norm = RMSNorm(T1_DIM)
        self.lm_head = nn.Linear(T1_DIM, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(HistoricalScaledCLM._init_weights)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.token_embedding(ids)
        for block in self.blocks:
            h = block(h)
        return self.lm_head(self.final_norm(h))


def build_model(name: str, vocab_size: int = VOCAB_SIZE) -> nn.Module:
    if name == C0_NAME:
        return HistoricalScaledCLM(vocab_size)
    if name == T1_NAME:
        return ModernTransformerLM(vocab_size)
    raise ValueError(name)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_config(name: str, vocab_size: int = VOCAB_SIZE) -> dict[str, object]:
    if name == C0_NAME:
        return {
            "name": name, "provenance": "scaled reconstruction of Experiment 007",
            "vocab_size": vocab_size, "context_length": CONTEXT_LENGTH,
            "dim": C0_DIM, "heads": C0_HEADS, "ffn_dim": C0_FFN,
            "windows": list(C0_WINDOWS), "iterations": list(C0_ITERATIONS),
            "normalization": "LayerNorm", "gru_carry_bias": C0_CARRY_BIAS,
            "step_embedding": True, "tied_embeddings": True,
        }
    if name == T1_NAME:
        return {
            "name": name, "vocab_size": vocab_size, "context_length": CONTEXT_LENGTH,
            "dim": T1_DIM, "layers": T1_LAYERS, "heads": T1_HEADS,
            "kv_heads": T1_KV_HEADS, "ffn_dim": T1_FFN,
            "normalization": "RMSNorm", "position": "RoPE",
            "activation": "SwiGLU", "tied_embeddings": True,
        }
    raise ValueError(name)


def parameter_summary(vocab_size: int = VOCAB_SIZE) -> dict[str, object]:
    counts = {name: count_parameters(build_model(name, vocab_size)) for name in MODEL_NAMES}
    ratio = counts[C0_NAME] / counts[T1_NAME]
    return {**counts, "c0_over_t1": ratio, "relative_error": abs(ratio - 1.0)}


def _local_pairs(length: int, window: int) -> int:
    return sum(min(i + 1, window) for i in range(length))


def estimate_forward_flops(name: str, sequence_length: int = CONTEXT_LENGTH, vocab_size: int = VOCAB_SIZE) -> int:
    # Analytical matmul-heavy FLOP estimate; one multiply-add = 2 FLOPs.
    t = sequence_length
    if name == T1_NAME:
        d, ff, h, kvh = T1_DIM, T1_FFN, T1_HEADS, T1_KV_HEADS
        kv_dim = d * kvh // h
        qkvo = 2 * t * (d * d + 2 * d * kv_dim + d * d)
        attn = 4 * d * (t * (t + 1) // 2)
        swiglu = 6 * t * d * ff
        per_layer = qkvo + attn + swiglu
        return int(T1_LAYERS * per_layer + 2 * t * d * vocab_size)
    if name == C0_NAME:
        d, ff = C0_DIM, C0_FFN
        total = 0
        for window, iters in zip(C0_WINDOWS, C0_ITERATIONS):
            pairs = _local_pairs(t, window)
            attention_proj = 8 * t * d * d
            attention_mix = 4 * d * pairs
            ffn = 4 * t * d * ff
            gru = 12 * t * d * d
            total += iters * (attention_proj + attention_mix + ffn + gru)
        return int(total + 2 * t * d * vocab_size)
    raise ValueError(name)


def estimate_flops(name: str, *, sequence_length: int, tokens: int) -> dict[str, float]:
    forward_seq = float(estimate_forward_flops(name, sequence_length))
    forward_per_token = forward_seq / sequence_length
    return {
        "inference_forward_flops_per_token": forward_per_token,
        "train_flops_estimate": forward_per_token * tokens * 3.0,
        "convention": "matmul-heavy analytic estimate; multiply-add=2 FLOPs; training≈3x forward",
    }


def lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if step <= warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def fixed_validation_starts(stream_length: int, *, seed: int) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(seed)
    high = stream_length - CONTEXT_LENGTH - 1
    return tuple(
        tuple(rng.randrange(high) for _ in range(BATCH_SIZE))
        for _ in range(VALIDATION_BATCHES)
    )


@torch.no_grad()
def evaluate(model: nn.Module, validation: np.memmap, starts, device: torch.device, amp_dtype) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    enabled = device.type == "cuda"
    for batch_starts in starts:
        inputs, targets = memmap_batch(validation, batch_starts, CONTEXT_LENGTH, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=enabled):
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum")
        total_loss += float(loss.item())
        total_tokens += int(targets.numel())
    nll = total_loss / total_tokens
    return {"validation_nll": nll, "validation_ppl": math.exp(min(nll, 20.0)), "validation_tokens": total_tokens}


def _atomic_save(payload: object, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def environment_record() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "platform": platform.platform(),
    }


def train_one(
    name: str,
    *,
    corpus: Corpus,
    profile: Profile,
    output_dir: Path,
    seed: int,
    device: torch.device,
    resume: bool = True,
) -> dict[str, object]:
    tokenizer = load_tokenizer(corpus.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size != VOCAB_SIZE:
        raise RuntimeError(f"protocol requires vocab {VOCAB_SIZE}, got {vocab_size}")
    train = open_memmap(corpus.train_path)
    validation = open_memmap(corpus.validation_path)

    schedule_seed = seed
    model_seed = seed + (0 if name == C0_NAME else 1)
    validation_seed = seed + 10_000
    torch.manual_seed(model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(model_seed)
    model = build_model(name, vocab_size).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=BETAS, weight_decay=WEIGHT_DECAY)

    use_amp = device.type == "cuda"
    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    batch_rng = random.Random(schedule_seed)
    total_steps = profile.target_tokens // TOKENS_PER_STEP
    if profile.target_tokens % TOKENS_PER_STEP:
        raise RuntimeError("target token budget must be divisible by TOKENS_PER_STEP")
    validation_starts = fixed_validation_starts(len(validation), seed=validation_seed)
    high = len(train) - TRAIN_SEQUENCE_LENGTH - 1

    model_dir = output_dir / name
    resume_path = model_dir / "resume.pt"
    rows: list[dict[str, object]] = []
    start_step = 0
    elapsed_before = 0.0
    peak_before = 0
    if resume and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location="cpu")
        expected = {
            "format": "minicells.native-clm-resume.v1",
            "model": name,
            "profile": profile.name,
            "seed": seed,
            "corpus_manifest_sha256": hashlib.sha256(
                json.dumps(corpus.manifest, sort_keys=True).encode()
            ).hexdigest(),
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise RuntimeError(f"resume mismatch for {key}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        batch_rng.setstate(checkpoint["batch_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        start_step = int(checkpoint["step"])
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))
        peak_before = int(checkpoint.get("peak_vram_bytes", 0))
        rows = list(checkpoint.get("metrics", []))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    def elapsed() -> float:
        return elapsed_before + time.perf_counter() - started

    def peak() -> int:
        if device.type != "cuda":
            return 0
        return max(peak_before, int(torch.cuda.max_memory_allocated()))

    def save_resume(step: int):
        corpus_hash = hashlib.sha256(json.dumps(corpus.manifest, sort_keys=True).encode()).hexdigest()
        payload = {
            "format": "minicells.native-clm-resume.v1",
            "model": name,
            "profile": profile.name,
            "seed": seed,
            "corpus_manifest_sha256": corpus_hash,
            "step": step,
            "consumed_tokens": step * TOKENS_PER_STEP,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "batch_rng_state": batch_rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "metrics": rows,
            "elapsed_seconds": elapsed(),
            "peak_vram_bytes": peak(),
        }
        _atomic_save(payload, resume_path)

    checkpoint_set = set(profile.checkpoints)
    for step in range(start_step + 1, total_steps + 1):
        model.train()
        starts = tuple(batch_rng.randrange(high) for _ in range(BATCH_SIZE))
        lr = BASE_LR * lr_multiplier(step, total_steps, profile.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs, targets = memmap_batch(train, starts, TRAIN_SEQUENCE_LENGTH, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP).item())
        scaler.step(optimizer)
        scaler.update()

        consumed = step * TOKENS_PER_STEP
        if consumed in checkpoint_set:
            val = evaluate(model, validation, validation_starts, device, amp_dtype)
            e = elapsed()
            row = {
                "model": name, "step": step, "consumed_tokens": consumed,
                "train_loss": float(loss.detach().item()), "learning_rate": lr,
                "grad_norm": grad_norm, "parameters": params,
                "elapsed_seconds": e, "tokens_per_second": consumed / e,
                "peak_vram_bytes": peak(), **val,
            }
            rows.append(row)
            print(
                f"{name:24s} tokens={consumed:>10,d} train={row['train_loss']:.4f} "
                f"val_ppl={row['validation_ppl']:.3f} tok/s={row['tokens_per_second']:.0f}",
                flush=True,
            )
            save_resume(step)

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = model_dir / "checkpoints.csv"
    if rows:
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "format": "minicells.native-clm-worker.v1",
        "model": name,
        "profile": profile.name,
        "seed": seed,
        "parameters": params,
        "model_config": model_config(name, vocab_size),
        "consumed_tokens": profile.target_tokens,
        "final": rows[-1] if rows else None,
        "flops": estimate_flops(name, sequence_length=TRAIN_SEQUENCE_LENGTH, tokens=profile.target_tokens),
        "elapsed_seconds": elapsed(),
        "peak_vram_bytes": peak(),
        "environment": environment_record(),
        "corpus_manifest": corpus.manifest,
    }
    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
