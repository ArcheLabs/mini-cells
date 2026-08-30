from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .language_models import TextNCALM
from .upcycled_cellular_textnca import UpcyclingConfig, UpcycledCellularTextNCA, convert_textnca_to_upcycled


BUNDLE_FORMAT = "minicells.clm-0.1.bundle.v1"
MODEL_FORMAT = "minicells.clm-0.1.model.v1"
CLM01_MODEL_SHA256 = "87d36c408ae3873ffd567ebf17050661b42ddae2c8d5d1bab84b2c27c3c7e7a0"
CLM01_RELEASE_VALIDATION_PPL = 17.968933276012226


def verify_clm01_model_sha256(path: str | Path, expected: str = CLM01_MODEL_SHA256) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != expected:
        raise RuntimeError(f"CLM-0.1 checkpoint hash mismatch: expected {expected}, observed {observed}")
    return observed


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: tuple[int, ...]
    routing_usage: tuple[tuple[float, ...], ...] = ()


def locked_textnca_config() -> dict[str, Any]:
    return {
        "vocab_size": 2048,
        "max_context": 128,
        "dim": 128,
        "heads": 4,
        "ffn_dim": 512,
        "windows": [8, 32, 128],
        "iterations": [4, 4, 4],
        "carry_bias": 2.0,
        "rms_norm": False,
        "tie_embeddings": True,
        "stage_supervision": False,
    }


def build_release_model(*, num_experts: int = 4, router_scale: float = 4.0) -> UpcycledCellularTextNCA:
    cfg = locked_textnca_config()
    source = TextNCALM(
        vocab_size=int(cfg["vocab_size"]),
        max_context=int(cfg["max_context"]),
        dim=int(cfg["dim"]),
        heads=int(cfg["heads"]),
        ffn_dim=int(cfg["ffn_dim"]),
        windows=tuple(cfg["windows"]),
        iterations=tuple(cfg["iterations"]),
        carry_bias=float(cfg["carry_bias"]),
        rms_norm=bool(cfg["rms_norm"]),
        tie_embeddings=bool(cfg["tie_embeddings"]),
        stage_supervision=bool(cfg["stage_supervision"]),
    )
    return convert_textnca_to_upcycled(
        source,
        config=UpcyclingConfig(
            num_experts=num_experts,
            top_k=1,
            router_scale=router_scale,
            execution_backend="sparse_dispatch",
        ),
    )


def save_release_bundle(
    model: UpcycledCellularTextNCA,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    *,
    provenance: dict[str, Any],
    metrics: dict[str, Any],
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_source = Path(tokenizer_path)
    tokenizer_target = output / "tokenizer.json"
    tokenizer_target.write_bytes(tokenizer_source.read_bytes())
    checkpoint = {
        "format": MODEL_FORMAT,
        "model_state": model.state_dict(),
        "num_experts": model.config.num_experts,
        "router_scale": model.config.router_scale,
        "provenance": provenance,
        "metrics": metrics,
    }
    torch.save(checkpoint, output / "model.pt")
    config = {
        "format": BUNDLE_FORMAT,
        "release": "MiniCells CLM-0.1 Research Preview",
        "architecture": locked_textnca_config(),
        "routing": {
            "num_experts": model.config.num_experts,
            "top_k": 1,
            "router": "strictly-pointwise-local-cosine-prototype",
            "execution_backend": "sparse_dispatch",
        },
        "provenance": provenance,
        "metrics": metrics,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return output


class CLM:
    """Minimal public inference wrapper for MiniCells CLM-0.1."""

    def __init__(self, model: UpcycledCellularTextNCA, tokenizer: Any, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(cls, path: str | Path, *, device: str | torch.device = "cpu") -> "CLM":
        root = Path(path)
        config = json.loads((root / "config.json").read_text())
        if config.get("format") != BUNDLE_FORMAT:
            raise RuntimeError(f"unsupported CLM bundle format: {config.get('format')!r}")
        checkpoint = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
        if checkpoint.get("format") != MODEL_FORMAT:
            raise RuntimeError(f"unsupported CLM model format: {checkpoint.get('format')!r}")
        model = build_release_model(
            num_experts=int(checkpoint["num_experts"]),
            router_scale=float(checkpoint["router_scale"]),
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.set_execution_backend("sparse_dispatch")
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("CLM.from_pretrained requires the 'tokenizers' package") from exc
        tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        return cls(model, tokenizer, torch.device(device))

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 48,
        temperature: float = 0.8,
        top_k: int = 40,
        seed: int | None = None,
        return_routing: bool = False,
    ) -> str | GenerationResult:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        encoded = self.tokenizer.encode(prompt)
        ids = list(encoded.ids)
        if not ids:
            raise ValueError("prompt produced no tokens")
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        routing: list[tuple[float, ...]] = []
        for _ in range(max_new_tokens):
            context = ids[-self.model.max_context :]
            input_ids = torch.tensor([context], dtype=torch.long, device=self.device)
            output, stats = self.model(input_ids, return_stats=True)
            logits = output.logits[0, -1].float()
            if temperature == 0:
                next_id = int(logits.argmax())
            else:
                logits = logits / temperature
                if top_k > 0 and top_k < logits.numel():
                    values, indices = logits.topk(top_k)
                    probs = values.softmax(-1)
                    picked = torch.multinomial(probs, 1, generator=generator)
                    next_id = int(indices[picked])
                else:
                    next_id = int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
            ids.append(next_id)
            if return_routing:
                routing.append(tuple(float(v) for v in stats.program_usage.detach().cpu()))
        text = self.tokenizer.decode(ids)
        result = GenerationResult(text=text, token_ids=tuple(ids), routing_usage=tuple(routing))
        return result if return_routing else result.text

    @torch.no_grad()
    def score(self, token_ids: torch.Tensor) -> float:
        if token_ids.ndim != 2 or token_ids.shape[1] < 2:
            raise ValueError("token_ids must have shape [batch, length>=2]")
        token_ids = token_ids.to(self.device)
        output = self.model(token_ids[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            output.logits.flatten(0, 1), token_ids[:, 1:].reshape(-1)
        )
        return math.exp(float(loss))
