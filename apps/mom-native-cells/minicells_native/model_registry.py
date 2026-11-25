"""Pinned model and tokenizer pairs supported by the native runtime."""
from dataclasses import dataclass
from pathlib import Path

from .model_assets import ASSETS_13B, MODEL_13B, REVISION_13B

MODEL = "linear-moe-hub/MoM-Gated-Deltanet-340M"
REVISION = "f4081422ea2cf01b1b5d646cac05cad62a5d5438"
TOKENIZER = "fla-hub/gla-1.3B-100B"
TOKENIZER_REV = "46b15820a4df269e99aed9d709e017677c15d24b"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    revision: str
    tokenizer: str
    tokenizer_revision: str
    hidden_size: int
    checkpoint_files: tuple[str, ...]
    tokenizer_in_snapshot: bool = False


MODELS = {
    MODEL: ModelSpec(MODEL, REVISION, TOKENIZER, TOKENIZER_REV, 1024,
                     ("config.json", "model.safetensors")),
    MODEL_13B: ModelSpec(MODEL_13B, REVISION_13B, MODEL_13B, REVISION_13B, 2048,
                         tuple(ASSETS_13B), tokenizer_in_snapshot=True),
}
MODEL_BY_REVISION = {spec.revision: spec for spec in MODELS.values()}


def snapshot_model(snapshot):
    path = Path(snapshot)
    spec = MODEL_BY_REVISION.get(path.name)
    if spec is None or not path.is_dir():
        raise ValueError("Use a registered MoM model ID or its pinned snapshot directory.")
    return spec


def load_tokenizer(spec, snapshot):
    if spec.tokenizer_in_snapshot:
        # The published fast tokenizer contains its normalization and BOS template.
        # AutoTokenizer's Llama wrapper requests a missing SentencePiece model.
        import json
        from transformers import PreTrainedTokenizerFast

        special = json.loads((snapshot / "special_tokens_map.json").read_text(encoding="utf-8"))
        return PreTrainedTokenizerFast(
            tokenizer_file=str(snapshot / "tokenizer.json"),
            **{name: value["content"] if isinstance(value, dict) else value
               for name, value in special.items()},
            clean_up_tokenization_spaces=False,
        )
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
