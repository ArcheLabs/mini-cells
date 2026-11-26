"""Pinned asset and tensor checks for the designated Native MoM 1.3B model."""
from hashlib import file_digest, sha256
import json
from math import prod
from pathlib import Path

MOM = "125038087b71b635c171cec857bd0f21d10955b6"
MODEL_13B = "linear-moe-hub/MoM-Gated-Deltanet-1.3B"
REVISION_13B = "5c8be94196cb1827e820077680ac2ee46ac949bc"
ASSETS_13B = {
    "config.json": "b99c19cdd75493abb6dfe0d5e991ebaac8d5af963b75c0ea2f73a6f496204341",
    "generation_config.json": "dfbf1d6403235b70efa8fc7a6ebf225aeb886b9d89ad09e2178c422cede8abc2",
    "model.safetensors": "08820287a91c2bc10f1fafe8b149dbf027ae51768530027c711dc240f7d53771",
    "special_tokens_map.json": "4452b845ab9ca2afa863f59f1411a959e8a32bb4334e264b0e71d5d17dd4dcb9",
    "tokenizer.json": "a2699839c243202a04a90537576fc719283f638e90fe80feb469888275289575",
    "tokenizer_config.json": "cc3fdc5126e4d2e3a3934bc29779bf811c2eeefb60237017cddfd03acf9a7562",
}
TOKENIZER_FILES_13B = {k: ASSETS_13B[k] for k in (
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json")}
TIED_WEIGHTS = ("lm_head.weight", "model.embeddings.weight")


def sha(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def tokenizer_hash(files):
    # Preserve the original Cell identity representation.
    return sha256(repr(dict(sorted(files.items()))).encode()).hexdigest()


def expected_shapes(config):
    hidden = config["hidden_size"]
    heads, width = config["num_heads"], config["head_dim"]
    key, value = heads * width, heads * width * config["expand_v"]
    intermediate = config["intermediate_size"]
    if intermediate is None:
        intermediate = 256 * ((int(hidden * config["hidden_ratio"] * 2 / 3) + 255) // 256)
    shapes = dict.fromkeys(TIED_WEIGHTS, (config["vocab_size"], hidden))
    shapes["model.norm.weight"] = (hidden,)
    attention = {
        "A_log": (heads,), "D": (heads,), "dt_bias": (heads,),
        "gate.weight": (config["num_memories"], hidden),
        "q_proj.weight": (key, hidden), "g_proj.weight": (value, hidden),
        "o_proj.weight": (hidden, value), "o_norm.weight": (value // heads,),
        "shared_a.weight": (heads, hidden), "shared_b.weight": (heads, hidden),
        "shared_k.weight": (key, hidden), "shared_v.weight": (value, hidden),
        **{f"{kind}_conv1d.weight": (dim, 1, config["conv_size"])
           for kind, dim in (("q", key), ("k", key), ("v", value))},
        **{f"{kind}_proj.{bank}.weight": (dim, hidden)
           for bank in range(config["num_memories"])
           for kind, dim in (("a", heads), ("b", heads), ("k", key), ("v", value))},
    }
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        shapes.update({prefix + "attn." + name: shape for name, shape in attention.items()})
        shapes.update({prefix + "attn_norm.weight": (hidden,), prefix + "mlp_norm.weight": (hidden,),
                       prefix + "mlp.gate_proj.weight": (2 * intermediate, hidden),
                       prefix + "mlp.down_proj.weight": (hidden, intermediate)})
    return shapes


def validate_shapes(observed, expected):
    missing, unexpected = sorted(expected.keys() - observed.keys()), sorted(observed.keys() - expected.keys())
    mismatches = {k: [observed[k], expected[k]] for k in observed.keys() & expected.keys()
                  if observed[k] != expected[k]}
    if missing or unexpected or mismatches:
        raise ValueError(f"Strict tensor validation failed: {missing=}, {unexpected=}, {mismatches=}")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate Safetensors entry: {key}")
        result[key] = value
    return result


def weight_header(stream):
    length = int.from_bytes(stream.read(8), "little")
    if not 0 < length <= 100_000_000:
        raise ValueError("Invalid Safetensors header length")
    header = json.loads(stream.read(length), object_pairs_hook=_unique_pairs)
    header.pop("__metadata__", None)
    return 8 + length, header


def inspect_weights(path, config):
    with path.open("rb") as stream:
        payload_offset, header = weight_header(stream)
        shapes = {name: tuple(entry["shape"]) for name, entry in header.items()}
        validate_shapes(shapes, expected_shapes(config))
        position = 0
        for name, entry in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
            start, end = entry["data_offsets"]
            if entry["dtype"] != "BF16" or start != position or end - start != prod(shapes[name]) * 2:
                raise ValueError(f"Invalid checkpoint dtype or byte layout for {name}")
            position = end
        if payload_offset + position != path.stat().st_size:
            raise ValueError("Checkpoint payload size mismatch")

        def tensor_hash(name):
            start, end = header[name]["data_offsets"]
            stream.seek(payload_offset + start)
            digest = sha256()
            while start < end:
                block = stream.read(min(end - start, 8 * 1024 * 1024))
                if not block:
                    raise ValueError("Truncated tied weight")
                digest.update(block)
                start += len(block)
            return digest.hexdigest()

        aliases = {name: tensor_hash(name) for name in TIED_WEIGHTS}
        if len(set(aliases.values())) != 1:
            raise ValueError("Checkpoint has unequal copies of its tied embedding and LM head")
    stored = sum(map(prod, shapes.values()))
    return dict(tensor_count=len(shapes), stored_parameter_count=stored,
                parameter_count=stored - prod(shapes[TIED_WEIGHTS[0]]),
                tensor_dtype_summary={"BF16": {"tensors": len(shapes), "elements": stored}},
                tied_weight_sha256=aliases)


def audit_checkpoint(snapshot, tokenizer_snapshot):
    snapshot, tokenizer_snapshot = Path(snapshot), Path(tokenizer_snapshot)
    if snapshot.name != REVISION_13B or tokenizer_snapshot.name != REVISION_13B:
        raise ValueError("Use the pinned 1.3B model and tokenizer snapshot directories")
    for name, expected in ASSETS_13B.items():
        root = tokenizer_snapshot if name in TOKENIZER_FILES_13B else snapshot
        if sha(root / name) != expected:
            raise ValueError(f"Checkpoint asset SHA-256 mismatch: {name}")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    receipt = inspect_weights(snapshot / "model.safetensors", config)
    return dict(
        status="PASS", gates=dict(MODEL_IDENTITY="PASS", MODEL_REVISION_PINNED="PASS",
                                  CONFIG_AUDIT="PASS", CHECKPOINT_INTEGRITY="PASS",
                                  PARAMETER_COUNT_VERIFIED="PASS"),
        model_id=MODEL_13B, model_revision=REVISION_13B, mom_revision=MOM,
        architecture=config["model_type"], runtime_dtype="float16", source_dtype="bfloat16",
        config=config, config_hash=ASSETS_13B["config.json"], tokenizer_revision=REVISION_13B,
        tokenizer_hash=tokenizer_hash(TOKENIZER_FILES_13B), tokenizer_files=TOKENIZER_FILES_13B,
        generation_config_hash=ASSETS_13B["generation_config.json"], shard_names=["model.safetensors"],
        shard_sizes={"model.safetensors": (snapshot / "model.safetensors").stat().st_size},
        shard_sha256_expected={"model.safetensors": ASSETS_13B["model.safetensors"]},
        shard_sha256_observed={"model.safetensors": ASSETS_13B["model.safetensors"]}, **receipt)
