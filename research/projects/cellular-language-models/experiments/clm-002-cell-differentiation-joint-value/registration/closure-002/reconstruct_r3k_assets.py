"""Match reconstructed research arrays/windows to the historical published asset manifest."""

import argparse
import ast
import builtins
import hashlib
import json
import sys
import tempfile
import types
from pathlib import Path

from recover_role_membership import digest


def reconstruct(source, output):
    import numpy as np
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    base = Path(__file__).resolve().parent
    experiments = base.parents[2]
    sys.path.insert(0, str(experiments / "006-path-first-clm/implementation"))
    sys.path.insert(
        0,
        str(
            experiments
            / "006-path-first-clm/extension-parameter-depth/exploration-fast-001/implementation"
        ),
    )
    from exp006.data import encode_stories

    # Execute only the original pure sampler function; importing its package would
    # import historical model modules unrelated to this no-model data audit.
    sampler_path = (
        experiments
        / "006-path-first-clm/extension-parameter-depth/exploration-fast-001/implementation/fast001/data.py"
    )
    tree = ast.parse(sampler_path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "grouped_windows"
    )
    compiled = compile(ast.Module(body=[function], type_ignores=[]), str(sampler_path), "exec")
    sampler_code = next(
        code
        for code in compiled.co_consts
        if isinstance(code, types.CodeType) and code.co_name == "grouped_windows"
    )
    grouped_windows = types.FunctionType(sampler_code, {"np": np, "__builtins__": vars(builtins)})

    lineage = experiments / "clm-001-qfirst-integrated"
    identity_path = base / "source-evidence/r3k_EXTRACTED_ASSET_IDENTITY.json"
    extracted = json.loads(identity_path.read_text())
    expected = extracted["data_identity"]
    bindings = json.loads((base / "RECOVERY_BINDINGS.json").read_text())
    if (
        digest(source) != expected["source_sha256"]
        or digest(source) != bindings["source_shard_sha256"]
    ):
        raise RuntimeError("FROZEN_SOURCE_HASH_MISMATCH")
    c = json.loads((lineage / "r3k/config/frozen.json").read_text())
    if (
        hashlib.sha256(json.dumps(c, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        != expected["config_sha256"]
    ):
        raise RuntimeError("HISTORICAL_RESEARCH_CONFIG_CHANGED")
    tokenizer_path = (
        lineage / "r3m/registration/artifact-recovery-001/independent-review-002/tokenizer.json"
    )
    if digest(tokenizer_path) != expected["tokenizer_sha256"]:
        raise RuntimeError("ORIGINAL_TOKENIZER_CHANGED")
    boundary_path = base / "source-evidence/r3k_DATA_BOUNDARY_PROOF.json"
    if digest(boundary_path) != expected["files"]["DATA_BOUNDARY_PROOF.json"]:
        raise RuntimeError("HISTORICAL_BOUNDARY_CHANGED")
    boundary = json.loads(boundary_path.read_text())
    texts = {}
    index = 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=4096, columns=["text"]):
        for text in batch.column(0).to_pylist():
            if 8192 <= index < 8960:
                texts[index] = text
            index += 1
        if index >= 8960:
            break
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    windows = {}
    role_summary = {}
    with tempfile.TemporaryDirectory(prefix="clm002-r3k-reconstructed-") as temporary:
        temporary = Path(temporary)
        for role in ("predictor_probe", "value_holdout"):
            rows = boundary["roles"][role]["accepted"]
            ids = [row["story_index"] for row in rows]
            stories = [texts[i] for i in ids]
            if any(
                hashlib.sha256(text.encode()).hexdigest() != row["text_sha256"]
                for text, row in zip(stories, rows)
            ):
                raise RuntimeError("RAW_STORY_HASH_CHANGED")
            encoded, starts, groups, hashes = encode_stories(stories, tokenizer, 64)
            np.save(temporary / (role + ".npy"), encoded)
            windows[role] = {
                "starts": starts,
                "groups": groups,
                "story_offsets": np.cumsum(
                    [0] + [len(tokenizer.encode(t).ids) + 1 for t in stories]
                ).tolist(),
                "raw_story_ids": ids,
                "story_hashes": hashes,
                "evaluation": grouped_windows(
                    starts,
                    groups,
                    c["predictor_windows" if role == "predictor_probe" else "value_windows"],
                    c["evaluation_seed"],
                ),
            }
            role_summary[role] = {
                "stories": len(stories),
                "tokens": len(encoded),
                "eligible_windows": len(starts),
                "fixed_evaluation_windows": len(windows[role]["evaluation"]["starts"]),
            }
        (temporary / "windows.json").write_text(
            json.dumps(windows, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        matched = {path.name: digest(path) for path in temporary.iterdir()}
        if any(expected["files"][name] != value for name, value in matched.items()):
            raise RuntimeError("RECONSTRUCTED_RESEARCH_ASSET_OR_WINDOW_HASH_MISMATCH")
    result = {
        "status": "EXACT_R3K_RESEARCH_ARRAYS_AND_WINDOWS_RECONSTRUCTED",
        "original_manifest": expected,
        "matching_file_sha256": matched,
        "roles": role_summary,
        "remote_evidence_commit": extracted["remote_commit"],
        "published_full_file_sha256": extracted["original_sha256"],
        "published_json_pointer": "/data_identity",
        "extracted_manifest_sha256": digest(identity_path),
        "auditor_sha256": digest(Path(__file__)),
        "encoder_sha256": digest(experiments / "006-path-first-clm/implementation/exp006/data.py"),
        "window_sampler_sha256": digest(
            experiments
            / "006-path-first-clm/extension-parameter-depth/exploration-fast-001/implementation/fast001/data.py"
        ),
        "numpy_version": np.__version__,
        "private_arrays_retained": False,
        "model_loaded": False,
        "optimizer_updates": 0,
        "gpu_used": False,
        "scientific_registration": "BLOCKED",
        "r3l_arrays_recovered": False,
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "roles": role_summary}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reconstruct(args.source_parquet, args.output)
