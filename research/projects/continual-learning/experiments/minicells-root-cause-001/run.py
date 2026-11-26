"""Run the independent single-T4 Base Runtime diagnostic."""
import argparse
from contextlib import nullcontext
from hashlib import sha256
from itertools import accumulate
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "apps/mom-native-cells")]

import torch

from fixtures import prompts
from minicells import MiniCells
from minicells_native.generation import generate
from minicells_native.model_assets import MODEL_13B, REVISION_13B
from minicells_native.query import run_query
from run_t4_1b_capacity import environment, write

FORMAT = "minicells.root-cause.001.phase-a.v1"
PARITY = dict(atol=0.0625, rtol=0.005)
SPLIT = dict(atol=0.25, rtol=0.02)
OUTPUT_TOKENS = 32
MIN_MATCHES = 4


def encode(tokenizer, text, bos):
    plain = tokenizer.encode(text, add_special_tokens=False)
    encoded = tokenizer.encode(text, add_special_tokens=bos)
    expected = [tokenizer.bos_token_id, *plain] if bos else plain
    if not plain or encoded != expected or any(t in tokenizer.all_special_ids for t in plain):
        raise ValueError("Expected exactly one optional initial BOS and no other special token")
    return encoded


def known_completion(answer, target):
    word = re.match(r"[A-Za-z]+|[0-9]+", answer.lstrip())
    return word is not None and word.group().casefold() == target.casefold()


def differences(reference, observed, tolerance):
    valid = reference.shape == observed.shape and bool(
        torch.isfinite(reference).all() and torch.isfinite(observed).all())
    if not valid:
        return dict(pass_=False, reference_shape=list(reference.shape), observed_shape=list(observed.shape))
    reference, observed = reference.float(), observed.float()
    error = (reference - observed).abs()
    inside = error <= tolerance["atol"] + tolerance["rtol"] * reference.abs()
    return dict(pass_=bool(inside.all()), shape=list(observed.shape), max_abs=float(error.max()),
                mean_abs=float(error.mean()), within_fraction=float(inside.float().mean()),
                top1_agreement=float((reference.argmax(-1) == observed.argmax(-1)).float().mean()))


def native_step(backbone):
    cache = backbone.stream({})
    return lambda ids: backbone.step(ids, [cache], [1.0]), lambda: cache


def reference_step(backbone):
    cache = None

    def step(ids):
        nonlocal cache
        tokens = torch.tensor([ids], dtype=torch.long, device=backbone.device)
        result = backbone.model.model(input_ids=tokens, past_key_values=cache, use_cache=True,
                                      output_hidden_states=False, return_dict=True)
        cache = result.past_key_values
        return result.last_hidden_state

    return step, lambda: cache


def split_input(tokens, schedule):
    if schedule == "whole":
        return [tokens]
    if schedule == "prefill_decode":
        return [tokens[:-4], *[[t] for t in tokens[-4:]]]
    return [[t] for t in tokens]


def measure_stream(backbone, tokens, schedule, factory):
    step, cache = factory(backbone)
    logits, cursors = [], []
    for chunk in split_input(tokens, schedule):
        logits.append(backbone.model.lm_head(step(chunk)).float())
        cursors.append(cache().get_seq_length())
    return torch.cat(logits, dim=1), cursors, cache()


def cache_fields(cache):
    result = {}
    def collect(key, value):
        if torch.is_tensor(value):
            result[key] = value
        elif isinstance(value, (tuple, list)):
            for idx, child in enumerate(value):
                collect(f"{key}.{idx}", child)
        elif value is not None:
            raise TypeError(f"Unexpected native cache value for {key}")
    for layer, state in enumerate(cache.to_legacy_cache()):
        for name in ("recurrent_state", "conv_state", "attn_state", "ffn_state"):
            collect(f"{layer}.{name}", state[name])
    return result


def compare_caches(reference, native):
    ref, sdk = cache_fields(reference), cache_fields(native)
    mismatched, max_error = [], 0.0
    for name in sorted(ref.keys() & sdk.keys()):
        a, b = ref[name], sdk[name]
        if a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            mismatched.append(name)
            continue
        max_error = max(max_error, float((a.float() - b.float()).abs().max()))
        if not torch.allclose(a, b, **PARITY):
            mismatched.append(name)
    return dict(pass_=ref.keys() == sdk.keys() and not mismatched, tensor_count=len(ref),
                missing=sorted(ref.keys() - sdk.keys()), unexpected=sorted(sdk.keys() - ref.keys()),
                mismatched=mismatched, max_abs=max_error)


class GenerationProbe:
    stop_on_eos = True

    def __init__(self, eos):
        self.eos = eos
        self.reason = "max_new_tokens"

    def phase(self, *_):
        return nullcontext()

    def logits(self, logits, token):
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite generation logits")
        if token == self.eos:
            self.reason = "eos"

    def cache(self, _):
        pass


def probe_generation(mc, row, ids):
    probe = GenerationProbe(mc._backbone.tokenizer.eos_token_id)
    with mc._operation():
        result = run_query(mc, row["text"], [], OUTPUT_TOKENS, 0, token_ids=ids, probe=probe)
    return dict(text=result.text, token_ids=probe.tokens, stop_reason=probe.reason,
                seconds=result.latency, known_completion=known_completion(result.text, row["target"]))


def official_wrapper(backbone, tokens):
    """Report the official no-label CausalLM entry point independently."""
    try:
        result = backbone.model(input_ids=torch.tensor([tokens], device=backbone.device),
                                use_cache=True, return_dict=True)
    except TypeError as exc:
        return dict(status="FAIL", error_type=type(exc).__name__, reason=str(exc))
    return dict(status="PASS" if torch.isfinite(result.logits).all() else "FAIL")


@torch.inference_mode()
def phase_a(mc, out):
    backbone = mc._backbone
    tok = backbone.tokenizer
    config = backbone.model.config
    token_policy = dict(bos=tok.bos_token_id, eos=tok.eos_token_id, model_bos=config.bos_token_id,
                        model_eos=config.eos_token_id, independent_generation_bos=[False, True],
                        current_sdk_write_bos=False, current_sdk_query_bos=False)
    if (tok.bos_token_id, tok.eos_token_id) != (config.bos_token_id, config.eos_token_id):
        raise ValueError("Tokenizer and model special token IDs disagree")
    rows = []
    wrapper = None
    for fixture in prompts():
        for bos in (False, True):
            ids = encode(tok, fixture["text"], bos)
            if not 5 <= len(ids) <= 64:
                raise ValueError("Fixed fixture must exercise the T4 recurrent path")
            if wrapper is None:
                wrapper = official_wrapper(backbone, ids)
            row = dict(**fixture, bos=bos, prompt_token_ids=ids, comparisons={})
            full = None
            schedules = ("whole", "prefill_decode", "tokenwise") if fixture["id"] == "france" else ("whole",)
            for schedule in schedules:
                ref, ref_positions, ref_cache = measure_stream(backbone, ids, schedule, reference_step)
                got, positions, cache = measure_stream(backbone, ids, schedule, native_step)
                row["comparisons"][schedule] = dict(
                    logits=differences(ref, got, PARITY),
                    cache=compare_caches(ref_cache, cache), positions=positions,
                    reference_positions=ref_positions, position_parity=positions == ref_positions,
                    logical_positions=list(accumulate(map(len, split_input(ids, schedule)))),
                    consumed_tokens=len(ids))
                if schedule == "whole":
                    full = ref
                else:
                    row["comparisons"][schedule]["upstream_vs_whole"] = differences(full, ref, SPLIT)
                del ref_cache, cache
            row["native"] = probe_generation(mc, fixture, ids)
            if fixture["id"] == "france":
                probe = GenerationProbe(tok.eos_token_id)
                step, _ = reference_step(backbone)
                text, generated = generate(backbone, ids, step, OUTPUT_TOKENS, 0, observe=probe.logits)
                row["reference_generation"] = dict(text=text, token_ids=generated, stop_reason=probe.reason,
                    matches_native=(generated == row["native"]["token_ids"] and probe.reason == row["native"]["stop_reason"]))
            rows.append(row)
            with (out / "phase_a_rows.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(f"{fixture['id']} {fixture['style']} bos={bos} complete", flush=True)
    checks = [check for row in rows for check in row["comparisons"].values()]
    counts = {f"{style}/bos={bos}": sum(r["native"]["known_completion"]
                    for r in rows if r["style"] == style and r["bos"] == bos)
                  for style in ("continuation", "fact_completion", "question_answer") for bos in (False, True)}
    generation_parity = all(r["reference_generation"]["matches_native"] for r in rows if "reference_generation" in r)
    gates = dict(tokenizer="PASS", native_reference="PASS" if all(
                    c["logits"]["pass_"] and c["cache"]["pass_"] and c["position_parity"] for c in checks) else "FAIL",
                 upstream_split="PASS" if all(c["upstream_vs_whole"]["pass_"]
                     for c in checks if "upstream_vs_whole" in c) else "FAIL",
                 greedy_reference="PASS" if generation_parity else "FAIL",
                 base_completion="PASS" if counts["continuation/bos=True"] >= MIN_MATCHES else "FAIL")
    return dict(gates=gates, official_wrapper=wrapper, tokenizer=token_policy, counts=counts, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True)
    environmental = environment()
    start = time.monotonic()
    protocol = dict(parity=PARITY, split=SPLIT, max_new_tokens=OUTPUT_TOKENS,
                    minimum_bos_continuations=MIN_MATCHES, use_cache=True)
    result = dict(format=FORMAT, status="INVALID", engineering_status="FAIL", protocol=protocol,
                  environment=environmental, model_revision=REVISION_13B)
    try:
        if (torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != "Tesla T4"
                or torch.cuda.get_device_capability(0) != (7, 5)):
            raise RuntimeError("Expected a single CUDA-visible Tesla T4 at cuda:0")
        torch.cuda.set_device(0)
        torch.set_num_threads(2)
        result["environment"].update(cuda_runtime=torch.version.cuda, torch=str(torch.__version__))
        print("MODEL_LOAD_STARTED", flush=True)
        with MiniCells.open(MODEL_13B, out / "work" / "empty", device="cuda:0", dtype="fp16") as mc:
            result["identity"] = mc.model_info
            print("MODEL_LOAD_COMPLETED", flush=True)
            if mc.cells.list() or mc.model_info["model_revision"] != REVISION_13B:
                raise RuntimeError("Base diagnostic requires the audited revision and an empty registry")
            result.update(phase_a(mc, out))
        status = "PASS" if set(result["gates"].values()) == {"PASS"} else "FAIL"
        result.update(status=status, engineering_status="PASS",
                      diagnosis="Phase A Base Runtime gates evaluated. Phase B remains a separate experiment.")
    except Exception as exc:
        result["diagnosis"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["seconds"] = time.monotonic() - start
        write(out, "phase_a.json", result)
        write(out, "decision.json", {k: result[k] for k in ("format", "status", "engineering_status", "diagnosis")})
        lines = ["# MiniCells Root Cause · Phase A", "", f"**{result['status']}**. {result['diagnosis']}", ""]
        for gate, value in result.get("gates", {}).items():
            lines.append(f"- {gate}: {value}")
        if "counts" in result:
            lines += ["", "## Known completions, out of six", ""]
            lines.extend(f"- {key}: {value}" for key, value in result["counts"].items())
            lines += ["", f"Official CausalLM no-label wrapper: {result['official_wrapper']['status']}.",
                      "", "The auxiliary results include the official wrapper probe, raw outputs, per-schedule cache "
                      "positions, and full logit comparison statistics."]
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write(out, "manifest.json", dict(source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), protocol=protocol,
            artifacts_sha256={file.name: sha256(file.read_bytes()).hexdigest()
                              for file in sorted(out.iterdir()) if file.is_file()}))


if __name__ == "__main__":
    main()
