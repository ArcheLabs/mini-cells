"""Independent T4 inference gate for MINICELLS-T4-1B-CAPACITY-001."""
import argparse
from collections import Counter
from functools import wraps
from hashlib import sha256
import json
from importlib.metadata import version
from pathlib import Path
import subprocess
import time
import traceback
from unittest.mock import patch

from run_t4_1b_capacity import HERE, build_prompts, checked_runtime, environment, read, validate_target, write

FORMAT = "minicells.t4.1b.preflight.v1"
GATES = ("GPU_DETECTION", "CHECKPOINT_GPU_LOAD", "STRICT_LOADING", "FORWARD",
         "FLA_NATIVE_KERNEL", "LOGITS_SANITY", "PREFILL", "GENERATION", "MEMORY_MEASUREMENT")


def approved(receipt, target, revision):
    return (receipt.get("format") == FORMAT and receipt.get("status") == "PASS"
            and receipt.get("target") == target and receipt.get("source_commit") == revision
            and receipt.get("gates") == dict.fromkeys(GATES, "PASS"))


def source_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()


def check_logits(torch, logits, length, vocab):
    if not (logits.shape == (1, length, vocab) and torch.isfinite(logits).all()
            and (logits.amax(-1) > logits.amin(-1)).all()):
        raise ValueError("Expected finite, nonconstant logits for every input token")
    return dict(shape=list(logits.shape), dtype=str(logits.dtype),
                minimum=float(logits.min()), maximum=float(logits.max()))


def cache_bytes(cache, torch):
    def size(value):
        if torch.is_tensor(value):
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite tensor in a native cache")
            return value.numel() * value.element_size()
        if isinstance(value, (list, tuple)):
            return sum(map(size, value))
        if value is None:
            return 0
        raise TypeError(f"Unknown cache field type {type(value)}")

    return {name: sum(size(layer[name]) for layer in cache.to_legacy_cache())
            for name in ("recurrent_state", "conv_state", "attn_state")}


def run(out, config):
    out.mkdir(parents=True, exist_ok=True)
    gates = dict.fromkeys(GATES, "NOT_RUN")
    report = dict(format=FORMAT, status="BLOCKED", gates=gates, environment=environment(),
                  target=config.get("approved_target"), measurements={}, source_commit=None,
                  protocol=dict(device="cuda:0", input_tokens=1024, generated_tokens=128,
                                prefill_chunk_tokens=64, stop_on_eos=False))
    report["target_manifest_sha256"] = sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    started = time.monotonic()
    phase = "GPU_DETECTION"
    meter = None
    try:
        from capacity_meter import Meter
        import torch

        report["source_commit"] = source_commit()
        reason = validate_target(config)
        if reason:
            raise RuntimeError(reason)
        if (torch.cuda.device_count() != 1 or "Tesla T4" not in torch.cuda.get_device_name(0)
                or torch.cuda.get_device_capability(0) != (7, 5)):
            raise RuntimeError("Expected one visible Tesla T4 (sm75) at cuda:0")
        torch.cuda.set_device(0)
        torch.set_num_threads(2)
        report["environment"].update(cuda_runtime=torch.version.cuda, torch=str(torch.__version__),
                                     name=torch.cuda.get_device_name(0), compute_capability=[7, 5])
        gates[phase] = "PASS"

        phase = "MEMORY_MEASUREMENT"
        meter = Meter()
        meter.baseline = meter.sample()
        driver = meter.nvml.nvmlSystemGetDriverVersion()
        report["environment"].update(driver=driver.decode() if isinstance(driver, bytes) else driver,
                                     device_uuid=meter.device_uuid, baseline=meter.baseline)
        stats = report["measurements"]

        phase = "CHECKPOINT_GPU_LOAD"
        print("MODEL_LOAD_STARTED", flush=True)
        stats["load"] = {}
        with checked_runtime(meter, out, config["approved_target"], out / "work" / "preflight-cells",
                             stats["load"], device="cuda:0") as mc:
            backbone = mc._backbone
            model = backbone.model
            target_device = torch.device("cuda:0")
            if any(p.device != target_device for p in model.parameters()):
                raise RuntimeError("Some parameters are outside cuda:0")
            gates[phase] = gates["STRICT_LOADING"] = "PASS"
            report["model"] = read(out, "model_manifest.json")
            report["environment"]["packages"] = {
                name: version(name) for name in ("transformers", "flash-linear-attention", "triton")
            }
            print(f"MODEL_LOAD_COMPLETED params={backbone.parameters} dtype=float16 device=cuda:0", flush=True)

            from minicells_native.generation import generate
            from mom.layers import mom_gated_deltanet

            calls = Counter()
            entrypoint = mom_gated_deltanet.fused_recurrent_gated_delta_rule

            @wraps(entrypoint)
            def traced(*args, **kwargs):
                result = entrypoint(*args, **kwargs)
                calls["forward" if phase == "FORWARD" else phase.lower()] += 1
                return result

            tokens = build_prompts(backbone.tokenizer)["1024"]
            vocab = model.config.vocab_size
            report["prompt_token_ids"] = tokens
            report["kernel"] = dict(name=entrypoint.__name__, module=entrypoint.__module__,
                                    completed_calls=calls)
            with patch.object(mom_gated_deltanet, "fused_recurrent_gated_delta_rule", traced):
                phase = "FORWARD"
                stats["forward"] = {}
                with meter.measure(stats["forward"]), torch.inference_mode():
                    cache = backbone.stream({})
                    hidden = backbone.step(tokens[:16], [cache], [1.0])
                    logits = model.lm_head(hidden).float()
                gates[phase] = "PASS"

                phase = "FLA_NATIVE_KERNEL"
                if (calls["forward"] < backbone.layer_count
                        or not report["kernel"]["module"].startswith("fla.")):
                    raise RuntimeError("No completed upstream FLA recurrent kernel call")
                gates[phase] = "PASS"

                phase = "LOGITS_SANITY"
                report["forward_logits"] = check_logits(torch, logits, 16, vocab)
                gates[phase] = "PASS"
                del cache, hidden, logits

                phase = "PREFILL"
                stats["prefill"] = {}
                cache = backbone.stream({})
                with meter.measure(stats["prefill"]):
                    for start in range(0, len(tokens), 64):
                        hidden = backbone.step(tokens[start:start + 64], [cache], [1.0])
                if calls["prefill"] < 16 * backbone.layer_count:
                    raise RuntimeError("Prefill did not execute the native kernel")
                gates[phase] = "PASS"

                phase = "GENERATION"
                stats["generation"] = {}

                def step(ids):
                    nonlocal hidden
                    if hidden is not None:
                        first, hidden = hidden, None
                        return first
                    return backbone.step(ids, [cache], [1.0])

                def observe(logits, _token):
                    check_logits(torch, logits.unsqueeze(1), 1, vocab)

                with meter.measure(stats["generation"]):
                    text, ids = generate(backbone, tokens[-64:], step, 128, 0,
                                         stop_on_eos=False, observe=observe)
                if (len(ids) != 128 or not text.strip()
                        or calls["generation"] < 127 * backbone.layer_count):
                    raise RuntimeError("Expected 128 native greedy tokens with nonempty decoded text")
                report["generation"] = dict(text=text, token_ids=ids,
                                            tokens_per_second=len(ids) / stats["generation"]["seconds"],
                                            eos_positions=[i for i, token in enumerate(ids)
                                                           if token == backbone.tokenizer.eos_token_id],
                                            final_cache_bytes=cache_bytes(cache, torch))
                gates[phase] = "PASS"

        phase = "MEMORY_MEASUREMENT"
        if any(item["device_sampler_errors"] for item in stats.values()):
            raise RuntimeError("NVML sampling failed during preflight")
        for interval in stats.values():
            for point in ("before", "after"):
                sample = interval[point]
                sample["device_free"] = sample["device_total"] - sample["device_used"]
        gates[phase] = "PASS"
        report["status"] = "PASS"
    except Exception as exc:
        gates[phase] = "FAIL" if phase != "GPU_DETECTION" else "BLOCKED"
        report.update(failed_gate=phase, error_type=type(exc).__name__, reason=str(exc),
                      traceback=traceback.format_exc(), oom="out of memory" in str(exc).lower())
    finally:
        if meter:
            meter.close()
        report["elapsed_seconds"] = time.monotonic() - started
        write(out, "preflight_result.json", report)
        lines = ["# T4 single-GPU preflight", "", f"**{report['status']}**", "",
                 "| Gate | Result |", "| --- | --- |"]
        lines.extend(f"| {name} | {state} |" for name, state in gates.items())
        if report.get("reason"):
            lines.extend(("", report["reason"]))
        lines.extend(("", "See preflight_result.json for device, timings, and memory measurements.", ""))
        (out / "T4_SINGLE_GPU_PREFLIGHT.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"T4_PREFLIGHT={report['status']}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=Path, default=HERE / "capacity_target.json")
    args = parser.parse_args()
    return int(run(args.output, read(args.target.parent, args.target.name))["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
