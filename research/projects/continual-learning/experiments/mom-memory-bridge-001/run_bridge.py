"""Bridge continuous native memory and persisted single-Cell state on MoM 1.3B."""
import argparse
from dataclasses import asdict
from hashlib import file_digest, sha256
from importlib.metadata import version
from pathlib import Path
from statistics import mean
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import torch

from bridge_fixtures import CASES, ROUTE_SOURCE, memory_text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "minicells-root-cause-001"))

from run import MODEL_13B, PARITY, REVISION_13B, ROOT, MiniCells, cache_fields, environment, write
from run_a3 import interventions
from run_a31 import expert_padding_policy
from run_a32 import expert_delta_policy

FORMAT = "minicells.mom-memory-bridge.001.v1"
A32_RESULTS = "6db20d54ee066816e2319e71be390fb32a026dbc"
A4_RESULTS = "a0a873ca4ceee4513af09237afd9551be8fa99d8"
CHUNK = 64
ARMS = ("A_empty", "B_live_exact", "C_recurrent_exact", "D_full_exact",
        "E_recurrent_domain", "E_live_domain", "E_full_domain")
CONTRASTS = {
    "context_b_vs_a": ("B_live_exact", "A_empty"),
    "recurrent_c_vs_b": ("C_recurrent_exact", "B_live_exact"),
    "full_d_vs_b": ("D_full_exact", "B_live_exact"),
    "domain_e_vs_c": ("E_recurrent_domain", "C_recurrent_exact"),
    "live_domain_vs_exact": ("E_live_domain", "B_live_exact"),
    "full_domain_vs_exact": ("E_full_domain", "D_full_exact"),
}


def encode_case(tokenizer, case):
    query = tokenizer.encode(case.query, add_special_tokens=False)
    targets = {}
    for label, answer in (("gold", case.gold), ("distractor", case.distractor)):
        whole = tokenizer.encode(f"{case.query} {answer}", add_special_tokens=False)
        if not query or whole[:len(query)] != query or len(whole) == len(query):
            raise ValueError(f"Unstable answer token boundary in case {case.index}, {label}")
        targets[label] = whole[len(query):]
    if targets["gold"] == targets["distractor"]:
        raise ValueError(f"Identical answer token IDs in case {case.index}")
    return query, targets


def advance(backbone, cache, ids):
    for start in range(0, len(ids), CHUNK):
        hidden = backbone.step(ids[start:start + CHUNK], [cache], [1.0])
    return hidden


def score(backbone, cache, query, target):
    ids = query + target[:-1]
    if len(ids) > CHUNK:
        raise ValueError("The frozen bridge fixture requires a single scoring chunk")
    hidden = advance(backbone, cache, ids)
    first = len(query) - 1
    logits = backbone.model.lm_head(hidden[:, first:]).float()
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite answer logits")
    values = logits.log_softmax(-1)[0, list(range(len(target))), target].tolist()
    return dict(token_logp=values, total_logp=sum(values), mean_logp=mean(values))


def evaluate(backbone, new_stream, query, targets):
    candidates, before, after = {}, {}, {}
    for label, target in targets.items():
        cache = new_stream()
        before[label] = cache.get_seq_length()
        candidates[label] = score(backbone, cache, query, target)
        after[label] = cache.get_seq_length()
    gold, distractor = candidates["gold"], candidates["distractor"]
    return dict(candidates=candidates, gold_nll=-gold["mean_logp"],
                margin=gold["mean_logp"] - distractor["mean_logp"],
                total_margin=gold["total_logp"] - distractor["total_logp"],
                query_start_cursors=before, query_end_cursors=after)


def tensor_parity(left, right):
    mismatched = sorted((left.keys() ^ right.keys()) |
                        {name for name in left.keys() & right.keys()
                         if left[name].dtype != right[name].dtype or not torch.equal(left[name], right[name])})
    return dict(exact=not mismatched, tensor_count=len(left), mismatched=mismatched)


def score_parity(live, restored):
    paired = list(zip(
        (value for r in live["candidates"].values() for value in r["token_logp"]),
        (value for r in restored["candidates"].values() for value in r["token_logp"]),
        strict=True))
    return dict(exact=all(a == b for a, b in paired),
                within_tolerance=all(abs(a-b) <= PARITY["atol"] + PARITY["rtol"] * abs(a) for a, b in paired),
                max_abs=max(abs(a-b) for a, b in paired))


def context_arms(mc, text, query, targets, directory, case_index, variant):
    backbone, store = mc._backbone, mc.cells._store
    ids = backbone.tokenizer.encode(text, add_special_tokens=False)

    def live():
        cache = backbone.stream({})
        advance(backbone, cache, ids)
        return cache

    # C uses the same registry write and reload as the SDK.
    cell = mc.cells.create(f"bridge-{case_index}-{variant}")
    cell = mc.cells.write(cell.id, text, chunk_size=CHUNK, resume=False)
    if cell.tokens_written != len(ids):
        raise RuntimeError("SDK write did not consume the matched prefix")
    state = mc.cells._state(cell.id)

    def recurrent():
        return backbone.stream(state)

    # D uses the existing full native stream checkpoint codec.
    checkpoint = directory / "full.stream"
    identity = dict(model=backbone.identity, fixture=case_index, variant=variant,
                    write_sha256=sha256(text.encode()).hexdigest())
    source = live()
    store.save_stream(checkpoint, identity, source, len(ids))

    def restore():
        cache, offset = store.load_stream(checkpoint, identity)
        if offset != len(ids):
            raise RuntimeError("Unexpected full stream write offset")
        return cache

    full = restore()
    full_check = tensor_parity(cache_fields(source), cache_fields(full))
    full_check.update(source_cursor=source.get_seq_length(), restored_cursor=full.get_seq_length(),
                      cursor_exact=source.get_seq_length() == full.get_seq_length() == len(ids))
    recurrent_check = tensor_parity(backbone.persistent(source), state)
    if not full_check["exact"] or not full_check["cursor_exact"]:
        raise RuntimeError("Full-cache round trip changed a field or the cache cursor")
    del source, full

    arms = dict(live=evaluate(backbone, live, query, targets),
                recurrent=evaluate(backbone, recurrent, query, targets),
                full=evaluate(backbone, restore, query, targets))
    if any(value != 0 for value in arms["recurrent"]["query_start_cursors"].values()):
        raise RuntimeError("Expected SDK recurrent-only cursor reset")
    full_check["probability_parity"] = score_parity(arms["live"], arms["full"])
    with checkpoint.open("rb") as stream:
        full_check["checkpoint_sha256"] = file_digest(stream, "sha256").hexdigest()
    bundle = store.path(cell.id)
    _, manifest_state = store.read(cell.id)
    recurrent_check["query_did_not_mutate_bundle"] = tensor_parity(state, manifest_state)["exact"]
    if not recurrent_check["query_did_not_mutate_bundle"]:
        raise RuntimeError("Scoring changed saved recurrent tensors")
    with bundle.open("rb") as stream:
        recurrent_check["bundle_sha256"] = file_digest(stream, "sha256").hexdigest()
    mc.cells.delete(cell.id)
    checkpoint.unlink()
    return arms, dict(write_tokens=ids, full=full_check, recurrent=recurrent_check)


def historical_bos_control(backbone, case, query, targets):
    tokenizer = backbone.tokenizer
    arms = {}
    for name, text in (("empty_wrapper", "Memory:\n"), ("exact_record", memory_text(case))):
        ids = [tokenizer.bos_token_id, *tokenizer.encode(text, add_special_tokens=False)]

        def live():
            cache = backbone.stream({})
            advance(backbone, cache, ids)
            return cache

        arms[name] = dict(evaluate(backbone, live, query, targets), write_tokens=ids)
    return arms


def summarize(rows):
    arms = {name: dict(positive_margins=sum(row["arms"][name]["margin"] > 0 for row in rows),
                       mean_margin=mean(row["arms"][name]["margin"] for row in rows),
                       mean_gold_nll=mean(row["arms"][name]["gold_nll"] for row in rows))
            for name in ARMS}
    paired = {}
    for key, (observed, reference) in CONTRASTS.items():
        margin = [r["arms"][observed]["margin"] - r["arms"][reference]["margin"] for r in rows]
        nll = [r["arms"][observed]["gold_nll"] - r["arms"][reference]["gold_nll"] for r in rows]
        paired[key] = dict(mean_delta_margin=mean(margin), mean_delta_gold_nll=mean(nll),
                           improved_margin=sum(x > 0 for x in margin), lower_gold_nll=sum(x < 0 for x in nll),
                           cases=len(rows))
    controls = dict(
        full_cache_fields_exact=all(r[v]["full"]["exact"] and r[v]["full"]["cursor_exact"]
                                    for r in rows for v in ("exact", "domain")),
        full_cache_score_parity=all(r[v]["full"]["probability_parity"]["within_tolerance"]
                                    for r in rows for v in ("exact", "domain")),
        sdk_recurrent_write_parity=all(r[v]["recurrent"]["exact"] for r in rows for v in ("exact", "domain")))
    historical = [row["historical_bos"] for row in rows]
    bos_control = dict(
        exact_positive_margins=sum(r["exact_record"]["margin"] > 0 for r in historical),
        empty_positive_margins=sum(r["empty_wrapper"]["margin"] > 0 for r in historical),
        mean_delta_margin=mean(r["exact_record"]["margin"] - r["empty_wrapper"]["margin"] for r in historical),
        mean_delta_gold_nll=mean(r["exact_record"]["gold_nll"] - r["empty_wrapper"]["gold_nll"] for r in historical))
    return dict(arms=arms, paired=paired, controls=controls, historical_bos_control=bos_control)


@torch.inference_mode()
def experiment(mc, result, out):
    from fla.modules import convolution
    from mom.layers import mom_gated_deltanet as ops

    backbone = mc._backbone
    # Validate every scoring boundary before writing the first Cell.
    encoded = [(case, *encode_case(backbone.tokenizer, case)) for case in CASES]
    if any(len(query) + len(target) - 1 > CHUNK
           for _, query, targets in encoded for target in targets.values()):
        raise RuntimeError("Expected every answer scoring arm to stay within the recurrent path")
    rows = []
    result["rows"] = rows
    with (interventions(ops, convolution, packing=True, history=True),
          expert_padding_policy(ops, convolution),
          expert_delta_policy(ops, backbone.model.model.layers, apply=True),
          TemporaryDirectory(dir=out / "work") as temp):
        for case, query, targets in encoded:
            exact, exact_trace = context_arms(mc, memory_text(case), query, targets, Path(temp), case.index, "exact")
            domain, domain_trace = context_arms(mc, memory_text(case, domain=True), query, targets,
                                                Path(temp), case.index, "domain")
            empty = evaluate(backbone, lambda: backbone.stream({}), query, targets)
            arms = dict(zip(ARMS, (empty, exact["live"], exact["recurrent"], exact["full"],
                                   domain["recurrent"], domain["live"], domain["full"]), strict=True))
            row = dict(case=asdict(case), question=case.query, query_ids=query, targets=targets,
                       equal_target_lengths=len(targets["gold"]) == len(targets["distractor"]),
                       exact=exact_trace, domain=domain_trace, arms=arms,
                       historical_bos=historical_bos_control(backbone, case, query, targets))
            rows.append(row)
            with (out / "cases.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(f"BRIDGE_CASE_COMPLETED {case.index + 1}/{len(encoded)}", flush=True)
    result["summary"] = summarize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    (out / "work").mkdir(parents=True)
    result = dict(format=FORMAT, status="INVALID", engineering_status="FAIL",
                  environment=environment(), model_revision=REVISION_13B, historical_source=ROUTE_SOURCE,
                  runtime_evidence=dict(a32=A32_RESULTS, a4=A4_RESULTS), write_chunk=CHUNK, bos=False,
                  arms=list(ARMS), score_tolerance=PARITY)
    started = time.monotonic()
    try:
        if (torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != "Tesla T4"
                or torch.cuda.get_device_capability(0) != (7, 5)):
            raise RuntimeError("Expected one visible Tesla T4 at cuda:0")
        torch.cuda.set_device(0)
        torch.set_num_threads(2)
        result["environment"].update(cuda_runtime=torch.version.cuda, torch=str(torch.__version__),
                                     fla=version("flash-linear-attention"), triton=version("triton"))
        with MiniCells.open(MODEL_13B, out / "work" / "registry", device="cuda:0", dtype="fp16",
                            write_chunk_size=CHUNK) as mc:
            result["identity"] = mc.model_info
            if mc.cells.list() or mc.model_info["model_revision"] != REVISION_13B:
                raise RuntimeError("Expected pinned weights and an empty Cell registry")
            experiment(mc, result, out)
        result.update(status="DEVELOPMENT_ONLY", engineering_status="PASS",
                      diagnosis="Matched single-Cell paths measured on the historical development fixture.")
    except Exception as exc:
        result["diagnosis"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        write(out, "bridge.json", result)
        write(out, "decision.json", {key: result[key] for key in ("format", "status", "engineering_status", "diagnosis")})
        lines = ["# MoM Memory Path Bridge", "", result["diagnosis"], "",
                 "| Arm | Positive margins | Mean margin | Mean gold NLL |",
                 "| --- | ---: | ---: | ---: |"]
        for name, arm in result.get("summary", {}).get("arms", {}).items():
            lines.append(f"| {name} | {arm['positive_margins']} | {arm['mean_margin']:.5g} | "
                         f"{arm['mean_gold_nll']:.5g} |")
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write(out, "manifest.json", dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            artifacts_sha256={f.name: sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()}))


if __name__ == "__main__":
    main()
