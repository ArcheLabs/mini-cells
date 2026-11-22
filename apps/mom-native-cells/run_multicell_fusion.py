"""MOM-MULTICELL-FUSION-001 one-shot development experiment."""
import argparse
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time

import torch

from backend import Backbone, sha
from cells import Cells
from compositional_recall import (PAIRS, candidate_tokens, dataset as v2_dataset, encode,
                                  gold_for, preflight as v2_preflight, prompt_for, read, write,
                                  restore_cells, write_cells)
from multicell_fusion import Diagnostics, FusionConfig, FusionGate, FusionReader, LEARNED

TRAIN_SEED = 731
SPLIT_SEEDS = {"train": 732, "validation": 733, "test": 734}
CONFIGS = {
    "legacy_equal": FusionConfig(),
    "hierarchical_equal": FusionConfig("hierarchical_equal"),
    "global_factorized": FusionConfig("global_factorized"),
    **{mode: FusionConfig(mode) for mode in LEARNED},
    "all4_with_prior": FusionConfig("global_all4_gate", router_prior=1),
}
TRAINABLE = {name: cfg for name, cfg in CONFIGS.items() if cfg.mode in LEARNED}
N_COUNTS = (1, 2, 4, 8, 16)
CATEGORIES = ("T1", "T2", "T3", "T4", "T5")


def question(text):
    return f"Question: {text}\nAnswer:"


def task_fixture(split, count):
    rng = random.Random(SPLIT_SEEDS[split])
    offset = {"train": 4000, "validation": 5000, "test": 6000}[split]
    values = iter(rng.sample(range(offset, offset + 500), count * 26))
    episodes = []
    for index in range(count):
        project, relay, other = (f"{prefix}-{next(values)}" for prefix in ("PX", "RX", "PX"))
        code, foil = (f" ZC-{next(values)}" for _ in range(2))
        facts = {
            "A": f"Project {project} is assigned relay key {relay}.",
            "B": f"Relay key {relay} resolves to recovery code {code.strip()}.",
            "direct": f"Project {project} uses recovery code {code.strip()}.",
        }
        for i in range(7):
            facts[f"D{i}"] = (f"Project PX-{next(values)} uses recovery code ZC-{next(values)}. "
                               f"Relay RX-{next(values)} belongs to an unrelated simulation.")
        direct = question(f"What recovery code does Project {project} use?")
        abstain = " Unknown"
        rows = [
            dict(category="T1", prompt=direct, gold=code, candidates=sorted((code, foil)),
                 cells=["direct"], relevant=["direct"]),
            dict(category="T2", prompt=prompt_for(project), gold=code, candidates=sorted((code, foil)),
                 cells=["A", "B"], relevant=["A", "B"]),
            dict(category="T4", prompt=question(f"Which code unlocks recovery for {project}?"),
                 gold=code, candidates=sorted((code, foil)), cells=["direct", "D0"], relevant=["direct"]),
            dict(category="T5", prompt=question(f"What recovery code does Project {other} use? "
                                               "If the information is unavailable, answer Unknown."),
                 gold=abstain, candidates=sorted((code, abstain)), cells=["direct", "D0"], relevant=[]),
        ]
        for n in (1, 2, 4, 8):
            rows.append(dict(category="T3", prompt=direct, gold=code, candidates=sorted((code, foil)),
                             cells=["direct", *(f"D{i}" for i in range(n - 1))], relevant=["direct"]))
        episodes.append(dict(group=f"{split}-{index}", facts=facts, rows=rows))
    return episodes


def validate_fixtures(backbone, assets):
    v2_preflight(backbone.tokenizer, assets["v2_replay"])
    for split in SPLIT_SEEDS:
        for group in assets[split]:
            for row in group["rows"]:
                lengths = []
                for answer in row["candidates"]:
                    prompt, target = candidate_tokens(backbone.tokenizer, row["prompt"], answer)
                    if len(prompt) + len(target) - 1 > 64:
                        raise ValueError(f"Context outside recurrent limit in {group['group']}")
                    lengths.append(len(target))
                if row["category"] != "T5" and len(set(lengths)) != 1:
                    raise ValueError(f"Unequal synthetic answers in {group['group']}")


@torch.inference_mode()
def candidate_score(reader, states, config, gate, prompt, answer, diagnostics=None):
    model = reader.backbone
    context, target = candidate_tokens(model.tokenizer, prompt, answer)
    hidden = reader.step(context + target[:-1], reader.start(states), config, gate, diagnostics)
    logits = model.model.lm_head(hidden[:, len(context) - 1:]).float()
    logp = logits.log_softmax(-1)[0, range(len(target)), target]
    if not torch.isfinite(logp).all():
        raise FloatingPointError("Non-finite answer score")
    return logp.tolist()


@torch.inference_mode()
def generate(reader, states, config, gate, text, limit):
    model = reader.backbone
    tokens = encode(model.tokenizer, text)
    results, cache = [], reader.start(states)
    if not cache:
        cache = reader.start([{}])
        config = FusionConfig("single_cell")
    for _ in range(limit):
        hidden = reader.step(tokens, cache, config, gate)
        token = int(model.model.lm_head(hidden[:, -1]).argmax(-1))
        if token == model.tokenizer.eos_token_id:
            break
        results.append(token)
        tokens = [token]
    return model.tokenizer.decode(results, skip_special_tokens=True).strip()


def evaluate(reader, tensors, row, labels, config, gate, *, generate_answer=False, trace=False):
    states = [tensors[label] for label in labels]
    diag = Diagnostics() if trace and states and config.mode not in ("single_cell", "legacy_equal") else None
    scores = {answer: candidate_score(reader, states, config, gate, row["prompt"], answer,
                                       diag if answer == row["gold"] else None)
              for answer in row["candidates"]}
    means = {answer: sum(values) / len(values) for answer, values in scores.items()}
    # Equal-length candidates use the same ranking under either score. T5 compares
    # an abstention with a code, so mean token scores avoid a length-only preference.
    rank = means if row["category"] == "T5" else {answer: sum(values) for answer, values in scores.items()}
    gold = row["gold"]
    margin = rank[gold] - max(value for answer, value in rank.items() if answer != gold)
    answer = None
    if generate_answer:
        limit = max(map(len, scores.values()))
        answer = generate(reader, states, config, gate, row["prompt"], limit)
    return dict(nll=-means[gold], top1=margin > 0, margin=margin, scores=scores,
                generation=answer, exact_match=answer == gold.strip() if answer is not None else None,
                layers=diag.layers if diag else None)


def fingerprint(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def new_gate(backbone, seed):
    torch.manual_seed(seed)
    return FusionGate().to(backbone.device).eval()


def write_temporary(backbone, root, group):
    store = Cells(root, backbone)
    meta, tensors = write_cells(backbone, store, group)
    return store, meta, tensors


def smoke(backbone, args, restored):
    out = args.output
    reader = FusionReader(backbone)
    group = task_fixture("validation", 1)[0]
    group["facts"] = {key: group["facts"][key] for key in ("A", "B")}
    store = Cells(args.scratch / "smoke-original", backbone)
    if restored:
        meta = read(out / "state-manifest.json")
        tensors = restore_cells(store, Cells(args.scratch / "smoke-imported", backbone), meta)
    else:
        meta, tensors = write_cells(backbone, store, group)
        write(out / "state-manifest.json", meta)
    before = {key: fingerprint(state) for key, state in tensors.items()}
    token_ids = encode(backbone.tokenizer, group["rows"][1]["prompt"])
    gate = new_gate(backbone, TRAIN_SEED)
    gates = {"single_cell": FusionConfig("single_cell"), **CONFIGS,
             "null": FusionConfig("global_topk_gate", null_candidate=True)}
    result, summaries, failures = {}, {}, []
    stage = "preflight-restored" if restored else "preflight"
    fp16_tolerance = {"atol": 0.025, "rtol": 0.01}
    local_weight_atol = 1e-6

    def check(name, actual, expected, tolerance=fp16_tolerance):
        try:
            torch.testing.assert_close(actual, expected, **tolerance)
        except AssertionError as exc:
            failures.append(dict(check=name, detail=str(exc)))

    def progress():
        write(out / f"{stage}-progress.json", dict(
            status="FAIL" if failures else "RUNNING", modes=summaries, failures=failures,
            state_hashes=before, fp16_tolerance=fp16_tolerance,
            fixed_input_weight_atol=local_weight_atol))
    bundle_hashes = {key: sha(store.path(item["cell_id"])) for key, item in meta.items()}

    def run(config, labels, ids=token_ids, trace=None):
        return reader.step(ids, reader.start(tensors[key] for key in labels), config, gate, trace).float()

    for name, config in gates.items():
        labels = ("A",) if name == "single_cell" else ("A", "B")
        trace = Diagnostics(check_permutation=True)
        result[name] = run(config, labels, trace=trace)
        reversed_trace = Diagnostics()
        reverse = run(config, labels[::-1], trace=reversed_trace)
        same = run(config, labels)
        for label, output in (("forward", result[name]), ("reverse", reverse), ("repeat", same)):
            if not torch.isfinite(output).all():
                raise FloatingPointError(f"Nonfinite {label} preflight output in {name}")
        check(f"{name}:permutation", result[name], reverse)
        check(f"{name}:repeat", result[name], same)
        local = {}
        rollout = {}
        topk_changes = 0
        for left, right in zip(trace.layers, reversed_trace.layers, strict=True):
            errors = left.get("fixed_input_permutation_max_abs", {})
            for field, error in errors.items():
                local[field] = max(local.get(field, 0), error)
                if not error <= local_weight_atol:
                    failures.append(dict(check=f"{name}:layer{left['layer']}:{field}",
                                         detail=f"Same-input weight permutation error {error:.8g}"))
            same_topk = left["topk"] == right["topk"]
            topk_changes += not same_topk
            width = 4 if config.mode == "global_all4_gate" else 2
            for field, lanes in (("cell", 1), ("unit", width)):
                if left[field] is None or (field == "unit" and width != 4 and not same_topk):
                    continue
                count = len(labels) * lanes
                a, b = (torch.tensor(entry[field]["mean"]) for entry in (left, right))
                aligned = torch.cat((b[:count].reshape(len(labels), lanes).flip(0).flatten(), b[count:]))
                rollout[field] = max(rollout.get(field, 0), float((a - aligned).abs().max()))
        summaries[name] = dict(permutation_max_abs=float((result[name] - reverse).abs().max()),
                               repeat_max_abs=float((result[name] - same).abs().max()),
                               fixed_input_weight_max_abs=local, rollout_weight_max_abs=rollout,
                               layers_with_topk_changes=topk_changes)
        print(stage, name, summaries[name], flush=True)
        progress()

    for length in (1, len(token_ids)):
        native = run(FusionConfig("single_cell"), ("A",), token_ids[:length])
        experimental = run(FusionConfig("hierarchical_equal"), ("A",), token_ids[:length])
        check(f"single_parity_{length}", native, experimental)
        summaries[f"single_parity_{length}"] = float((native - experimental).abs().max())
    check("factorization", result["hierarchical_equal"], result["global_factorized"])
    summaries["factorization_max_abs"] = float((result["hierarchical_equal"] - result["global_factorized"]).abs().max())
    progress()
    if {key: fingerprint(state) for key, state in tensors.items()} != before:
        raise RuntimeError("Persistent source tensors were modified by a query")
    # Mutate one temporary recurrent state and the position of its stream.
    left, right = reader.start(tensors[k] for k in ("A", "B"))
    right_first = right[0]["recurrent_state"][-1].clone()
    left[0]["recurrent_state"][-1].zero_()
    torch.testing.assert_close(right[0]["recurrent_state"][-1], right_first, rtol=0, atol=0)
    for key, item in meta.items():
        if sha(store.path(item["cell_id"])) != bundle_hashes[key]:
            raise RuntimeError("Persistent bundle changed")
    if failures:
        raise RuntimeError(f"{stage} failed {len(failures)} checks; see {stage}-progress.json")
    write(out / f"{stage}.json",
          dict(status="PASS", modes=summaries, state_hashes=before,
               conv_and_position="transient", pid=os.getpid(),
               fp16_tolerance=fp16_tolerance, fixed_input_weight_atol=local_weight_atol,
               identity=backbone.identity))


def save_training(path, gate, optimizer, step, losses):
    staging = path.with_suffix(".tmp")
    torch.save(dict(gate=gate.state_dict(), optimizer=optimizer.state_dict(), step=step, losses=losses), staging)
    staging.replace(path)


def train_gate(reader, fixtures, path, config, seed):
    gate = new_gate(reader.backbone, seed)
    optimizer = torch.optim.Adam(gate.parameters(), lr=0.001)
    if path.exists():
        checkpoint = torch.load(path, map_location=reader.backbone.device, weights_only=True)
        gate.load_state_dict(checkpoint["gate"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        starting, losses = checkpoint["step"], checkpoint["losses"]
    else:
        starting, losses = 0, []
    params = list(gate.parameters())
    for step, (row, states) in enumerate(fixtures[starting:], start=starting):
        rng = torch.Generator(device=reader.backbone.device).manual_seed(seed + step + 1)
        directions = [torch.randint(0, 2, p.shape, device=p.device, generator=rng).float() * 2 - 1
                      for p in params]
        originals = [p.detach().clone() for p in params]
        epsilon = 0.02
        nll = []
        with torch.no_grad():
            for sign in (1, -1):
                for p, base, direction in zip(params, originals, directions, strict=True):
                    p.copy_(base + sign * epsilon * direction)
                score = candidate_score(reader, states, config, gate, row["prompt"], row["gold"])
                nll.append(-sum(score) / len(score))
            scale = (nll[0] - nll[1]) / (2 * epsilon)
            for p, base, direction in zip(params, originals, directions, strict=True):
                p.copy_(base)
                p.grad = direction * scale
        grad = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        optimizer.step()
        losses.append(dict(step=step + 1, positive_nll=nll[0], negative_nll=nll[1], grad_norm=grad))
        save_training(path, gate, optimizer, step + 1, losses)
    return gate, losses


def ablate(reader, tensors, row, config, gate, score):
    active = row["cells"]
    if len(active) <= 1:
        return {}
    without = {}
    for label in active:
        subset = [key for key in active if key != label]
        reduced = evaluate(reader, tensors, row, subset, config, gate)
        without[label] = reduced["nll"] - score["nll"]
    relevant_nll = None
    if row["relevant"]:
        relevant_nll = evaluate(reader, tensors, row, row["relevant"], config, gate)["nll"]
    units = None
    if row["category"] == "T2" and config.mode in ("global_topk_gate", "global_all4_gate"):
        states = [tensors[label] for label in active]
        units = {}
        for unit in range(4):
            logp = candidate_score(reader, states, config, gate, row["prompt"], row["gold"],
                                   Diagnostics(removed_unit=unit))
            units[str(unit)] = -sum(logp) / len(logp) - score["nll"]
    return dict(cell_contribution=without, unit_contribution=units,
                interference=score["nll"] - relevant_nll if relevant_nll is not None else None,
                complementary_gain=min(without.values()) if row["category"] == "T2" else None)


def evaluate_group(reader, tensors, group, modes, gates, *, full=True):
    rows = []
    for row in group["rows"]:
        row = dict(row)
        labels = row["cells"]
        conditions = {"no_cell": ([], FusionConfig(), None)}
        for label in row["relevant"]:
            conditions[f"single_{label}"] = ([label], FusionConfig("single_cell"), None)
        conditions.update({name: (labels, cfg, gates.get(name)) for name, cfg in modes.items()})
        arms = {}
        for name, (active, cfg, gate) in conditions.items():
            result = evaluate(reader, tensors, row, active, cfg, gate, generate_answer=full,
                              trace=full and len(labels) > 1)
            if full and name in modes:
                result.update(ablate(reader, tensors, row, cfg, gate, result))
            arms[name] = result
        rows.append(dict(group=group["group"], **row, arms=arms))
    return rows


def summarize(rows):
    metrics = {}
    for category in CATEGORIES:
        relevant = [row for row in rows if row["category"] == category]
        record = {}
        for name in sorted({name for row in relevant for name in row["arms"]}):
            values = [row["arms"][name] for row in relevant if name in row["arms"]]
            record[name] = dict(count=len(values), top1=sum(x["top1"] for x in values),
                                exact_match=sum(x["exact_match"] is True for x in values),
                                token_nll=sum(x["nll"] for x in values) / len(values))
        metrics[category] = record
    return metrics


def training_cells(backbone, scratch, group):
    root = scratch / group["group"]
    manifest = root / "cells.json"
    store = Cells(root, backbone)
    if manifest.exists():
        info = read(manifest)
        if info["facts"] != group["facts"]:
            raise ValueError("Training Cell checkpoint fixture changed")
        states = {}
        for label, expected in info["cells"].items():
            meta, state = store.read(expected["cell_id"])
            if meta != expected:
                raise ValueError("Training Cell checkpoint changed")
            states[label] = state
        return states
    meta, states = write_cells(backbone, store, group)
    write(manifest, dict(facts=group["facts"], cells=meta))
    return states


def experiment(backbone, args, assets):
    out = args.output
    reader = FusionReader(backbone)
    weights_before = fingerprint(backbone.model.state_dict())
    if any(p.requires_grad for p in backbone.model.parameters()):
        raise RuntimeError("Backbone must be frozen")
    fixtures = []
    expected = {}
    training_states = {}
    for group in assets["train"]:
        tensors = training_cells(backbone, args.scratch, group)
        # A repeatable schedule covers all five tasks and all distractor counts.
        for row in group["rows"]:
            labels = row["cells"]
            fixtures.append((row, [tensors[key] for key in labels]))
        training_states[group["group"]] = tensors
        expected[group["group"]] = {key: fingerprint(state) for key, state in tensors.items()}
    random.Random(TRAIN_SEED).shuffle(fixtures)
    paths = out / "gate-checkpoints"
    paths.mkdir(exist_ok=True)
    training_id = dict(model=weights_before, source=sha(Path(__file__)),
                       fusion_source=sha(Path(__file__).with_name("multicell_fusion.py")),
                       dataset=sha(out / "datasets.json"), cells=expected,
                       configs={key: asdict(config) for key, config in TRAINABLE.items()})
    identity_path = paths / "training-identity.json"
    if identity_path.exists() and read(identity_path) != training_id:
        raise ValueError("Saved gate checkpoints belong to a different training run")
    write(identity_path, training_id)
    gates, diagnostics = {}, {}
    for name, config in TRAINABLE.items():
        print(f"Train {name}", flush=True)
        gates[name], diagnostics[name] = train_gate(reader, fixtures, paths / f"{name}.pt", config, TRAIN_SEED)
    validation = []
    for group in assets["validation"]:
        with tempfile.TemporaryDirectory(dir=args.scratch) as temp:
            _, _, tensors = write_temporary(backbone, temp, group)
            validation.extend(evaluate_group(reader, tensors, group, TRAINABLE, gates, full=False))
    ranking = sorted(TRAINABLE, key=lambda name: sum(row["arms"][name]["nll"] for row in validation))
    selection_path = paths / "validation-selection.json"
    choice = read(selection_path)["mode"] if selection_path.exists() else ranking[0]
    write(selection_path, {"mode": choice})
    modes = {**CONFIGS, "null": FusionConfig(TRAINABLE[choice].mode,
                                          router_prior=TRAINABLE[choice].router_prior, null_candidate=True)}
    gates["null"], diagnostics["null"] = train_gate(reader, fixtures, paths / "null.pt", modes["null"], TRAIN_SEED)
    write(out / "training.json", dict(selection="validation_nll", selected_for_null=choice,
                                     validation=summarize(validation), traces=diagnostics, modes={
                                         name: asdict(cfg) for name, cfg in modes.items()}))
    actual = {group: {key: fingerprint(state) for key, state in states.items()}
              for group, states in training_states.items()}
    if actual != expected or any(p.grad is not None for p in backbone.model.parameters()):
        raise RuntimeError("Gate training changed frozen Cell state or produced backbone gradients")
    del fixtures, training_states
    test = []
    for group in assets["test"]:
        print(f"Test {group['group']}", flush=True)
        with tempfile.TemporaryDirectory(dir=args.scratch) as temp:
            _, _, tensors = write_temporary(backbone, temp, group)
            before = {key: fingerprint(value) for key, value in tensors.items()}
            test.extend(evaluate_group(reader, tensors, group, modes, gates))
            if before != {key: fingerprint(value) for key, value in tensors.items()}:
                raise RuntimeError("Test query modified stored tensor values")
        write(out / "test-queries.json", test)
    replay = []
    for group in assets["v2_replay"]:
        print(f"Replay v2 {group['group']}", flush=True)
        with tempfile.TemporaryDirectory(dir=args.scratch) as temp:
            _, _, tensors = write_temporary(backbone, temp, group)
            for index, project in enumerate(group["projects"]):
                for pair, labels in PAIRS.items():
                    row = dict(category="T2", prompt=prompt_for(project),
                               candidates=group["candidates"], gold=gold_for(group, index, pair),
                               cells=list(labels), relevant=list(labels))
                    evaluated = evaluate_group(reader, tensors, dict(group=group["group"], rows=[row]),
                                               modes, gates, full=False)[0]
                    replay.append(dict(project=project, pair=pair, **evaluated))
        write(out / "v2-replay.json", replay)
    if weights_before != fingerprint(backbone.model.state_dict()):
        raise RuntimeError("Gate training modified the frozen model")
    reports = {}
    for name in modes:
        by_query = {}
        for row in replay:
            by_query.setdefault((row["group"], row["project"]), []).append(row["arms"][name]["top1"])
        reports[name] = dict(top1=sum(sum(v) for v in by_query.values()), total=len(replay),
                             all_four=sum(all(v) for v in by_query.values()), queries=len(by_query))
    distractor = {str(n): summarize([row for row in test if row["category"] == "T3"
                                      and len(row["cells"]) == n])["T3"] for n in (1, 2, 4, 8)}
    write(out / "quality.json", dict(status="DEVELOPMENT_ONLY", test=summarize(test),
                                    distractor_by_count=distractor, replay=reports,
                                    historical_v2=dict(top1=65, total=128, all_four=0),
                                    frozen_sha256=weights_before, train_cell_hashes=expected))


def benchmark(backbone, args):
    out = args.output
    reader = FusionReader(backbone)
    probe = task_fixture("validation", 1)[0]
    source = Cells(args.scratch / "smoke-imported", backbone)
    meta = read(out / "state-manifest.json")
    original = [source.read(item["cell_id"])[1] for item in meta.values()]
    modes = read(out / "training.json")["modes"]
    tokens = encode(backbone.tokenizer, probe["rows"][1]["prompt"])
    rows = []
    for name, config in modes.items():
        config = FusionConfig(**config)
        gate = None
        if config.mode in LEARNED:
            gate = new_gate(backbone, TRAIN_SEED)
            checkpoint = torch.load(out / "gate-checkpoints" / f"{name}.pt", weights_only=True)
            gate.load_state_dict(checkpoint["gate"])
        for n in N_COUNTS:
            states = [original[i % len(original)] for i in range(n)]
            try:
                for timed in (False, True):
                    diag = Diagnostics(timing=True) if timed else None
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    hidden = reader.step(tokens, reader.start(states), config, gate, diag)
                    torch.cuda.synchronize()
                    total = time.perf_counter() - started
                    if not timed:
                        unsynchronized = total
                read_sec = diag.read_seconds if name != "legacy_equal" else None
                fusion_sec = diag.fusion_seconds if name != "legacy_equal" else None
                rows.append(dict(mode=name, cells=n, latency_s=unsynchronized, instrumented_s=total,
                                 memory_read_s=read_sec, fusion_s=fusion_sec,
                                 backbone_other_s=total-read_sec-fusion_sec if read_sec is not None else None,
                                 peak_vram_bytes=torch.cuda.max_memory_allocated(),
                                 cell_bytes=sum(t.numel() * t.element_size() for state in states for t in state.values()),
                                 tokens_per_second=len(tokens) / unsynchronized, status="COMPLETE"))
                del hidden
            except torch.OutOfMemoryError:
                if n <= 8:
                    raise
                rows.append(dict(mode=name, cells=n, status="HARDWARE_LIMIT"))
                torch.cuda.empty_cache()
            write(out / "benchmark.json", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "snapshot", "tokenizer", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--stage", choices=("preflight", "restore", "experiment", "benchmark"))
    parser.add_argument("--scratch", type=Path)
    args = parser.parse_args()
    if not args.stage:
        args.output.mkdir(parents=True, exist_ok=True)
        scratch = args.output / "working-state"
        scratch.mkdir(exist_ok=True)
        started = time.monotonic()
        stages = {"preflight": "preflight.json", "restore": "preflight-restored.json",
                  "experiment": "quality.json", "benchmark": "benchmark-complete.json"}
        for stage, receipt in stages.items():
            if (args.output / receipt).is_file():
                print(f"Resume {stage}: existing receipt", flush=True)
                continue
            if stage in ("preflight", "restore"):
                path = scratch / ("smoke-original" if stage == "preflight" else "smoke-imported")
                if path.exists():
                    shutil.rmtree(path)
            print(f"Stage {stage}", flush=True)
            subprocess.run([sys.executable, "-u", __file__, *(str(getattr(args, name))
                            for name in ("source", "snapshot", "tokenizer", "output")),
                            "--stage", stage, "--scratch", str(scratch)], check=True)
            if stage == "restore":
                before, after = (read(args.output / f"{stem}.json") for stem in ("preflight", "preflight-restored"))
                if (before["pid"] == after["pid"] or before["identity"] != after["identity"]
                        or before["state_hashes"] != after["state_hashes"]):
                    raise RuntimeError("Fresh process restored a different Cell state or model")
        data = args.output / "datasets.json"
        write(args.output / "decision.json", dict(status="DIAGNOSTIC_COMPLETE",
              diagnosis="Engineering gates passed. Compare quality, factorial effects and cost before selecting a mode.",
              elapsed_seconds=time.monotonic() - started, dataset_sha256=hashlib.sha256(data.read_bytes()).hexdigest()))
        return
    torch.set_num_threads(2)
    backbone = Backbone(args.source, args.snapshot, args.tokenizer, dtype="float16")
    if args.stage in ("preflight", "restore"):
        smoke(backbone, args, restored=args.stage == "restore")
    elif args.stage == "benchmark":
        benchmark(backbone, args)
        write(args.output / "benchmark-complete.json", {"status": "COMPLETE"})
    else:
        assets = {split: task_fixture(split, count) for split, count in
                  (("train", 2), ("validation", 1), ("test", 3))}
        assets["v2_replay"] = v2_dataset(backbone.tokenizer)
        validate_fixtures(backbone, assets)
        write(args.output / "datasets.json", assets)
        experiment(backbone, args, assets)


if __name__ == "__main__":
    main()
