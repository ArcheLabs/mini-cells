"""MOM-MULTICELL-RESIDUAL-001 diagnostics on frozen MoM."""
import argparse
from contextlib import nullcontext
import gzip
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import torch

from backend import Backbone, LAYERS, sha
from cells import Cells
from compositional_recall import encode, read, write, write_cells, restore_cells
from gdr_diagnostics import FIELDS, OUTPUT_TOLERANCE, STATE_TOLERANCE, ResidualTrace, parity_report, reference
from multicell_fusion import FusionConfig, FusionReader
from residual_metrics import group_bootstrap, metrics, rank_signal, ranking
from run_multicell_fusion import evaluate, fingerprint, question

LABELS = ("A", "B", "C", "D", "random", "empty")
CONFIG = FusionConfig("hierarchical_equal")
SEEDS = (731, 1907, 3001)
GROUPS_PER_SEED = 8
EXPANDED_PER_SEED = 32
PRIMARY = "routed_relative_residual_norm"
SUBSYSTEMS = ("routed", "shared", "combined")
CHANNELS = ("raw_residual_norm", "relative_residual_norm", "relative_delta_norm",
            "prediction_cosine", "read_output_norm")
SWAP = dict(zip(LABELS, ("B", "D", "A", "C", "random", "empty"), strict=True))
SCORE_TOLERANCE = dict(atol=0.005, rtol=0.01)


def dataset(seed, count):
    rng = random.Random(seed)
    offset = {117: 10000, 731: 30000, 1907: 50000, 3001: 70000}[seed]
    identifiers = iter(rng.sample(range(offset, offset + 20000), count * 12))
    rows = []
    for index in range(count):
        ids = tuple(next(identifiers) for _ in range(12))
        p, other, missing = (f"PX-{i}" for i in ids[:3])
        key, other_key = (f"RX-{i}" for i in ids[3:5])
        code, foil, other_code = (f"ZC-{i}" for i in ids[5:8])
        roles = rng.sample(LABELS[:4], 4)
        left, right, similar, near = roles
        facts = {
            left: f"Project {p} is assigned relay key {key}.",
            right: f"Relay key {key} resolves to recovery code {code}.",
            similar: f"Relay key {other_key} resolves to recovery code {other_code}.",
            near: f"Project {p} belongs to the network of Project {other}.",
            "random": f"Container CN-{ids[8]} has an inventory label IV-{ids[9]}.",
        }
        specs = (
            ("T1", f"What relay key is assigned to Project {p}?", [left], key, other_key),
            ("T2", f"Which restoration identifier can I get from relay {key}?", [right], code, foil),
            ("T3", f"Which recovery code does relay key {other_key} resolve to?", [similar], other_code, code),
            ("T4", f"Which relay key is assigned to Project {missing}?", [], None, None),
            ("T5", f"What recovery code is assigned to Project {p}?", [left, right], code, foil),
        )
        group = f"{seed}-{index}"
        queries = [
            dict(id=f"{group}-{kind}", group=group, seed=seed, category=kind,
                 prompt=question(text), relevant=gold, gold=" " + answer if answer else None,
                 candidates=sorted((" " + answer, " " + distractor)) if answer else [])
            for kind, text, gold, answer, distractor in specs
        ]
        rows.append(dict(group=group, seed=seed, facts=facts, queries=queries, identifiers=ids))
    return rows


def configure_data(backbone):
    from compositional_recall import candidate_tokens

    smoke = dataset(117, 4)
    full = [row for seed in SEEDS for row in dataset(seed, EXPANDED_PER_SEED)]
    pilot = [row for row in full if int(row["group"].split("-")[1]) < GROUPS_PER_SEED]
    seen = set()
    for group in [*smoke, *full]:
        if seen.intersection(group["identifiers"]):
            raise ValueError("Cross-group identifier collision")
        seen.update(group["identifiers"])
        for query in group["queries"]:
            prompt = query["prompt"]
            lengths = []
            for answer in query["candidates"]:
                tokens, target = candidate_tokens(backbone.tokenizer, prompt, answer)
                lengths.append(len(target))
                if len(tokens) + len(target) - 1 > 64:
                    raise ValueError("Candidate exceeds recurrent limit")
            if len(set(lengths)) > 1 or len(encode(backbone.tokenizer, prompt)) > 64:
                raise ValueError("Incompatible prompt or candidate tokens")
    return dict(smoke=smoke, pilot=pilot, expansion=[row for row in full if row not in pilot])


def score_trace(trace, tokens, labels, query, trace_writer=None):
    index = {field: n for n, field in enumerate(FIELDS)}
    heads = trace.banks[0][-1].shape[1]
    shape = (len(labels), LAYERS, 5, len(tokens))
    values = torch.zeros((*shape, heads, len(FIELDS)))
    present = torch.zeros(shape, dtype=torch.bool)
    for slot, layer, bank, positions, result in trace.banks:
        if not torch.isfinite(result).all():
            raise FloatingPointError(f"Non-finite residual in {query['id']} at {layer}/{bank}")
        values[slot, layer, bank, positions] = result
        present[slot, layer, bank, positions] = True
        if trace_writer is not None:
            entry = dict(query_id=query["id"], cell=labels[slot], layer=layer, bank=bank,
                         token_index=positions.tolist(), values=result.tolist())
            trace_writer(entry)
    if not torch.all(present[:, :, :4].sum(2) == 2) or not torch.all(present[:, :, 4]):
        raise RuntimeError(f"Expected exactly two native banks and one shared bank per token for {query['id']}")
    for item in trace.routers:
        for token, selected in enumerate(item["topk_indices"]):
            actual = present[:, item["layer"], :4, token]
            expected = torch.zeros(4, dtype=torch.bool)
            expected[selected] = True
            if not torch.equal(actual, expected.expand_as(actual)):
                raise RuntimeError(f"Routed-bank remapping differs from native Top-2 for {query['id']}")
    row = {**query, "router_mass": dict.fromkeys(labels, sum(sum(x["topk_mass"]) for x in trace.routers)
                                                   / (len(tokens) * LAYERS))}
    row["random_rank"] = dict(zip(labels, random.Random(query["id"]).sample(range(len(labels)), len(labels))))
    for bank_type, bank_slice in zip(SUBSYSTEMS, (slice(0, 4), slice(4, 5), slice(None)), strict=True):
        count = present[:, :, bank_slice].sum(dim=2)
        # Bank mean first, then head, token, layer. Preserve the independent axes.
        by_token = values[:, :, bank_slice].sum(dim=2) / count[..., None, None]
        aggregate = by_token.mean(3).mean(2)
        for channel in CHANNELS:
            scores = aggregate[..., index[channel]].mean(1)
            sign = 1 if channel in ("prediction_cosine", "read_output_norm") else -1
            row[f"{bank_type}_{channel}"] = dict(zip(labels, (scores * sign).tolist(), strict=True))
        rel = -by_token[..., index["relative_residual_norm"]]
        row[f"{bank_type}_layers"] = {label: layer.tolist() for label, layer in
                                       zip(labels, rel.mean(3).mean(2), strict=True)}
        row[f"{bank_type}_heads"] = {label: head.tolist() for label, head in
                                      zip(labels, rel.mean(2).mean(1), strict=True)}
    return row


def collect(reader, states, labels, query, kernel, *, verify=False, writer=None):
    trace = ResidualTrace(kernel, verify=verify)
    tokens = encode(reader.backbone.tokenizer, query["prompt"])
    diag = SimpleNamespace(residual=trace, timing=True, measure=lambda *_: nullcontext(), removed_unit=None)
    # Only the native kernel's outputs and state continue through the model.
    output = reader.step(tokens, reader.start(states[label] for label in labels), CONFIG, diagnostics=diag)
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError("Nonfinite native read")
    # Store the fields keyed by physical labels, independent of execution order.
    row = score_trace(trace, tokens, labels, query, writer)
    return row, trace


def snapshot_cells(store, meta):
    return {label: dict(zip_sha256=sha(store.path(item["cell_id"])),
                        state_sha256=fingerprint(store.read(item["cell_id"])[1]),
                        revision=item["revision"]) for label, item in meta.items()}


def compare_scores(before, after, mapping, **tolerance):
    maximum = 0.0
    for key in [f"{s}_{c}" for s in SUBSYSTEMS for c in CHANNELS]:
        for target, source in mapping.items():
            a, b = after[key][target], before[key][source]
            maximum = max(maximum, abs(a - b))
            if not abs(a - b) <= tolerance["atol"] + tolerance["rtol"] * abs(b):
                raise AssertionError(f"{key}, {target} ← {source}: {a} vs {b}")
    return maximum


def pin_groups(backbone, root, group):
    store = Cells(root, backbone)
    meta, states = write_cells(backbone, store, group)
    meta["empty"] = store.create("empty")
    states["empty"] = {}
    return store, meta, states


def verify_trace(before, after, mapping, labels, observed_labels=LABELS, **tolerance):
    a = {(labels[cell], layer, bank): (tokens, values) for cell, layer, bank, tokens, values in before.banks}
    b = {(mapping[observed_labels[cell]], layer, bank): (tokens, values)
         for cell, layer, bank, tokens, values in after.banks}
    if set(a) != set(b):
        raise AssertionError("Residual trace bank coverage changed")
    maximum = 0.0
    for key, (tokens, actual) in b.items():
        original_tokens, expected = a[key]
        torch.testing.assert_close(tokens, original_tokens, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, **tolerance)
        maximum = max(maximum, float((actual - expected).abs().max()) if actual.numel() else 0)
    return maximum


def run_group(reader, root, group, kernel, *, verify=False, check_controls=False,
              writer=None, persistent=None, qa=False, original_traces=None):
    if persistent is None:
        store, meta, states = pin_groups(reader.backbone, root, group)
    else:
        store, meta, states = persistent
    before = snapshot_cells(store, meta)
    if {label: fingerprint(value) for label, value in states.items()} != {
            label: item["state_sha256"] for label, item in before.items()}:
        raise AssertionError("Persisted and in-memory state digests differ")
    rows, routers, parity, controls, answers = [], [], {}, {}, []
    for i, query in enumerate(group["queries"]):
        row, trace = collect(reader, states, LABELS, query, kernel, verify=verify and i == 0, writer=writer)
        if original_traces is not None:
            originals = SimpleNamespace(banks=original_traces[query["id"]])
            row["restored_trace_max_abs"] = verify_trace(originals, trace, dict(zip(LABELS, LABELS)),
                                                         LABELS, **SCORE_TOLERANCE)
        rows.append(row)
        routers.append(dict(query_id=query["id"], cell_slots=LABELS, layers=trace.routers))
        if trace.parity:
            parity = trace.parity
        if check_controls and i in (0, 4):
            identity = dict(zip(LABELS, LABELS, strict=True))
            repeated, repeat_trace = collect(reader, states, LABELS, query, kernel)
            verify_trace(trace, repeat_trace, identity, LABELS, atol=0, rtol=0)
            reversed_labels = LABELS[::-1]
            reordered, reverse_trace = collect(reader, states, reversed_labels, query, kernel)
            swaps = Cells(Path(root) / "swapped", reader.backbone)
            assigned = {label: swaps.store({**meta[label],
                                            "tokens_written": meta[source]["tokens_written"]},
                                           states[source])
                        for label, source in SWAP.items()}
            moved = {label: swaps.read(item["cell_id"])[1] for label, item in assigned.items()}
            swapped, _ = collect(reader, moved, LABELS, query, kernel)
            control = {"order_trace_max_abs": verify_trace(trace, reverse_trace, identity, LABELS,
                                                           reversed_labels, **SCORE_TOLERANCE)}
            for label, probe, mapping in (("repeat", repeated, identity), ("order", reordered, identity),
                                          ("swap", swapped, SWAP)):
                control[label] = compare_scores(row, probe, mapping, **SCORE_TOLERANCE)
            # The observed ranking must independently follow content on non-tied pairs.
            original = row[PRIMARY]
            flipped = swapped[PRIMARY]
            different = [(a, b) for a in LABELS for b in LABELS if a < b
                         and abs(original[SWAP[a]] - original[SWAP[b]]) > 1e-5]
            control["swap_pairs"] = len(different)
            control["swap_agreement"] = sum(
                (original[SWAP[a]] > original[SWAP[b]]) == (flipped[a] > flipped[b])
                for a, b in different) / len(different) if different else None
            controls[query["id"]] = control
        if qa and query["relevant"]:
            gold = query["relevant"]
            k = len(gold)
            answers.append(dict(query_id=query["id"], group=query["group"], category=query["category"], arms={
                arm: evaluate(reader, states, query, selected, CONFIG, None, generate_answer=True)
                for arm, selected in (("all", LABELS), ("oracle", gold), ("residual", ranking(row[PRIMARY])[:k]),
                                      ("random", ranking(row["random_rank"])[:k]))
            }))
    after = snapshot_cells(store, meta)
    if before != after or {label: fingerprint(value) for label, value in states.items()} != {
            label: item["state_sha256"] for label, item in before.items()}:
        raise AssertionError("A query changed persistent Cell state")
    return dict(rows=rows, router=routers, parity=parity, controls=controls, answers=answers, integrity=before)


def standalone_parity(backbone, diagnostic):
    from mom.layers import mom_gated_deltanet as ops

    rng = torch.Generator(device=backbone.device).manual_seed(671)
    cases = []
    for length, warm in ((1, False), (1, True), (19, True), (64, False)):
        q, k, v = [torch.randn((1, length, 4, 256), generator=rng, device=backbone.device).half() * 0.1
                   for _ in range(3)]
        q, k = ops.l2_norm(q), ops.l2_norm(k)
        g = -torch.rand((1, length, 4), device=backbone.device, generator=rng) * 0.1
        beta = torch.rand(g.shape, device=backbone.device, generator=rng).half()
        state = torch.randn((1, 4, 256, 256), device=backbone.device, generator=rng) * 0.01 if warm else None
        inputs = (q, k, v, g, beta)
        native = ops.fused_recurrent_gated_delta_rule(*inputs, initial_state=state, output_final_state=True)
        observed = diagnostic(*inputs, state.clone() if state is not None else None)
        expected = reference(*inputs, state)
        checks = {name: parity_report(actual, truth, **tol) for name, actual, truth, tol in (
            ("output", observed[0], native[0], OUTPUT_TOLERANCE),
            ("state", observed[1], native[1], STATE_TOLERANCE),
            ("reference_prediction", observed[2], expected[2], STATE_TOLERANCE),
            ("reference_residual", observed[3], expected[3], STATE_TOLERANCE),
            ("reference_output", observed[0], expected[0], OUTPUT_TOLERANCE),
            ("reference_state", observed[1], expected[1], STATE_TOLERANCE))}
        cases.append(dict(length=length, warm=warm, checks=checks))
    return cases


def sliced_metrics(rows, variant, axis):
    key = f"{variant}_{axis}"
    count = len(next(iter(rows[0][key].values())))
    slots, excluded = [], []
    for index in range(count):
        def project(without):
            data = []
            for row in rows:
                scores = {cell: ((sum(vals) - vals[index]) / (count - 1) if without else vals[index])
                          for cell, vals in row[key].items()}
                data.append({**row, "slice": scores})
            return metrics(data, "slice")["auroc"]
        slots.append(project(False))
        excluded.append(project(True))
    return dict(auroc=slots, leave_one_out_auroc=excluded)


def summaries(rows, controls):
    keys = [f"{subsystem}_{channel}" for subsystem in SUBSYSTEMS for channel in CHANNELS]
    keys += ["random_rank", "router_mass"]
    outcomes = {key: metrics(rows, key) for key in keys}
    outcomes[PRIMARY]["ci95"] = group_bootstrap(rows, PRIMARY)
    by_seed = {str(seed): metrics([r for r in rows if r["seed"] == seed], PRIMARY)
               for seed in sorted({r["seed"] for r in rows})}
    by_category = {category: metrics([r for r in rows if r["category"] == category], PRIMARY)
                   for category in ("T1", "T2", "T3", "T5")}
    slices = {variant: {axis: sliced_metrics(rows, variant, axis) for axis in ("layers", "heads")}
              for variant in SUBSYSTEMS}
    excluded = [auc for item in slices["routed"].values() for auc in item["leave_one_out_auroc"]]
    swap = [value["swap_agreement"] for value in controls.values() if value["swap_pairs"] > 0]
    verdict = rank_signal(outcomes[PRIMARY], by_seed, excluded,
                          bool(swap) and min(swap) >= 0.95, outcomes["random_rank"]["auroc"])
    calibration = {}
    for label in ("empty", "random"):
        wins = [(row[PRIMARY][cell] > row[PRIMARY][label]) + 0.5 * (
            row[PRIMARY][cell] == row[PRIMARY][label])
                for row in rows for cell in row["relevant"]]
        calibration[label] = sum(wins) / len(wins)
    return dict(status=verdict, primary=PRIMARY, methods=outcomes, seeds=by_seed,
                categories=by_category, axes=slices, controls=dict(swap_agreement=swap,
                relevant_beats_control=calibration), fact_groups=len({row["group"] for row in rows}))


def qa_report(rows):
    subsets = {"answerable": rows, "T5": [row for row in rows if row["category"] == "T5"]}
    report = {}
    for key, picked in subsets.items():
        report[key] = {arm: dict(count=len(picked),
                                 nll=sum(row["arms"][arm]["nll"] for row in picked) / len(picked),
                                 top1=sum(row["arms"][arm]["top1"] for row in picked),
                                 exact_match=sum(row["arms"][arm]["exact_match"] for row in picked))
                       for arm in ("all", "oracle", "residual", "random")}
    return report


def archive_stage(reader, args, groups, diagnostic, stage, *, verify=False):
    root, out = args.scratch, args.output
    field_order = out / "residual-fields.json"
    if not field_order.exists():
        write(field_order, dict(fields=FIELDS, layout="query/cell/layer/bank, [token,head,field]",
                                shared_bank=4, routed_banks="actual native Top-2 only",
                                scope="Only the first Smoke group's raw trace is returned to GitHub"))
    rows, routers, controls, integrity, parity, records = [], [], {}, {}, {}, []
    for i, group in enumerate(groups):
        key = group["group"]
        keep = stage == "smoke" and i == 0
        check_controls = stage == "smoke" or key.endswith("-0")
        archive = out / ("sample-residual-records.jsonl.gz" if keep else
                         f"residual-records/{stage}-{key}.jsonl.gz")
        archive.parent.mkdir(exist_ok=True)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(dir=root) if not keep else nullcontext(root / "original") as target:
            with gzip.open(archive, "wt", compresslevel=2, encoding="utf-8") as writer:
                # Separate only serializing/writing from native and diagnostic compute.
                export_seconds = 0.0

                def emit(row):
                    nonlocal export_seconds
                    start = time.monotonic()
                    writer.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    export_seconds += time.monotonic() - start

                result = run_group(reader, target, group, diagnostic, verify=verify,
                                   check_controls=check_controls, writer=emit)
            elapsed = time.monotonic() - started
        rows.extend(result["rows"])
        routers.extend(result["router"])
        controls.update(result["controls"])
        integrity[key] = result["integrity"]
        parity[key] = result["parity"]
        records.append(dict(group=key, file=archive.relative_to(out).as_posix(), sha256=sha(archive),
                            bytes=archive.stat().st_size, serialization_and_gzip_seconds=export_seconds,
                            elapsed_seconds=elapsed))
        if keep:
            store = Cells(root / "original", reader.backbone)
            manifests = {store.read(path.stem)[0]["name"]: store.read(path.stem)[0]
                         for path in store.root.glob("*.zip")}
            write(out / "restore-cells.json", manifests)
        for stem, value in (("cell-scores", rows), ("router-diagnostics", routers),
                            ("state-integrity", integrity), ("controls", controls),
                            ("trace-index", records), ("kernel-parity", parity)):
            write(out / f"{stage}-{stem}.json", value)
        print(f"{stage} {key}: {time.monotonic()-started:.1f}s", flush=True)
    return rows, controls, records


def restored_check(backbone, args, assets, diagnostic):
    source = Cells(args.scratch / "original", backbone)
    meta = read(args.output / "restore-cells.json")
    imported = Cells(args.scratch / "imported", backbone)
    imported_states = restore_cells(source, imported, meta)
    initial = snapshot_cells(source, meta)
    if any(fingerprint(state) != initial[label]["state_sha256"] for label, state in imported_states.items()):
        raise AssertionError("Restored recurrent tensors differ")
    group = assets["smoke"][0]
    original_traces = {query["id"]: [] for query in group["queries"]}
    with gzip.open(args.output / "sample-residual-records.jsonl.gz", "rt", encoding="utf-8") as source:
        for line in source:
            entry = json.loads(line)
            values = torch.tensor(entry["values"], dtype=torch.float32).reshape(-1, 4, len(FIELDS))
            original_traces[entry["query_id"]].append(
                (LABELS.index(entry["cell"]), entry["layer"], entry["bank"],
                 torch.tensor(entry["token_index"], dtype=torch.long), values))
    result = run_group(FusionReader(backbone), imported.root, group, diagnostic, verify=True,
                       persistent=(imported, meta, imported_states), original_traces=original_traces)
    original = {row["id"]: row for row in read(args.output / "smoke-cell-scores.json")}
    identity = dict(zip(LABELS, LABELS, strict=True))
    comparisons = {row["id"]: compare_scores(original[row["id"]], row, identity, **SCORE_TOLERANCE)
                   for row in result["rows"]}
    write(args.output / "restore.json", dict(status="PASS", pid=os.getpid(), comparisons=comparisons,
                                             token_head_max_abs={row["id"]: row["restored_trace_max_abs"]
                                                                 for row in result["rows"]},
                                             tolerance=SCORE_TOLERANCE, parity=result["parity"],
                                             state_integrity=result["integrity"]))


def benchmark(backbone, args, assets, diagnostic):
    reader = FusionReader(backbone)
    store = Cells(args.scratch / "original", backbone)
    meta = read(args.output / "restore-cells.json")
    states = [store.read(meta[label]["cell_id"])[1] for label in LABELS[:4]]
    row = assets["smoke"][0]["queries"][0]
    tokens = encode(backbone.tokenizer, row["prompt"])
    data = []

    def timing(fn):
        fn()  # Warm the exact shape and backend before measuring.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return dict(seconds=time.perf_counter()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated())

    for count in (1, 2, 4, 8):
        labels = [f"Cell{i}" for i in range(count)]
        active = dict(zip(labels, (states[i % len(states)] for i in range(count)), strict=True))
        native = timing(lambda: reader.step(tokens, reader.start(active.values()), FusionConfig()))
        shared = timing(lambda: reader.step(tokens, reader.start(active.values()), CONFIG))
        measured = timing(lambda: collect(reader, active, labels, row, diagnostic))
        with tempfile.TemporaryDirectory(dir=args.scratch) as target:
            torch.cuda.synchronize()
            started = time.perf_counter()
            path = Path(target) / "benchmark-residuals.jsonl.gz"
            with gzip.open(path, "wt", compresslevel=2, encoding="utf-8") as writer:
                def emit(entry):
                    writer.write(json.dumps(entry, separators=(",", ":"), allow_nan=False) + "\n")

                collect(reader, active, labels, row, diagnostic, writer=emit)
            torch.cuda.synchronize()
            export = time.perf_counter() - started
            data.append(dict(cells=count, native=native, shared_hidden=shared, diagnostic=measured,
                             diagnostic_overhead_seconds=measured["seconds"] - shared["seconds"],
                             diagnostic_and_export_seconds=export,
                             estimated_export_overhead_seconds=export-measured["seconds"],
                             compressed_bytes=path.stat().st_size))
        write(args.output / "benchmark.json", data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "snapshot", "tokenizer", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--stage", choices=("smoke", "restore", "pilot", "benchmark"))
    parser.add_argument("--scratch", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    args.output = out
    if not args.stage:
        out.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="residual-cells-") as scratch:
            for stage in ("smoke", "restore", "pilot", "benchmark"):
                print(f"Stage {stage}", flush=True)
                subprocess.run([sys.executable, "-u", __file__, str(args.source), str(args.snapshot),
                                str(args.tokenizer), str(out), "--stage", stage, "--scratch", scratch], check=True)
        status = read(out / "retrieval-metrics.json")["final"]["status"]
        qa = read(out / "qa-evaluation.json")
        if status == "NO_POSITIVE_SIGNAL":
            recommendation = "STOP_PRIMARY_RESIDUAL_ROUTING"
        elif status == "UNSTABLE_SIGNAL":
            recommendation = "INSPECT_LAYER_HEAD_AND_SHARED_EFFECTS"
        else:
            overall = qa["metrics"]["answerable"]
            eligible = all(overall[arm]["nll"] < overall["random"]["nll"]
                           and overall[arm]["top1"] > overall["random"]["top1"]
                           for arm in ("oracle", "residual"))
            recommendation = "RESIDUAL_002_CANDIDATE" if eligible else "QA_GAIN_NOT_ESTABLISHED"
        write(out / "decision.json", dict(status="DIAGNOSTIC_COMPLETE", engineering="PASS",
                                         research=status, qa=qa["status"], next_step=recommendation,
                                         elapsed_seconds=time.monotonic()-started,
                                         dataset_hash=sha(out / "dataset.json")))
        return
    torch.set_num_threads(2)
    started = time.monotonic()
    backbone = Backbone(args.source, args.snapshot, args.tokenizer, dtype="float16")
    if any(p.requires_grad for p in backbone.model.parameters()):
        raise RuntimeError("Backbone must remain frozen")
    from gdr_residual_triton import diagnostic

    identity = {**backbone.identity, "gpu": torch.cuda.get_device_name(),
                "cuda_runtime": torch.version.cuda, "fast_ops": os.getenv("FLA_USE_FAST_OPS", "0")}
    if args.stage == "smoke":
        assets = configure_data(backbone)
        write(out / "environment.json", identity)
        write(out / "dataset.json", assets)
        cases = standalone_parity(backbone, diagnostic)
        write(out / "synthetic-kernel-parity.json", cases)
        if not all(check["passed"] for case in cases for check in case["checks"].values()):
            raise RuntimeError("Diagnostic kernel disagrees with native or FP32 reference")
        archive_stage(FusionReader(backbone), args, assets["smoke"], diagnostic, "smoke", verify=True)
        write(out / "smoke-complete.json", dict(status="PASS", pid=os.getpid()))
        return
    if read(out / "environment.json") != identity:
        raise RuntimeError("Model or runtime identity changed between workers")
    assets = read(out / "dataset.json")
    if args.stage == "restore":
        if read(out / "smoke-complete.json")["pid"] == os.getpid():
            raise RuntimeError("Restore must run in an independent process")
        restored_check(backbone, args, assets, diagnostic)
    elif args.stage == "benchmark":
        benchmark(backbone, args, assets, diagnostic)
    else:
        reader = FusionReader(backbone)
        rows, controls, records = archive_stage(reader, args, assets["pilot"], diagnostic, "pilot")
        pilot = summaries(rows, controls)
        expansion = "SKIPPED_GATE"
        final = pilot
        if pilot["status"] == "RESEARCH_PASS":
            expected = sum(item["elapsed_seconds"] for item in records) / len(records)
            if (time.monotonic() - started) + 1.3*expected*len(assets["expansion"]) < 4500:
                new_rows, new_controls, _ = archive_stage(reader, args, assets["expansion"], diagnostic, "expansion")
                expansion = "COMPLETE"
                final = summaries(rows + new_rows, {**controls, **new_controls})
            else:
                expansion = "SKIPPED_BUDGET"
        write(out / "retrieval-metrics.json", dict(pilot=pilot, expansion=expansion, final=final))
        if final["status"] != "RESEARCH_PASS":
            write(out / "qa-evaluation.json", dict(status="SKIPPED_GATE"))
        else:
            answers = []
            for group in assets["pilot"]:
                with tempfile.TemporaryDirectory(dir=args.scratch) as target:
                    result = run_group(reader, target, group, diagnostic, qa=True)
                answers.extend(result["answers"])
                write(out / "qa-evaluation.json", dict(status="RUNNING", queries=answers))
            write(out / "qa-evaluation.json", dict(status="COMPLETE", queries=answers, metrics=qa_report(answers)))


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, RuntimeError, FloatingPointError, ValueError) as error:
        if "--stage" in sys.argv:
            out = Path(sys.argv[4])
            write(out / "failure.json", dict(stage=sys.argv[sys.argv.index("--stage") + 1],
                                             error=str(error), type=type(error).__name__))
        raise
