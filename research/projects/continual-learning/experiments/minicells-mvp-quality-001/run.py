"""Run the frozen eight-Cell quality protocol with independent per-question streams."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "apps/mom-native-cells"))

import torch
from minicells import MiniCells
from minicells_native.model_assets import MODEL_13B, REVISION_13B
from run_t4_1b_capacity import environment, write
from dataset import COUNTS, DOMAINS, VERSION, build
from evaluation import answer_ids, metrics, question_only, teacher
from reporting import FORMAT, MIN_AUTO_EM_GAIN, calculate, decision, publish_tables

EXPERIMENT = "minicells-mvp-quality-001"
SPLITS = tuple(COUNTS)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


@contextmanager
def deadline(seconds):
    def expired(*_):
        raise TimeoutError(f"Evaluation exceeded {seconds} seconds")
    if not hasattr(signal, "SIGALRM"):
        raise RuntimeError("The measured runner requires a Linux process")
    prior = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prior)


def prepare_cells(mc, work, records, source=None):
    """Store actual independent Cell bundles, then evaluate from their imports."""
    bundles = work / "bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    texts = {domain: "\n".join(fact["text"] for fact in facts) for domain, facts in records.items()}
    corpus_hash = {domain: sha256(text.encode()).hexdigest() for domain, text in texts.items()}
    receipt_file = work / "bundle_manifest.json"
    if source is not None and not receipt_file.exists():
        receipt = read(source / "bundle_manifest.json")
        import shutil
        for domain in DOMAINS:
            shutil.copyfile(source / "bundles" / f"{domain}.zip", bundles / f"{domain}.zip")
        write(work, receipt_file.name, receipt)
    if receipt_file.exists():
        receipt = read(receipt_file)
    else:
        receipt = {}
        for domain, facts in records.items():
            file = bundles / f"{domain}.zip"
            matches = [c for c in mc.cells.list() if c.name == domain]
            cell = matches[0] if matches else mc.cells.create(domain)
            if not cell.tokens_written:
                cell = mc.cells.write(cell.id, texts[domain], resume=True)
            if not file.exists():
                mc.cells.export(cell.id, file)
            meta = mc.cells._get(cell.id)
            receipt[domain] = dict(id=cell.id, revision=cell.revision, tokens_written=cell.tokens_written,
                                   source_record_ids=[x["record_id"] for x in facts],
                                   bundle_sha256=sha(file), state_sha256=meta["tensor_sha256"],
                                   write_text_sha256=corpus_hash[domain],
                                   write_method="SDK create then write from empty state")
        write(work, receipt_file.name, receipt)
    confirmed = {c.id: c for c in mc.cells.list()}
    if set(receipt) != set(DOMAINS):
        raise ValueError("Expected collections for the eight fixed domains")
    for domain, entry in receipt.items():
        file = bundles / f"{domain}.zip"
        if sha(file) != entry["bundle_sha256"]:
            raise ValueError(f"Bundle checksum mismatch for {domain}")
        if entry["id"] in confirmed:
            mc.cells.delete(entry["id"])
        cell = mc.cells.import_cell(file)
        stored = mc.cells._get(cell.id)
        if (cell.id != entry["id"] or cell.name != domain or cell.revision != entry["revision"]
                or cell.tokens_written <= 0 or cell.tokens_written != entry["tokens_written"]
                or stored["tensor_sha256"] != entry["state_sha256"]
                or entry["write_text_sha256"] != corpus_hash[domain]
                or entry["source_record_ids"] != [x["record_id"] for x in records[domain]]):
            raise ValueError(f"Imported Cell identity differs for {domain}")
    if (len(mc.cells.list()) != 8 or len({x["state_sha256"] for x in receipt.values()}) != 8
            or len({x["bundle_sha256"] for x in receipt.values()}) != 8):
        raise ValueError("Expected eight registered Cells with distinct recurrent tensors")
    return receipt


def requests(row, ids, arms):
    relevant = [ids[key] for key in row["relevant_cell_ids"]]
    control = [ids[key] for key in row["irrelevant_cell_ids"]]
    options = dict(A=[], B=relevant, C=control, D="auto")
    for arm in arms:
        yield arm, options[arm]
    if len(relevant) == 2 and "B" in arms:
        yield "single_1", relevant[:1]
        yield "single_2", relevant[1:]


def trial(mc, row, arm, requested, ids, args, result=None):
    began = time.monotonic()
    if result is None:
        result = {}
    result.update(sample_id=row["sample_id"], task_type=row["task_type"], split=row["split"],
                  prompt=row["prompt"], arm=arm)
    with deadline(args.timeout):
        context, trace, generated = question_only(mc, row["prompt"], requested, args.max_new_tokens,
                                                   args.evaluation != "teacher")
        # Gold and its token count enter only after SDK selection and free decoding.
        gold = row["gold_answer"]
        result.update(status="OK", gold_answer=gold, selection=trace,
                      relevant_cell_ids=[ids[d] for d in row["relevant_cell_ids"]],
                      selected_count=len(trace["selected"]), fallback=arm == "D" and bool(trace["fallback"]),
                      execution_mismatch=(trace["executed"] is not None
                                          and trace["executed"] != trace["selected"]))
        if arm == "D" and (trace["unscored"] or set(trace["candidates"]) != set(ids.values())):
            result["execution_mismatch"] = True
        if generated is not None:
            result.update(**generated, **metrics(generated["text"], gold))
        if args.evaluation != "generation":
            target = answer_ids(mc._backbone.tokenizer, row["prompt"], gold)
            if len(target) > args.max_new_tokens:
                raise ValueError("Gold would be truncated by the frozen generation limit")
            values = teacher(mc, context, target, trace["selected"])
            result["teacher_seconds"] = values.pop("latency_seconds")
            result.update(values)
            result["teacher_executed_cell_ids"] = list(trace["selected"])
    return {**result, "wall_seconds": time.monotonic() - began}


def dump_rows(path, items):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        for item in items:
            stream.write(json.dumps(item, allow_nan=False) + "\n")
    tmp.replace(path)


def report_run(out, data, latest, attempts, arms, args):
    rows = list(latest.values())
    wanted = [split for split in SPLITS
              if args.split in ("all", split) or any(r["split"] == split for r in rows)]
    summaries, tables = {}, []
    status, reason = "INCONCLUSIVE", "Held-out evaluation has not run."
    for split in wanted:
        split_rows = [row for row in rows if row["split"] == split]
        values, table = calculate(data[split], split_rows, arms, args.evaluation)
        summaries[split] = values
        tables.extend(dict(split=split, **item) for item in table)
        if split == "held_out":
            status, reason = decision(split, data[split], split_rows, values, arms, args.evaluation)
    failures = [r for r in attempts if r["status"] != "OK" or r.get("fallback")
                or r.get("execution_mismatch") or r.get("parse_failure")]
    write(out, "failures.json", failures)
    measured = status in ("PASS", "FAIL", "INCONCLUSIVE") and "held_out" in summaries and bool(
        summaries["held_out"]["relevant"]["contrasts"].get("D-A", {}).get("exact_match", {}).get("n"))
    complete = measured and status != "INVALID" and set(arms) == set("ABCD") and args.evaluation == "both"
    gates = dict(integrity="INVALID" if status == "INVALID" else "PASS" if complete else "NOT_EVALUATED",
                 evaluators="PASS" if complete and (out / "ready_for_held_out.json").exists() else "NOT_EVALUATED",
                 routing="MEASURED" if complete else "NOT_EVALUATED",
                 product_effect=status)
    write(out, "summary.json", dict(status=status, reason=reason, gates=gates, splits=summaries))
    publish_tables(out, tables)
    dump_rows(out / "sample_results.jsonl", attempts)
    keys = ("sample_id", "arm", "split", "task_type")
    def identity(row):
        return {k: row[k] for k in keys}
    dump_rows(out / "selection_traces.jsonl", [dict(identity(r), **r["selection"]) for r in attempts
                                              if "selection" in r])
    dump_rows(out / "teacher_forced_metrics.jsonl", [dict(identity(r), **{k: v for k, v in r.items()
        if k.startswith(("target_", "mean_target", "teacher_"))}) for r in attempts if "target_token_nll" in r])
    dump_rows(out / "generation_metrics.jsonl", [dict(identity(r), **{k: r[k] for k in
        ("text", "token_ids", "stop_reason", "generated_token_count", "latency_seconds", "exact_match",
         "token_f1", "task_accuracy", "constraint_satisfied", "parse_failure") if k in r})
        for r in attempts if "text" in r])
    lines = ["# MiniCells MVP quality", "", f"**{status}**. {reason}", "",
             "B uses prelabelled oracle selection. Lower NLL/PPL and higher exact match/F1 are better.",
             "Paired intervals resample questions 2,000 times. The frozen primary margin is +5 pp EM "
             "for D − A on related queries. Unrelated queries and pair-versus-single are secondary.", ""]
    for split, summary in summaries.items():
        lines.extend([f"## {split}", "", "| Group | Arm | n | Mean NLL | Mean PPL | Exact Match |",
                      "| --- | --- | ---: | ---: | ---: | ---: |"])
        for group, section in summary.items():
            for arm, values in section["groups"].items():
                m = [values[k]["mean"] for k in ("mean_target_nll", "target_perplexity", "exact_match")]
                shown = ["—" if x is None else f"{x:.4f}" for x in m]
                lines.append(f"| {group} | {arm} | {values['samples']} | " + " | ".join(shown) + " |")
        lines.extend(["", "| Group | Contrast | Metric | n | Mean | Paired 95% CI |",
                      "| --- | --- | --- | ---: | ---: | --- |"])
        for group, section in summary.items():
            for comparison, values in section["contrasts"].items():
                for metric, item in values.items():
                    if item["n"]:
                        lo, hi = item["ci95"]
                        lines.append(f"| {group} | {comparison} | {metric} | {item['n']} | "
                                     f"{item['mean']:.4f} | [{lo:.4f}, {hi:.4f}] |")
        lines.extend(["", "| Group | Routing metric | n | Mean |",
                      "| --- | --- | ---: | ---: |"])
        for group, section in summary.items():
            for metric, item in section["selection"].items():
                lines.append(f"| {group} | {metric} | {item['n']} | {item['mean']:.4f} |")
            for domain, item in section["domain_hit"].items():
                lines.append(f"| {group} | {domain} hit | {item['n']} | {item['mean']:.4f} |")
        lines.extend(["", "Accuracy and NLL by whether D selected all relevant Cells", "",
                      "| Group | Conditional metric | n | Mean |",
                      "| --- | --- | ---: | ---: |"])
        for group, section in summary.items():
            for metric, item in section["performance_by_supported_selection"].items():
                lines.append(f"| {group} | {metric} | {item['n']} | {item['mean']:.4f} |")
        lines.append("")
    lines += ["## Interpretation", "", *[f"- {gate}: {outcome}" for gate, outcome in gates.items()], ""]
    if complete:
        relevant = summaries["held_out"]["relevant"]
        contrasts = relevant["contrasts"]
        baseline = relevant["groups"]["A"]["exact_match"]["mean"]
        oracle = contrasts["B-A"]["exact_match"]["mean"]
        automatic_gap = contrasts["B-D"]["exact_match"]["mean"]
        pair = summaries["held_out"]["cross_cell"]["contrasts"]["B-best_single"]["exact_match"]["mean"]
        negative = summaries["held_out"]["irrelevant"]["contrasts"]["C-A"]["exact_match"]["mean"]
        lines.extend([
            f"- Irrelevant Cell impact on unrelated questions: C − A is {negative:+.1%} exact match.",
            f"- D missed at least one labelled Cell on {relevant['automatic_missing_support']} related questions. "
            f"The oracle-minus-automatic exact-match gap is {automatic_gap:+.1%}.",
            f"- The oracle exact-match effect is {oracle:+.1%} relative to a base accuracy of {baseline:.1%}. "
            "A small oracle effect leaves open write, retention, read and fusion limitations.",
            f"- Pair minus best single on cross-Cell questions is {pair:+.1%} exact match.",
            "Use the paired intervals and conditional metrics above to assess these point estimates.", ""])
    lines += [f"Flagged attempts, fallback and parse failures: {len(failures)}. See failures.json.",
              "The default SDK has no active rejection threshold in this run. Synthetic questions "
              "use domain cues and exact integer answers. Conclusions apply to this fixed fixture.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write(out, "decision.json", dict(format=FORMAT, status=status, diagnosis=reason,
          engineering_status="PASS" if status != "INVALID" and rows and not any(
              r["status"] != "OK" or r.get("execution_mismatch") for r in rows) else "FAIL"))


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--bundles", type=Path)
    cli.add_argument("--model", default=MODEL_13B, choices=[MODEL_13B])
    cli.add_argument("--model-revision", default=REVISION_13B, choices=[REVISION_13B])
    cli.add_argument("--tokenizer-revision", default=REVISION_13B, choices=[REVISION_13B])
    cli.add_argument("--dataset-version", default=VERSION, choices=[VERSION])
    cli.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    cli.add_argument("--evaluation", choices=["both", "teacher", "generation"], default="both")
    cli.add_argument("--arms", default="ABCD")
    cli.add_argument("--top-k", type=int, choices=[2], default=2)
    cli.add_argument("--seed", type=int, default=57219)
    cli.add_argument("--max-new-tokens", type=int, default=8)
    cli.add_argument("--timeout", type=int, default=180)
    cli.add_argument("--resume", action="store_true")
    return cli


def main():
    args = parser().parse_args()
    arms = list(args.arms)
    if (not arms or len(set(arms)) != len(arms) or set(arms) - set("ABCD")
            or not 1 <= args.max_new_tokens <= 512 or args.timeout < 1):
        raise ValueError("Use unique A/B/C/D arms and positive bounded generation settings")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    work = out / "work"
    attempts_dir = work / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    records, data = build()
    write(out, "dataset.json", dict(version=VERSION, records=records, splits=data))
    dataset_hash = sha(out / "dataset.json")
    write(out, "dataset_manifest.json", dict(version=VERSION, sha256=dataset_hash,
          sizes={k: len(v) for k, v in data.items()}, generator_seed=57219))
    attempts = [read(file) for file in sorted(attempts_dir.glob("*.json"))]
    latest = {(r["sample_id"], r["arm"]): r for r in attempts}
    wanted = SPLITS if args.split == "all" else (args.split,)
    run_id = dict(revision=git("rev-parse", "HEAD"), dataset_hash=dataset_hash,
                  code_hashes={p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                  top_k=args.top_k, threshold=None, seed=args.seed, arms=arms,
                  evaluation=args.evaluation, max_new_tokens=args.max_new_tokens,
                  timeout=args.timeout, min_auto_em_gain=MIN_AUTO_EM_GAIN)
    request_env = environment()
    run_id["dirty"] = bool(git("status", "--porcelain"))
    start = stamp()
    if (out / "manifest.json").exists():
        prior = read(out / "manifest.json")
        start = prior["started_at"]
        if run_id != prior["configuration"]:
            raise ValueError("Resume requires identical code, dataset, and evaluation settings")
    manifest = dict(experiment=EXPERIMENT, format=FORMAT, configuration=run_id,
                    repository="https://github.com/ArcheLabs/mini-cells", branch=git("branch", "--show-current"),
                    command=sys.argv, started_at=start, environment=request_env, python=platform.python_version(),
                    model_revision=args.model_revision, tokenizer_revision=args.tokenizer_revision,
                    sdk_version=version("mini-cells"), do_sample=False, num_beams=1,
                    answer_eos="excluded", generation_eos="natural_stop", retries_per_arm=1)
    try:
        if (torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != "Tesla T4"
                or torch.cuda.get_device_capability(0) != (7, 5)):
            raise RuntimeError("Expected one visible Tesla T4 at cuda:0")
        torch.cuda.set_device(0)
        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        manifest["environment"].update(torch=str(torch.__version__), cuda=torch.version.cuda,
                                       device=str(torch.cuda.get_device_properties(0)))
        print(f"MODEL_LOAD_STARTED {args.model}@{args.model_revision}", flush=True)
        with (
            MiniCells.open(args.model, work / "registry", device="cuda:0", dtype="fp16",
                           max_cells=8, selection_top_k=args.top_k) as mc,
            MiniCells._from_backbone(mc._backbone, work / "base", selection_top_k=args.top_k) as base,
        ):
            if base.cells.list():
                raise ValueError("Base arm requires an empty registry")
            identity = mc.model_info
            if identity["model_revision"] != args.model_revision or identity["tokenizer_revision"] != args.tokenizer_revision:
                raise ValueError("Loaded runtime identity differs from the frozen model and tokenizer")
            manifest["model"] = identity
            if "model" in locals().get("prior", {}) and prior["model"] != identity:
                raise ValueError("Cannot resume with different model or kernel identity")
            print("MODEL_LOAD_COMPLETED; creating or importing eight Cell Bundles", flush=True)
            cells = prepare_cells(mc, work, records, args.bundles)
            write(out, "cell_manifest.json", cells)
            freeze = dict(configuration=run_id, model=identity, cells_sha256=sha(out / "cell_manifest.json"),
                          dataset_sha256=dataset_hash)
            if (out / "frozen_protocol.json").exists() and read(out / "frozen_protocol.json") != freeze:
                raise ValueError("Frozen inputs changed")
            write(out, "frozen_protocol.json", freeze)
            write(out, "manifest.json", manifest)
            ids = {domain: cell["id"] for domain, cell in cells.items()}
            for split in wanted:
                if split == "held_out" and not (out / "ready_for_held_out.json").exists():
                    raise RuntimeError("A successful smoke and development in this output are required")
                for i, row in enumerate(data[split]):
                    for arm, selection in requests(row, ids, arms):
                        key = row["sample_id"], arm
                        old = latest.get(key)
                        if old and (old["status"] == "OK" or old["attempt"] > 1):
                            continue
                        attempt = 1 if old is None else 2
                        result = {}
                        try:
                            runtime = base if arm == "A" else mc
                            result = trial(runtime, row, arm, selection, ids, args, result)
                            if split == "smoke" and i == 0 and arm in ("A", "B", "D"):
                                repeat = trial(runtime, row, arm, selection, ids, args)
                                if (result.get("token_ids") != repeat.get("token_ids")
                                        or result["selection"]["selected"] != repeat["selection"]["selected"]
                                        or abs(result.get("mean_target_nll", 0)
                                               - repeat.get("mean_target_nll", 0)) > 0.02):
                                    raise ValueError("Repeated isolated smoke exceeded the frozen tolerance")
                        except Exception as exc:
                            result.update(sample_id=row["sample_id"], arm=arm, split=split,
                                          task_type=row["task_type"], prompt=row["prompt"],
                                          gold_answer=row["gold_answer"], status="ERROR",
                                          error_type=type(exc).__name__, reason=str(exc),
                                          stop_reason="timeout" if isinstance(exc, TimeoutError) else "exception")
                        result["attempt"] = attempt
                        write(attempts_dir, f"{key[0]}-{arm}-{attempt}.json", result)
                        latest[key] = result
                        attempts.append(result)
                    print(f"{split} {i + 1}/{len(data[split])} {row['sample_id']}", flush=True)
                if any(r["status"] != "OK" or r.get("execution_mismatch") for r in latest.values()
                       if r["split"] == split) and split != "held_out":
                    raise RuntimeError(f"{split} integrity failed; see recorded attempts")
                if split == "development":
                    required = [(row["sample_id"], arm) for part in ("smoke", "development")
                                for row in data[part] for arm, _ in requests(row, ids, arms)]
                    if all(key in latest and latest[key]["status"] == "OK"
                           and not latest[key]["execution_mismatch"] for key in required):
                        write(out, "ready_for_held_out.json", {"status": "PASS"})
            observed = {cell.name: sha(mc.cells._store.path(cell.id)) for cell in mc.cells.list()}
            if observed != {name: cell["bundle_sha256"] for name, cell in cells.items()}:
                raise RuntimeError("Evaluation changed persisted Cell state")
        report_run(out, data, latest, attempts, arms, args)
    except Exception as exc:
        report_run(out, data, latest, attempts, arms, args)
        write(out, "decision.json", dict(format=FORMAT, status="INVALID", engineering_status="FAIL",
              diagnosis=f"{type(exc).__name__}: {exc}"))
        summary = read(out / "summary.json")
        summary.update(status="INVALID", reason=f"{type(exc).__name__}: {exc}",
                       gates=dict(integrity="INVALID", evaluators="NOT_EVALUATED",
                                  routing="NOT_EVALUATED", product_effect="INVALID"))
        write(out, "summary.json", summary)
        failures = read(out / "failures.json")
        failures.append(dict(status="ERROR", stage="experiment", error_type=type(exc).__name__, reason=str(exc)))
        write(out, "failures.json", failures)
        with (out / "report.md").open("a", encoding="utf-8") as stream:
            stream.write(f"\nExecution failure: {type(exc).__name__}: {exc}\n")
        raise
    finally:
        manifest["finished_at"] = stamp()
        manifest["outcomes"] = {a: {k: sum(row["arm"] == a and predicate(row) for row in latest.values())
            for k, predicate in (("total", lambda row: True), ("success", lambda row: row["status"] == "OK"),
                                 ("failure", lambda row: row["status"] != "OK"),
                                 ("fallback", lambda row: row.get("fallback", False)))}
            for a in (*arms, "single_1", "single_2")}
        manifest["result_sha256"] = {p.name: sha(p) for p in sorted(out.iterdir())
                                     if p.is_file() and p.name != "manifest.json"}
        write(out, "manifest.json", manifest)


if __name__ == "__main__":
    main()
