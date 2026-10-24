"""Two independent T4 shards per launch; private checkpoints, isolated reports, no automatic Main."""

import inspect
import time
from functools import lru_cache
import argparse
import copy
import json
from pathlib import Path
import uuid
import torch
from minicells_artifacts.hf_storage import HubBackend
from minicells_r3e.common import read, atomic, canonical, source_sha, fingerprint
from minicells_r3e.runner import secrets
from minicells_r3e.storage import snapshot_lock
from minicells_r3h import runner as coordinator
from minicells_r3j.storage import Store, save
from minicells_r3l.storage import recover_latest
from . import anchor, data, recovery
from .training import restore as trainer_restore
from .storage import Publisher, restore as restore_private
from .engine import run_arm
from .evaluation import repeated, attribution, specialization
from .reports import write

BASE = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def implementation_source():
    return source_sha()


def identity(c):
    row = dict(
        experiment_id=c["experiment_id"],
        source_sha=implementation_source(),
        config_sha256=canonical(c),
        phase=c.get("phase", "PILOT"),
    )
    return dict(row, run_id=canonical(row))


def location(output, seed, env):
    return Path(output) / "private-checkpoints/shards" / ("%s-%s" % (seed, env))


def arm_identity(ident, seed, env, arm):
    return dict(ident, seed=seed, environment=env, arm=arm)


def evaluate_shard(c, output, research, seed, env, device, tokenizer):
    root = location(output, seed, env)
    if (root / "EVALUATION.json").exists():
        return
    ident = identity(c)
    runtime = anchor.runtime(c)
    pressure_records = {}
    values = {}
    cost = {}
    curves = {}
    transactions = {}
    model_identity = {}
    grids = {}
    spec = {}
    for arm in c["pilot_arms"]:
        aid = arm_identity(ident, seed, env, arm)
        directory = root / arm
        if not (directory / "COMPLETE.json").exists():
            raise RuntimeError("SHARD_TRAINING_BARRIER_INCOMPLETE")
        values[arm] = {}
        for horizon in c["barriers"]:
            store = Store(directory, aid)
            receipt = read(store.receipt_file(horizon))
            payload = store.verified(receipt)
            tr = trainer_restore(runtime, payload["trainer"], seed, device)
            values[arm][str(horizon)] = repeated(tr.model, research, "pilot_report", tokenizer)
            if horizon == 600:
                aux = payload["aux"]
                pressure_records[arm] = aux["pressure_events"]
                cost[arm] = dict(
                    main_updates=aux["main_updates"],
                    main_tokens=aux["main_updates"] * c["tokens_per_step"],
                    shadow_updates=aux["shadow_updates"],
                    discarded_shadow_updates=aux["discarded_shadow_updates"],
                    proposal_optimizer_updates=aux["proposal_optimizer_updates"],
                    trainer_cost=tr.cost,
                    report_seconds=sum(x["seconds"] for x in values[arm].values()),
                    strict_equal_total_compute=False,
                )
                curves[arm] = aux["curve"]
                transactions[arm] = aux["transactions"]
                model_identity[arm] = dict(
                    parameters=sum(p.numel() for p in tr.model.parameters()),
                    cells=sum(p.numel() for p in tr.model.cells.parameters()),
                    connection=sum(p.numel() for p in tr.model.connection.parameters()),
                    step=tr.step,
                    receipt=receipt,
                    connection_sha=fingerprint(tr.model.connection.state_dict()),
                )
    for horizon in (400, 600):
        models = []
        for arm in ("P0", "P1"):
            store = Store(root / arm, arm_identity(ident, seed, env, arm))
            payload = store.verified(read(store.receipt_file(horizon)))
            models.append(trainer_restore(runtime, payload["trainer"], seed, device).model)
        grids[str(horizon)] = attribution(models[0], models[1], research, "pilot_report", tokenizer)
        spec[str(horizon)] = {
            arm: specialization(m, research, "pilot_report", tokenizer)
            for arm, m in zip(("P0", "P1"), models)
        }
    from collections import Counter

    frequencies = Counter(r["answer"] for r in research["arithmetic"]["train"])
    majority = frequencies.most_common(1)[0][0]
    report = research["arithmetic"]["pilot_report"]
    atomic(
        root / "EVALUATION.json",
        dict(
            seed=seed,
            environment=env,
            evaluations=values,
            cost=cost,
            curves=curves,
            pressure=pressure_records,
            transactions=transactions,
            model_identity=model_identity,
            attribution=grids,
            specialization=spec,
            answer_frequency_baseline=sum(r["answer"] == majority for r in report) / len(report),
        ),
    )


def worker(c, output, folder, index, count, phase):
    from tokenizers import Tokenizer

    backend = HubBackend(c["baseline"]["hf_reference"]["hf_repo"])
    runtime = anchor.runtime(c)
    historical = data.historical_assets(runtime, backend, Path(folder) / "historical", ("train",))
    research, manifest = data.load(folder, c)
    tokenizer = Tokenizer.from_file(str(Path(folder) / "historical/tokenizer.json"))
    ident = identity(c)
    jobs = read(Path(output) / "SHARD_PLAN.json")["jobs"]
    for job in jobs[index::count]:
        seed, env = job["seed"], job["environment"]
        root = location(output, seed, env)
        if phase == "train":
            tr, evidence = anchor.load(
                c, backend, Path(output) / "parent-r3h", seed, "cuda:%s" % index
            )
            atomic(root / "ANCHOR.json", evidence)
            # Each arm starts from the same verified state; no target-aware arm choice.
            for arm in c["pilot_arms"]:
                aid = arm_identity(ident, seed, env, arm)
                with snapshot_lock(Path(output) / "private-checkpoints"):
                    atomic(root / arm / "identity.json", aid)
                run_arm(
                    c,
                    copy.deepcopy(runtime),
                    tr,
                    aid,
                    root / arm,
                    historical,
                    research,
                    env,
                    arm,
                    "cuda:%s" % index,
                )
        elif phase == "evaluate":
            evaluate_shard(c, output, research, seed, env, "cuda:%s" % index, tokenizer)


def automatic(c, path, output, folder, report_only=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    folder = Path(folder)
    data.boundaries(c)
    ident = identity(c)
    session = uuid.uuid4().hex
    secrets()
    backend = HubBackend(c["baseline"]["hf_reference"]["hf_repo"])
    backend.check_write()
    revision = backend.info().sha
    pointer = backend.optional("r3m/" + ident["run_id"] + "/latest.json", revision)
    prior = dict(pointer, hf_repo=backend.repo, hf_revision=revision) if pointer else None
    private = output / "private-checkpoints"
    if prior:
        restore_private(backend, prior, private, output, ident, session, c["barriers"])
    publisher = Publisher(backend, private, output, ident, session, prior)

    last_publication = [0.0]

    def persist(force=True):
        if not force and time.monotonic() - last_publication[0] < 180:
            return
        with snapshot_lock(private):
            publisher.publish(force)
            last_publication[0] = time.monotonic()

    failures = []
    try:
        if not report_only:
            if torch.cuda.device_count() != 2:
                raise RuntimeError("TWO_T4_GPUS_REQUIRED_FOR_PAIRED_SHARDS")
            runtime = anchor.runtime(c)
            if not (folder / "manifest.json").exists():
                data.prepare(c, runtime, backend, folder)
            research, manifest = data.load(folder, c)
            atomic(output / "DATA_BOUNDARY_PROOF.json", read(folder / "DATA_BOUNDARY_PROOF.json"))
            engineering = private / "engineering"
            eid = dict(ident, kind="CROSS_SESSION_REFERENCE")
            ref = recover_latest(engineering, eid)
            if ref is None:
                tr, evidence = anchor.load(
                    c, backend, output / "parent-r3h", c["pilot_seeds"][0], "cuda:0"
                )
                historical = data.historical_assets(
                    runtime, backend, folder / "historical", ("train",)
                )
                from minicells_r3c_moe_bridge.data import train_batches

                rng = copy.deepcopy(tr.data_rng)
                batches = train_batches(historical, runtime["engine"], rng, "cuda:0")
                payload = recovery.reference(
                    tr,
                    batches,
                    c,
                    dict(ident, seed=c["pilot_seeds"][0]),
                    sampled_rng_state=copy.deepcopy(rng.bit_generator.state),
                )
                payload["anchor_evidence"] = evidence
                payload["asset_manifest"] = manifest
                save(engineering, eid, payload, 0)
                persist()
                print(
                    json.dumps(
                        dict(
                            status="CROSS_SESSION_REFERENCE_PERSISTED",
                            next="Save Version; start a NEW Kaggle session and Run All the SAME Notebook. It will restore automatically.",
                            production_updates=0,
                            hf_reference=publisher.prior,
                        )
                    ),
                    flush=True,
                )
                return
            if not (private / "CROSS_SESSION_PROOF.json").exists():
                historical = data.historical_assets(
                    runtime, backend, folder / "historical", ("train",)
                )
                from minicells_r3c_moe_bridge.data import train_batches

                proof = recovery.verify(
                    ref,
                    runtime,
                    c,
                    "cuda:0",
                    batch_factory=lambda tr: train_batches(
                        historical, runtime["engine"], tr.data_rng, "cuda:0"
                    ),
                )
                proof.update(
                    identity=ident, pinned_hf_reference=prior, asset_sha256=canonical(manifest)
                )
                atomic(private / "CROSS_SESSION_PROOF.json", proof)
                persist()
            atomic(output / "CROSS_SESSION_PROOF.json", read(private / "CROSS_SESSION_PROOF.json"))
            unfinished = []
            for seed in c["pilot_seeds"]:
                for env in c["environments"]:
                    if not (location(output, seed, env) / "EVALUATION.json").exists():
                        unfinished.append(dict(seed=seed, environment=env))
            # Exactly two independent seed/environment shards per Session. No process timer/kill.
            plan = dict(
                jobs=unfinished[:2], remaining_shards=max(0, len(unfinished) - 2), identity=ident
            )
            atomic(output / "SHARD_PLAN.json", plan)
            print(dict(status="R3M_PILOT_SHARDS", **plan), flush=True)
            namespace = dict(coordinator.workers.__globals__, BASE=BASE)
            exec(
                inspect.getsource(coordinator.workers).replace("[R3H_SNAPSHOT]", "[R3M_SNAPSHOT]"),
                namespace,
            )
            worker_pool = namespace["workers"]
            # Existing coordinator's flags and launch protocol are reused without historical edits.
            for phase in ("train", "evaluate"):
                worker_pool(runtime, path, output, folder, phase, lambda force: persist(force))
            persist()
    except Exception as error:
        failures.append(dict(code=type(error).__name__, message=str(error)[:180]))
        try:
            persist()
        except Exception as publication_error:
            failures.append(dict(code="HF_LOCAL_RETAINED", type=type(publication_error).__name__))
    report = write(c, output, ident, failures, publisher.prior)
    try:
        from .publication import publish

        report["publication"] = publish(output)
    except Exception as error:
        report["publication"] = dict(
            status="LOCAL_RETAINED", type=type(error).__name__, message=str(error)[:160]
        )
    atomic(output / "RUN_STATUS.json", report)
    print(json.dumps(report), flush=True)
    if failures:
        raise RuntimeError("R3M_STOPPED_SEE_RUN_STATUS_AND_RETAINED_CHECKPOINTS")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(BASE / "config/pilot.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--worker", type=int)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--phase")
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--report-existing", action="store_true")
    p.add_argument("--main", action="store_true")
    a = p.parse_args()
    c = read(a.config)
    if canonical(c) != (BASE / "config/releases.lock").read_text().strip():
        raise RuntimeError("FROZEN_CONFIG_CHANGED")
    if a.main or (a.phase or "").startswith("main-"):
        from .main_study import configure, automatic as main_auto, worker as main_worker

        if not (BASE / "release/PILOT_RUN_MANIFEST.json").exists():
            raise RuntimeError("MAIN_NOT_RELEASED_REVIEWED_PILOT_REQUIRED")
        c = configure(c, BASE / "release/PILOT_RUN_MANIFEST.json")
        if a.worker is not None:
            secrets()
            main_worker(c, a.output, a.assets, a.worker, a.workers, a.phase)
        else:
            main_auto(c, Path(a.config).resolve(), a.output, a.assets)
        return
    if a.worker is not None:
        secrets()
        worker(c, a.output, a.assets, a.worker, a.workers, a.phase)
    else:
        automatic(c, Path(a.config).resolve(), a.output, a.assets, a.report_existing)
