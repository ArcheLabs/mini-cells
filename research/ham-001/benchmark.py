"""HAM-001 CPU mechanism benchmark."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from statistics import mean, stdev
from datetime import datetime, timezone
import time
import tracemalloc

import numpy as np


@dataclass(frozen=True)
class Config:
    heads: int = 2
    dimension: int = 32
    beta: float = 1.
    alpha: float = .99
    normalize_keys: bool = True
    window: int = 16


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class Backend:
    def reset(self, config):
        self.__init__(config)

    def fresh(self):
        return type(self)(self.config)

    def step(self, query, key, value):
        result = self.read(query)
        self.write(key, value)
        return result

    def export_state(self):
        return {k: v.copy() for k, v in vars(self).items() if isinstance(v, np.ndarray)}

    def import_state(self, state):
        expected = self.export_state()
        if state.keys() != expected.keys():
            raise ValueError("State fields differ")
        for name, value in state.items():
            shape = expected[name].shape
            if value.dtype != np.float64 or not np.isfinite(value).all() or (
                value.shape != shape if name == "matrix" else value.shape[1:] != shape[1:]
            ):
                raise ValueError(f"Invalid state tensor: {name}")
        vars(self).update({k: v.copy() for k, v in state.items()})

    def get_stats(self):
        return dict(state_bytes=sum(x.nbytes for x in self.export_state().values()), parameters=0)


class Delta(Backend):
    def __init__(self, config):
        self.config = config
        self.matrix = np.zeros((config.heads, config.dimension, config.dimension))

    def read(self, query):
        return np.einsum("hvk,...hk->...hv", self.matrix, self.key(query))

    def key(self, key):
        return unit(key) if self.config.normalize_keys else key

    def write(self, key, value):
        self.matrix += self.config.beta * (value - self.read(key))[..., None] * self.key(key)[:, None, :]


class GatedDelta(Delta):
    def write(self, key, value):
        self.matrix *= self.config.alpha
        super().write(key, value)


class Attention(Backend):
    local = False
    hard = False

    def __init__(self, config):
        self.config = config
        self.keys = np.zeros((0, config.heads, config.dimension))
        self.values = self.keys.copy()

    def write(self, key, value):
        self.keys = np.concatenate((self.keys, unit(key)[None]))
        self.values = np.concatenate((self.values, value[None]))
        if self.local:
            self.keys = self.keys[-self.config.window:].copy()
            self.values = self.values[-self.config.window:].copy()

    def read(self, query):
        if not len(self.keys):
            return np.zeros_like(query)
        scores = np.einsum("...hd,nhd->...hn", unit(query), self.keys)
        if self.hard:
            best = len(self.keys)-1-np.argmax(scores[..., ::-1], axis=-1)
            return self.values[best, np.arange(self.config.heads)]
        weights = np.exp(16 * (scores - scores.max(axis=-1, keepdims=True)))
        return np.einsum("...hn,nhd->...hd", weights / weights.sum(axis=-1, keepdims=True), self.values)


class Local(Attention):
    local = True


class FullKV(Attention):
    hard = True


def dataset(seed, count, task, config):
    rng = np.random.default_rng(seed)
    shape = (count, config.heads, config.dimension)
    k = unit(rng.normal(size=shape))
    v = unit(rng.normal(size=shape))
    q = k.copy()
    target = v.copy()
    if task in ("correlated", "hard_negative"):
        rho = .95 if task == "hard_negative" else .5
        shared = k[::2]
        other = k[1::2]
        other = unit(other - (shared * other).sum(-1, keepdims=True) * shared)
        k[1::2] = rho * shared + np.sqrt(1-rho*rho)*other
        q = unit(k + (.012 if task == "hard_negative" else 0)*rng.normal(size=shape))
    if task == "overwrite":
        k = np.concatenate((k, k[:1]))
        v = np.concatenate((v, unit(rng.normal(size=shape[1:]))[None]))
        target[0] = v[-1]
    if task == "nonlinear":
        signs = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
        q = np.zeros((4, *shape[1:]))
        q[:, :, :2] = signs[:, None, :] / np.sqrt(2)
        target = np.zeros_like(q)
        target[:,:,0] = signs.prod(axis=1)[:,None]
        k,v = q[:3],target[:3]
    return k,v,q,target


def score(pred, target):
    error = np.square(pred-target).mean(-1)
    # A candidate is correct only when uniquely closest.
    distances = np.square(pred[:,None]-target[None]).mean(-1)
    diag = distances[np.arange(len(target)), np.arange(len(target))]
    distances[np.arange(len(target)), np.arange(len(target))] = np.inf
    margin = distances.min(axis=1)-diag
    return dict(top1=float((margin > 0).mean()), exact=float((error < 1e-12).mean()),
                mse=float(error.mean()), cosine=float((unit(pred)*unit(target)).sum(-1).mean()),
                margin=float(margin.mean()))


def evaluate(backend, data, task):
    k,v,q,target = data
    started = time.perf_counter()
    for a,b in zip(k,v):
        backend.write(a,b)
    write = time.perf_counter()-started
    started = time.perf_counter()
    pred = backend.read(q)
    read = time.perf_counter()-started
    result = dict(metrics=score(pred,target), write_seconds=write, read_seconds=read, **backend.get_stats())
    if task == "nonlinear":
        result["metrics"]["top1"] = float((np.sign(pred[...,0]) == target[...,0]).mean())
        result["unseen_accuracy"] = float((np.sign(pred[-1,:,0]) == target[-1,:,0]).mean())
        result["seen_accuracy"] = float((np.sign(pred[:-1,:,0]) == target[:-1,:,0]).mean())
        result["unseen_mse"] = float(np.square(pred[-1]-target[-1]).mean())
        result["seen_mse"] = float(np.square(pred[:-1]-target[:-1]).mean())
    if task == "overwrite":
        result["updated_mse"] = float(np.square(pred[0]-target[0]).mean())
        result["retained_mse"] = float(np.square(pred[1:]-target[1:]).mean())
    if task == "interference":
        backend.reset(backend.config)
        matrix = np.zeros((len(q),len(k)))
        prev = backend.read(q)
        for j,(a,b) in enumerate(zip(k,v)):
            backend.write(a,b)
            current = backend.read(q)
            matrix[:j,j] = (np.linalg.norm(current-prev,axis=-1)/np.linalg.norm(target,axis=-1)).mean(-1)[:j]
            prev = current
        valid = matrix[np.triu_indices(len(k),1)]
        result["interference"] = dict(matrix=matrix.tolist(), mean=float(valid.mean()),
            median=float(np.median(valid)), p95=float(np.quantile(valid,.95)), worst=float(valid.max()))
    if task == "transfer":
        # Independent instances; no replay of the prefix.
        backend.reset(backend.config)
        cuts = (len(k)//3, 2*len(k)//3, len(k))
        start = 0
        retention = []
        for end in cuts:
            for key, value in zip(k[start:end], v[start:end]):
                backend.write(key, value)
            state = backend.export_state()
            restored = backend.fresh()
            restored.import_state(state)
            np.testing.assert_array_equal(restored.read(q), backend.read(q))
            retention.append(score(restored.read(q[:cuts[0]]), target[:cuts[0]]))
            backend = restored
            start = end
        result["prefix_a_retention"] = retention
    return result


def main(output, titans_upstream=None):
    start = datetime.now(timezone.utc).isoformat()
    output.mkdir(parents=True)
    config = Config()
    arms = dict(B0=Local,B2=Delta,B3=GatedDelta,B5=Attention,full_kv=FullKV)
    if titans_upstream is not None:
        from titans_adapter import REVISION, Titans, load_upstream
        upstream = load_upstream(titans_upstream)
    rows = []
    tracemalloc.start()
    for task in ("random","correlated","hard_negative","overwrite","interference","nonlinear","transfer"):
        for count in ((8,16,32,64,128,256,512) if task == "random" else (16,)):
            for seed in (42,43,44):
                data = dataset(seed,count,task,config)
                digest = hashlib.sha256(b"".join(x.tobytes() for x in data)).hexdigest()
                backends = {arm: cls(config) for arm, cls in arms.items()}
                if titans_upstream is not None:
                    backends["B4"] = Titans(config, upstream, seed)
                for arm,backend in backends.items():
                    rows.append(dict(task=task,count=count,seed=seed,arm=arm,dataset_hash=digest,
                                     **evaluate(backend,data,task)))
    _,peak = tracemalloc.get_traced_memory()
    summaries = []
    for task,count,arm in sorted({(r["task"],r["count"],r["arm"]) for r in rows}):
        data = [r["metrics"]["top1"] for r in rows if (r["task"],r["count"],r["arm"])==(task,count,arm)]
        m,s = mean(data),stdev(data)
        half = 4.302653*s/np.sqrt(3)
        summaries.append(dict(task=task,count=count,arm=arm,mean=m,std=s,ci95=[m-half,m+half]))
    config_text = json.dumps(vars(config),sort_keys=True)
    report = dict(experiment_id="HAM-001", version=1, config=vars(config),
        config_hash=hashlib.sha256(config_text.encode()).hexdigest(), primary_metric="top1",
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        git_branch=subprocess.check_output(["git","branch","--show-current"],text=True).strip(),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        start_time=start,end_time=datetime.now(timezone.utc).isoformat(),
        hardware=platform.machine(), python=platform.python_version(),numpy=np.__version__,
        precision="float64", traced_peak_bytes=peak, rows=rows, summary=summaries,
        status={"H0":"CORE_VALIDATION","H1":"PARTIAL","H2-H5":"NOT_STARTED"},
        unavailable={"B1":"No validated input/state adapter for Mamba-2",
                     **({"B4":"Pinned upstream not supplied"} if titans_upstream is None else {})},
        titans_protocol=None if titans_upstream is None else dict(revision=REVISION,
            chunk_size=1, dtype="float32", initialization="fresh seeded synthetic geometry"))
    path = output/"results.json"
    path.write_text(json.dumps(report,allow_nan=False))
    (output/"sha256.json").write_text(json.dumps({path.name:hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--titans-upstream", type=Path)
    args = parser.parse_args()
    main(args.output, args.titans_upstream)
