"""CPU replay of P1.3 with fixed addresses and matched perturbation norms."""
import argparse
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
import subprocess
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "p1-3-diagnostic"))
from diagnose import MODEL, REV, UP, ART, SEEDS, build_episodes, verify, restore, save, sha
from p1 import install
import importlib.util
spec = importlib.util.spec_from_file_location("p14", Path(__file__).resolve().parents[1] / "p1-4/run.py")
p14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p14)
from instrument import observe, summarize, same_tree


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "artifact", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--reference", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    commit = subprocess.check_output(["git", "-C", str(a.upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != UP:
        raise ValueError("Wrong Titans revision")
    sys.path.insert(0, str(a.upstream.resolve()))
    verify(a.artifact)
    a.output.mkdir(parents=True)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    encode = lambda text: tok.encode(text, add_special_tokens=False)
    plans = build_episodes(encode, 16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.float32,
                                                attn_implementation="eager").eval()
    block = model.model.layers[3]
    native = block.self_attn
    stages = defaultdict(dict)
    by_step = defaultdict(dict)
    episode_stats = defaultdict(dict)
    for seed in SEEDS:
        block.self_attn = native
        torch.manual_seed(seed)
        wrapper = install(model).eval()
        # CPU install can retain expanded parameter storage because .to(cpu)
        # is a no-op. Keep Parameter identity and materialize writable storage.
        with torch.no_grad():
            for parameter in wrapper.parameters():
                if parameter.requires_grad and any(
                    size > 1 and stride == 0
                    for size, stride in zip(parameter.shape, parameter.stride())
                ):
                    parameter.data = parameter.detach().clone()

        for step in (0, 200):
            path = a.artifact / f"seed_{seed}/step_{step:03}.pt"
            restore(wrapper, path, step)
            folder = a.output / f"seed_{seed}_step_{step:03}"
            folder.mkdir()
            rows, aggregate = [], defaultdict(list)
            for index, plan in enumerate(plans):
                if seed == SEEDS[0] and step == 0 and index == 0:
                    original = []
                    commit_fn = wrapper.commit
                    def sample():
                        commit_fn()
                        original.append(wrapper.state)
                    wrapper.commit = sample
                    try:
                        _, control = p14.episode(model, wrapper, encode, plan, index)
                    finally:
                        del wrapper.commit
                with observe(wrapper) as observations:
                    row, tensors = p14.episode(model, wrapper, encode, plan, index)
                if seed == SEEDS[0] and step == 0 and index == 0:
                    same_tree(tuple(control['reads'].values()), tuple(tensors['reads'].values()))
                    for state, observed in zip(original, observations, strict=True):
                        same_tree(tuple(t[:, -1] for t in state.updates), observed["fast"])
                        same_tree(tuple(t[0] for t in state.states[1]), observed["momentum"])
                    print("INSTRUMENTATION_EQUIVALENCE_PASS", flush=True)
                row["stages"] = summarize(observations, tensors)
                rows.append(row)
                for condition, metrics in row["metrics"].items():
                    for metric, matrix in metrics.items():
                        aggregate[f"{condition}/{metric}"].append(list(map(mean, zip(*matrix))))
                print(f"seed={seed} step={step} episode={index+1}/16", flush=True)
            save(folder / "metrics.json", dict(checkpoint_sha256=sha(path), episodes=rows))
            stages[step][seed] = {
                stage: {arm: [mean(x for matrix in r["stages"][stage] for x in matrix[arm]["norm"])
                              for r in rows] for arm in ("swap_delta", "random_delta")}
                for stage in ("loss", "raw_gradient", "weighted_gradient", "surprise", "momentum", "fast", "read")
            }
            episode_stats[step][seed] = dict(aggregate)
            by_step[step][seed] = {k: list(map(mean, zip(*v))) for k, v in aggregate.items()}
    import json
    reference = json.loads(a.reference.read_text())["by_step"]
    for step, seeds in by_step.items():
        for seed, metrics in seeds.items():
            for key, heads in metrics.items():
                torch.testing.assert_close(torch.tensor(heads),
                    torch.tensor(reference[str(step)][str(seed)][key]), rtol=0.005, atol=1e-7,
                    msg=f"P1.4 reproduction mismatch: {seed}/{step}/{key}")
    change = {seed: {k: [b-a for a,b in zip(v, by_step[200][seed][k])]
                     for k,v in by_step[0][seed].items()} for seed in SEEDS}
    paired = {}
    rng = random.Random(20261120)
    draws = [rng.choices(range(16), k=16) for _ in range(10000)]
    for metric in ("cosine", "amplitude_ratio", "projection_coefficient",
                   "orthogonal_residual_norm", "orthogonal_residual_ratio"):
        def per_episode(step, arm, index):
            return mean(mean(episode_stats[step][seed][f"{arm}/{metric}"][index]) for seed in SEEDS)

        def interval(values):
            samples = sorted(mean(values[i] for i in draw) for draw in draws)
            return dict(mean=mean(values), ci95=[samples[250], samples[9749]])

        effect = {step: [per_episode(step, "swap", i) - per_episode(step, "random", i)
                         for i in range(16)] for step in (0, 200)}
        paired[metric] = dict(
            swap_minus_random={step: interval(v) for step, v in effect.items()},
            step_change={arm: interval([per_episode(200, arm, i) - per_episode(0, arm, i)
                                       for i in range(16)]) for arm in ("swap", "random")},
            interaction=interval([b-a for a,b in zip(effect[0], effect[200])]))
    stage_stats = {}
    for stage in stages[0][SEEDS[0]]:
        def episode_mean(step, arm, i):
            return mean(stages[step][seed][stage][arm][i] for seed in SEEDS)
        stage_stats[stage] = {
            arm: dict(by_step={step: interval([episode_mean(step, arm, i) for i in range(16)])
                               for step in (0, 200)},
                      step_change=interval([episode_mean(200, arm, i)-episode_mean(0, arm, i)
                                            for i in range(16)]))
            for arm in ("swap_delta", "random_delta")}
    report = ["# P1.5 — Update and retrieval response", "",
              "Mean perturbation norm across episodes, seeds and heads/matrices.", "",
              "| Stage | Step 0 swap | Step 200 swap | Step 0 random | Step 200 random |",
              "|---|---:|---:|---:|---:|"]
    for stage, data in stage_stats.items():
        report.append("| " + stage + " | " + " | ".join(
            f"{data[arm]['by_step'][step]['mean']:.6g}"
            for arm in ("swap_delta", "random_delta") for step in (0, 200)) + " |")
    (a.output / "RESULTS.md").write_text("\n".join(report) + "\n")
    save(a.output / "summary.json", dict(by_step=by_step, step_200_minus_0=change, paired_episode_bootstrap=paired,
                                         stage_a_inner_learning={k: stage_stats[k] for k in ("loss","raw_gradient","weighted_gradient","surprise")},
                                         stage_b_fast_weights={k: stage_stats[k] for k in ("momentum","fast")},
                                         stage_c_retrieval=stage_stats["read"],
                                         artifact_commit=ART, model_revision=REV, upstream_revision=UP,
                                         dtype="float32", torch=torch.__version__, outer_updates=0))
    save(a.output / "decision.json", dict(format="minicells.titans-p1-5.v1",
         status="DIAGNOSTIC_ONLY", diagnostic_status="COMPLETE", device="cpu",
         outer_updates=0, checkpoints=6, episodes_per_checkpoint=16, conditions_per_episode=3,
         p1_4_reproduction="PASS", instrumentation_equivalence="PASS"))
    save(a.output / "sha256.json", {str(p.relative_to(a.output)):sha(p)
                                   for p in a.output.rglob("*") if p.is_file()})


if __name__ == "__main__":
    main()
