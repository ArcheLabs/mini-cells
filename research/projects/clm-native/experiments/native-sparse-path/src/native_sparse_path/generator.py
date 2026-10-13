"""Generate graphs before solving. Candidate construction never sees answers."""
from dataclasses import dataclass, replace
import hashlib
import numpy as np
import torch

SPLITS = {'train': 100_000, 'dev': 200_000, 'sealed_test': 300_000, 'ood': 400_000}
FEATURES = 10  # scale[2], bias[2], key[2], priority, initial[2], root flag


@dataclass(frozen=True)
class Instance:
    x: torch.Tensor
    candidates: torch.Tensor
    roots: torch.Tensor
    depth: int
    task: str
    instance_id: str
    template_id: int
    fixed_candidates: bool = False

    def to(self, device):
        return replace(self, x=self.x.to(device), candidates=self.candidates.to(device),
                       roots=self.roots.to(device))


def generate(task, split, index, *, nodes=64, candidates=8, depth=4,
             sealed_authorized=False, fixed_candidates=False):
    if split not in SPLITS or task not in ('A', 'B', 'C'):
        raise ValueError('unknown task/split')
    if split == 'sealed_test' and not sealed_authorized:
        raise PermissionError('sealed_test is inaccessible before registration')
    if not 0 <= index < 100_000:
        raise ValueError('index outside isolated seed range')
    if candidates >= nodes or nodes < candidates * (depth + 1) and task != 'B':
        raise ValueError('insufficient nodes for unique candidate layers')
    template_id = 90_000 if fixed_candidates else SPLITS[split] + index % 64
    rng = np.random.default_rng(SPLITS[split] + index + ord(task) * 1_000_000)
    topology = np.random.default_rng(template_id + ord(task) * 1_000_000)
    x = np.zeros((nodes, FEATURES), dtype=np.float32)
    x[:, :2] = rng.uniform(0.65, 1.05, (nodes, 2))
    x[:, 2:4] = rng.uniform(-0.25, 0.25, (nodes, 2))
    x[:, 4:6] = rng.normal(size=(nodes, 2))
    x[:, 6] = topology.permutation(nodes) / nodes  # unique, address-invariant priority
    x[:, 7:9] = rng.uniform(-0.4, 0.4, (nodes, 2))
    ids = topology.permutation(nodes)
    roots = ids[:1 if task != 'B' else 2]
    x[roots, 9] = 1
    lists = np.empty((nodes, candidates), dtype=np.int64)
    for node in range(nodes):
        lists[node] = topology.choice(np.delete(np.arange(nodes), node), candidates, replace=False)
    if task != 'B':
        layers = ids[:candidates * (depth + 1)].reshape(depth + 1, candidates)
        roots = layers[0, :1]
        x[:, 9] = 0
        x[roots, 9] = 1
        for layer in range(depth):
            for node in layers[layer]:
                lists[node] = topology.permutation(layers[layer + 1])
    if not fixed_candidates:
        # Physical addressing is unrelated to value, label, or candidate position.
        permutation = rng.permutation(nodes)
        inverse = np.argsort(permutation)
        x = x[permutation]
        lists = inverse[lists[permutation]]
        roots = inverse[roots]
    digest = hashlib.sha256(x.tobytes() + lists.tobytes() + roots.tobytes()).hexdigest()
    return Instance(torch.tensor(x), torch.tensor(lists), torch.tensor(roots), depth,
                    task, digest, template_id, fixed_candidates)


def permute(instance, permutation):
    permutation = torch.as_tensor(permutation, device=instance.x.device)
    inverse = torch.argsort(permutation)
    return replace(instance, x=instance.x[permutation],
                   candidates=inverse[instance.candidates[permutation]],
                   roots=inverse[instance.roots])
