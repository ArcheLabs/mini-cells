"""Regression checks for the local Stage 7 tensor contracts."""
import torch

from run_stage7 import Memory, dataset
from statistics7 import paired


def test_clean_slate_gradient_and_diagnostics():
    torch.manual_seed(42)
    model = Memory()
    episode = tuple(torch.randn(4, heads, 256) for heads in (2, 8, 2))
    scores, losses = model(episode, mode='meta_train')
    (scores.square().mean() + sum(losses)).backward()
    for layer in model.address:
        assert layer.weight.grad.isfinite().all()
        assert layer.weight.grad.norm() > 0
    with torch.no_grad():
        trace = []
        scores, _ = model(episode, positions=torch.tensor([10, 20, 70, 520]),
                          width=64, tnt=True, trajectory=trace)
        assert scores.shape == (4, 4) and scores.isfinite().all()
        assert [r['seen'] for r in trace] == [2, 3, 4, 2, 3, 4]


def test_deterministic_split_and_paired_statistics():
    config = dict(seed=42, candidates=4, train_episodes=2,
                  validation_episodes=2, test_episodes=2)
    data = dataset(config)
    assert data == dataset(config)
    pools = [{v for episode in episodes for _, _, v in episode}
             for episodes in data.values()]
    assert len(set.union(*pools)) == sum(map(len, pools))
    assert paired([.25] * 4)['ci95'] == [.25, .25]
