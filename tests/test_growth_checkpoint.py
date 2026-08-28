import torch

from minicells.clm_growth import ProgressiveGrowthCLM
from minicells.growth_checkpoint import GlobalLRScheduler, load_growth_checkpoint, save_growth_checkpoint
from minicells.language_models import TextNCALM
from minicells.upcycled_cellular_textnca import UpcyclingConfig, convert_textnca_to_upcycled


def _model() -> ProgressiveGrowthCLM:
    source = TextNCALM(vocab_size=19, max_context=8, dim=8, heads=2, ffn_dim=12,
                       windows=(2, 3, 4), iterations=(1, 1, 1), carry_bias=2.0)
    return ProgressiveGrowthCLM(convert_textnca_to_upcycled(
        source, config=UpcyclingConfig(num_experts=4, top_k=1)))


def test_optimizer_inherits_parent_state_and_scheduler_continues(tmp_path) -> None:
    model = _model()
    inputs = torch.randint(0, 19, (2, 6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(inputs).logits.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    parent = model.stages[0].program_bank.experts["s0-e0"]
    parent_state = optimizer.state[next(parent.parameters())]["exp_avg"].clone()
    scheduler = GlobalLRScheduler(optimizer, lambda step: 0.01 / (1 + step), step=4)
    model.birth(stage=0, parent_id="s0-e0", routed_perceptions=torch.randn(512, 8),
                token=500_000, optimizer=optimizer)
    child = model.stages[0].program_bank.experts["s0-e4"]
    torch.testing.assert_close(optimizer.state[next(child.parameters())]["exp_avg"], parent_state)
    assert scheduler.step(4) == 0.002
    assert all(group["lr"] == 0.002 for group in optimizer.param_groups)

    path = tmp_path / "growth.pt"
    save_growth_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler,
                           consumed_tokens=500_000, training_step=4)
    restored = _model()
    optimizer2 = torch.optim.AdamW(restored.parameters(), lr=.5)
    scheduler2 = GlobalLRScheduler(optimizer2, lambda step: .01 / (1 + step))
    restored, payload = load_growth_checkpoint(path, model=restored, optimizer=optimizer2,
                                               scheduler=scheduler2)
    assert restored.expert_counts_by_stage() == [5, 4, 4]
    assert payload["consumed_tokens"] == 500_000
    torch.testing.assert_close(restored(inputs).logits, model(inputs).logits, rtol=1e-5, atol=1e-6)
