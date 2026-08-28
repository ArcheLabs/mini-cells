import torch

from minicells.clm_growth import ProgressiveGrowthCLM
from minicells.growth_pressure import cosine_kmeans_2, gradient_disagreement, pressure_score
from minicells.language_models import TextNCALM
from minicells.upcycled_cellular_textnca import UpcyclingConfig, convert_textnca_to_upcycled


def _model() -> ProgressiveGrowthCLM:
    torch.manual_seed(11)
    source = TextNCALM(vocab_size=23, max_context=8, dim=8, heads=2, ffn_dim=12,
                       windows=(2, 3, 4), iterations=(1, 1, 1), carry_bias=2.0)
    source = convert_textnca_to_upcycled(source, config=UpcyclingConfig(num_experts=4, top_k=1))
    return ProgressiveGrowthCLM(source)


def test_zero_birth_matches_clm01() -> None:
    torch.manual_seed(4)
    source = TextNCALM(vocab_size=23, max_context=8, dim=8, heads=2, ffn_dim=12,
                       windows=(2, 3, 4), iterations=(1, 1, 1), carry_bias=2.0)
    upcycled = convert_textnca_to_upcycled(source, config=UpcyclingConfig(num_experts=4, top_k=1))
    growth = ProgressiveGrowthCLM(upcycled)
    inputs = torch.randint(0, 23, (2, 6))
    torch.testing.assert_close(growth(inputs).logits, upcycled(inputs).logits, rtol=1e-5, atol=1e-6)


def test_two_births_preserve_function_and_are_recursive() -> None:
    model = _model()
    inputs = torch.randint(0, 23, (2, 6))
    perceptions = torch.randn(512, 8)
    first = model.birth(stage=1, parent_id="s1-e2", routed_perceptions=perceptions,
                        token=500_000, validation_inputs=inputs)
    second = model.birth(stage=1, parent_id=first["child"], routed_perceptions=perceptions,
                         token=1_000_000, validation_inputs=inputs)
    assert first["parity"]["status"] == "CLM_GROWTH_EQUIVALENCE"
    assert second["parity"]["status"] == "CLM_GROWTH_EQUIVALENCE"
    assert model.expert_counts_by_stage() == [4, 6, 4]
    assert model.growth_history[-1]["parent"] == first["child"]


def test_masked_dense_and_sparse_parity_after_growth() -> None:
    model = _model()
    model.birth(stage=0, parent_id="s0-e0", routed_perceptions=torch.randn(512, 8),
                token=500_000)
    inputs = torch.randint(0, 23, (2, 6))
    dense = model(inputs, execution_backend="masked_dense").logits
    sparse = model(inputs, execution_backend="sparse_dispatch").logits
    torch.testing.assert_close(dense, sparse, rtol=2e-5, atol=2e-6)


def test_pressure_formula_and_geometry_are_deterministic() -> None:
    gradients = [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])]
    assert gradient_disagreement(gradients) == 1.0
    assert pressure_score(.25, .4) == .35
    samples = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-.9, -.1]])
    torch.testing.assert_close(cosine_kmeans_2(samples), cosine_kmeans_2(samples))
