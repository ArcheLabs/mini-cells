import pytest
import torch
from minicells.metrics import echo_metrics,edit_similarity,levenshtein

def test_metrics_known_values():
    targets=torch.tensor([[1,2,0],[1,2,3]]); predictions=torch.tensor([[1,9,9],[1,2,3]])
    logits=torch.nn.functional.one_hot(predictions,10).float(); mask=targets.ne(0)
    result=echo_metrics(logits,targets,mask)
    assert result["token_accuracy"] == pytest.approx(0.8); assert result["exact_sequence_accuracy"] == 0.5
    assert levenshtein("kitten","sitting") == 3; assert edit_similarity("abc","adc") == pytest.approx(2/3)
