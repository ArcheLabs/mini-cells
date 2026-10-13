from dataclasses import dataclass
import torch


@dataclass
class CandidateBatch:
    ids: torch.Tensor
    states: torch.Tensor


class CandidateProvider:
    def __init__(self, limit):
        self.limit = limit

    def candidates(self, active_ids, state, instance, ledger):
        ids = instance.candidates.index_select(0, active_ids)
        if ids.shape[1] > self.limit:
            raise ValueError('candidate limit exceeded')
        ledger.add('index_reads', ids.numel())
        ledger.add('candidate_reads', ids.numel())
        ledger.add('node_gathers', ids.numel() + active_ids.numel())
        ledger.add('state_reads', (ids.numel() + active_ids.numel()) * state.shape[-1])
        states = state.index_select(0, ids.flatten()).view(*ids.shape, -1)
        return CandidateBatch(ids, states)
