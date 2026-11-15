"""Independent scalar/matrix checks for the Phase C contract."""
import torch
import pytest

from memory import NeuralMemory


def test_projection_matches_spectral_solution():
    memory = NeuralMemory(4, ridge=.25)
    q = torch.arange(4, dtype=torch.float32)[None]
    eigenvalues = torch.tensor([.5, .25, .125, 0.])
    matrix = eigenvalues.diag()
    torch.testing.assert_close(memory.project(q, matrix), q * eigenvalues / (eigenvalues + .25))
    matrix[0, 1] = 1
    with pytest.raises(ValueError, match='symmetric'):
        memory.project(q, matrix)


def test_reference_update_coefficients():
    memory = NeuralMemory(4)
    state = memory.create()
    key = torch.tensor([2., 0., 0., 0.])
    unit = key / key.norm()
    target = torch.arange(4, dtype=torch.float32)
    eta, beta, alpha, rho = memory.update.controls
    loss = (memory.predict(unit, state.fast) - target).square().mean()
    grads = torch.autograd.grad(loss, state.fast)
    next_state = memory.transition(key, target, state, state.fast)
    for old, grad, moment, updated in zip(state.fast, grads, next_state.momentum, next_state.fast):
        torch.testing.assert_close(moment, -memory.update.gamma * beta * grad)
        torch.testing.assert_close(updated, (1-memory.update.delta*alpha)*old + moment)
    torch.testing.assert_close(next_state.address, (1-rho)*torch.outer(unit, unit))
    assert next_state.step == 1
