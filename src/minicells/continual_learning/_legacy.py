from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..data import CopyDataGenerator
from ..quantization_localization import int_linear
from ..vocab import CharVocab

NUM_CELLS = 64
MAX_SEQ_LEN = 32
PARAMETER_COUNT = 4476
PARAMETER_MIN_Q = -2048
PARAMETER_MAX_Q = 2048
MARGIN_Q = 256
PERTURBATION_Q = 4
STEP_Q = 1
MARKER = "??"
PAYLOAD_SYMBOLS = "abcdefghijklmnopqrstuvwxyz0123456789"

EMBEDDING_OFFSET = 0
UPDATE_IN_WEIGHT_OFFSET = 352
UPDATE_IN_BIAS_OFFSET = 3168
UPDATE_OUT_WEIGHT_OFFSET = 3200
UPDATE_OUT_BIAS_OFFSET = 3712
OUTPUT_WEIGHT_OFFSET = 3728
OUTPUT_BIAS_OFFSET = 4432


@dataclass(frozen=True)
class TaskBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    mask: torch.Tensor
    lengths: torch.Tensor
    changed_mask: torch.Tensor

    def take(self, indices: list[int] | torch.Tensor) -> "TaskBatch":
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long)
        return TaskBatch(
            self.input_ids[indices],
            self.target_ids[indices],
            self.mask[indices],
            self.lengths[indices],
            self.changed_mask[indices],
        )

    @property
    def size(self) -> int:
        return int(self.input_ids.shape[0])


def _batch_from_encoded(
    inputs: list[list[int]], targets: list[list[int]], changed_positions: list[int | None]
) -> TaskBatch:
    if not inputs or len(inputs) != len(targets) or len(inputs) != len(changed_positions):
        raise ValueError("inputs, targets, and changed_positions must have equal non-zero length")
    ids = torch.zeros((len(inputs), NUM_CELLS), dtype=torch.long)
    out = torch.zeros_like(ids)
    lengths = torch.tensor([len(row) for row in inputs], dtype=torch.long)
    mask = torch.arange(NUM_CELLS).unsqueeze(0) < lengths.unsqueeze(1)
    changed = torch.zeros_like(mask)
    for row, (source, target, changed_pos) in enumerate(zip(inputs, targets, changed_positions)):
        if len(source) != len(target) or not 1 <= len(source) <= MAX_SEQ_LEN:
            raise ValueError("invalid sequence length")
        ids[row, : len(source)] = torch.tensor(source, dtype=torch.long)
        out[row, : len(target)] = torch.tensor(target, dtype=torch.long)
        if changed_pos is not None:
            changed[row, changed_pos] = True
    return TaskBatch(ids, out, mask, lengths, changed)


def build_old_pool(vocab: CharVocab, seed: int, examples: int) -> TaskBatch:
    generator = CopyDataGenerator(
        vocab,
        seed=seed,
        min_length=1,
        max_length=MAX_SEQ_LEN,
        num_cells=NUM_CELLS,
        random_fraction=0.7,
    )
    inputs: list[list[int]] = []
    while len(inputs) < examples:
        text = generator.sample_text()
        if text.startswith(MARKER):
            continue
        inputs.append(vocab.encode(text))
    return _batch_from_encoded(inputs, [row.copy() for row in inputs], [None] * len(inputs))


def _shift_payload_id(vocab: CharVocab, token_id: int) -> int:
    token = vocab.id_to_token[int(token_id)]
    index = PAYLOAD_SYMBOLS.index(token)
    return vocab.token_to_id[PAYLOAD_SYMBOLS[(index + 1) % len(PAYLOAD_SYMBOLS)]]


def build_adaptation_pool(vocab: CharVocab, seed: int, examples: int) -> TaskBatch:
    rng = random.Random(seed)
    marker_ids = vocab.encode(MARKER)
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    changed_positions: list[int] = []
    for _ in range(examples):
        payload_len = rng.randint(2, 10)
        first = rng.choice(PAYLOAD_SYMBOLS)
        tail = "".join(rng.choice(vocab.SYMBOLS) for _ in range(payload_len - 1))
        source = marker_ids + vocab.encode(first + tail)
        target = source.copy()
        target[len(marker_ids)] = _shift_payload_id(vocab, source[len(marker_ids)])
        inputs.append(source)
        targets.append(target)
        changed_positions.append(len(marker_ids))
    return _batch_from_encoded(inputs, targets, changed_positions)


def combine_batches(parts: list[TaskBatch]) -> TaskBatch:
    if not parts:
        raise ValueError("parts must not be empty")
    return TaskBatch(
        torch.cat([part.input_ids for part in parts]),
        torch.cat([part.target_ids for part in parts]),
        torch.cat([part.mask for part in parts]),
        torch.cat([part.lengths for part in parts]),
        torch.cat([part.changed_mask for part in parts]),
    )


def load_q88_model(path: Path) -> torch.Tensor:
    raw = path.read_bytes()
    if len(raw) != PARAMETER_COUNT * 2:
        raise ValueError(f"expected {PARAMETER_COUNT * 2} bytes, got {len(raw)}")
    values = np.frombuffer(raw, dtype="<i2").astype(np.int64)
    if values.size != PARAMETER_COUNT:
        raise ValueError("wrong parameter count")
    if values.min() < PARAMETER_MIN_Q or values.max() > PARAMETER_MAX_Q:
        raise ValueError("parameter outside Q8.8 V1 bounds")
    return torch.from_numpy(values.copy())


def save_q88_model(path: Path, flat: torch.Tensor) -> None:
    values = flat.detach().cpu().numpy().astype("<i2", copy=False)
    if values.size != PARAMETER_COUNT:
        raise ValueError("wrong parameter count")
    path.write_bytes(values.tobytes())


def unpack_flat(flat: torch.Tensor) -> dict[str, torch.Tensor]:
    if flat.numel() != PARAMETER_COUNT:
        raise ValueError("wrong parameter count")
    return {
        "embedding": flat[EMBEDDING_OFFSET:UPDATE_IN_WEIGHT_OFFSET].reshape(44, 8),
        "update_in_w": flat[UPDATE_IN_WEIGHT_OFFSET:UPDATE_IN_BIAS_OFFSET].reshape(32, 88),
        "update_in_b": flat[UPDATE_IN_BIAS_OFFSET:UPDATE_OUT_WEIGHT_OFFSET],
        "update_out_w": flat[UPDATE_OUT_WEIGHT_OFFSET:UPDATE_OUT_BIAS_OFFSET].reshape(16, 32),
        "update_out_b": flat[UPDATE_OUT_BIAS_OFFSET:OUTPUT_WEIGHT_OFFSET],
        "output_w": flat[OUTPUT_WEIGHT_OFFSET:OUTPUT_BIAS_OFFSET].reshape(44, 16),
        "output_b": flat[OUTPUT_BIAS_OFFSET:PARAMETER_COUNT],
    }


@torch.no_grad()
def exact_logits(flat: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    params = unpack_flat(flat.to(dtype=torch.int64, device="cpu"))
    ids = input_ids.to(dtype=torch.long, device="cpu")
    batch, num_cells = ids.shape
    if num_cells != NUM_CELLS:
        raise ValueError(f"expected {NUM_CELLS} cells")
    state = torch.zeros((batch, NUM_CELLS, 16), dtype=torch.int64)
    embedded = params["embedding"][ids]
    for _ in range(4):
        zero = torch.zeros((batch, 2, 16), dtype=torch.int64)
        padded = torch.cat((zero, state, zero), dim=1)
        neighborhood = torch.cat(
            [padded[:, offset : offset + NUM_CELLS] for offset in range(5)], dim=-1
        )
        update_input = torch.cat((neighborhood, embedded), dim=-1)
        hidden = int_linear(update_input, params["update_in_w"], params["update_in_b"])
        hidden = hidden.clamp(0, 32767)
        delta = int_linear(hidden, params["update_out_w"], params["update_out_b"])
        state = (state + delta).clamp(-256, 256)
    return int_linear(state, params["output_w"], params["output_b"])


@torch.no_grad()
def margin_loss(flat: torch.Tensor, batch: TaskBatch, margin_q: int = MARGIN_Q) -> int:
    logits = exact_logits(flat, batch.input_ids)
    targets = batch.target_ids.to(dtype=torch.long, device="cpu")
    mask = batch.mask.to(dtype=torch.bool, device="cpu")
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    competitors = logits.clone()
    competitors.scatter_(-1, targets.unsqueeze(-1), -(1 << 60))
    other = competitors.max(dim=-1).values
    loss = (margin_q - (target_logits - other)).clamp_min(0)
    return int(loss[mask].sum().item())


@torch.no_grad()
def evaluate_model(flat: torch.Tensor, batch: TaskBatch, chunk_size: int = 128) -> dict[str, float]:
    correct = 0
    total = 0
    exact = 0
    changed_correct = 0
    changed_total = 0
    total_loss = 0
    for start in range(0, batch.size, chunk_size):
        stop = min(start + chunk_size, batch.size)
        part = batch.take(torch.arange(start, stop))
        logits = exact_logits(flat, part.input_ids)
        targets = part.target_ids.to(dtype=torch.long)
        mask = part.mask.to(dtype=torch.bool)
        changed = part.changed_mask.to(dtype=torch.bool)
        pred = logits.argmax(dim=-1)
        correct += int(((pred == targets) & mask).sum().item())
        total += int(mask.sum().item())
        exact += int((((pred == targets) | (~mask)).all(dim=1)).sum().item())
        changed_correct += int(((pred == targets) & changed).sum().item())
        changed_total += int(changed.sum().item())

        target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        competitors = logits.clone()
        competitors.scatter_(-1, targets.unsqueeze(-1), -(1 << 60))
        other = competitors.max(dim=-1).values
        token_loss = (MARGIN_Q - (target_logits - other)).clamp_min(0)
        total_loss += int(token_loss[mask].sum().item())

    result = {
        "token_accuracy": correct / total if total else 0.0,
        "exact_sequence_accuracy": exact / batch.size,
        "margin_loss_per_token": total_loss / total if total else 0.0,
    }
    if changed_total:
        result["changed_accuracy"] = changed_correct / changed_total
    return result


def model_hash(flat: torch.Tensor) -> bytes:
    values = flat.detach().cpu().numpy().astype("<i2", copy=False)
    state = hashlib.blake2b(digest_size=32)
    state.update(b"mini-cells:model:v1")
    state.update(values.tobytes())
    return state.digest()


def _spsa_seed(parent_hash: bytes, generation: int) -> int:
    state = hashlib.blake2b(digest_size=32)
    state.update(b"mini-cells:spsa:v1")
    state.update(parent_hash)
    state.update(int(generation).to_bytes(8, "little", signed=False))
    return int.from_bytes(state.digest()[:8], "little", signed=False)


def delta_vector(parent_hash: bytes, generation: int, block_size: int) -> torch.Tensor:
    if not 1 <= block_size <= PARAMETER_COUNT:
        raise ValueError("invalid block_size")
    seed = np.uint64(_spsa_seed(parent_hash, generation))
    indices = np.arange(1, PARAMETER_COUNT + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = seed + np.uint64(0x9E3779B97F4A7C15) * indices
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        bits = (z ^ (z >> np.uint64(31))) & np.uint64(1)
    delta = np.where(bits == 0, -1, 1).astype(np.int64)
    if block_size < PARAMETER_COUNT:
        blocks = (PARAMETER_COUNT + block_size - 1) // block_size
        block = generation % blocks
        start = block * block_size
        stop = min(start + block_size, PARAMETER_COUNT)
        mask = np.zeros(PARAMETER_COUNT, dtype=bool)
        mask[start:stop] = True
        delta[~mask] = 0
    return torch.from_numpy(delta)


def candidate(flat: torch.Tensor, delta: torch.Tensor, side: int,
              perturbation_q: int = PERTURBATION_Q) -> torch.Tensor:
    return (flat + side * perturbation_q * delta).clamp(PARAMETER_MIN_Q, PARAMETER_MAX_Q)


def apply_update(flat: torch.Tensor, delta: torch.Tensor, loss_plus: int, loss_minus: int,
                 step_q: int = STEP_Q) -> tuple[torch.Tensor, bool, int]:
    if loss_plus == loss_minus:
        return flat, False, 0
    direction = 1 if loss_plus < loss_minus else -1
    updated = (flat + direction * step_q * delta).clamp(PARAMETER_MIN_Q, PARAMETER_MAX_Q)
    return updated, True, direction


def _selection_seed(domain: str, parent_hash: bytes, generation: int) -> int:
    state = hashlib.blake2b(digest_size=32)
    state.update(b"mini-cells:continual-batch:v1")
    state.update(domain.encode("utf-8"))
    state.update(parent_hash)
    state.update(int(generation).to_bytes(8, "little", signed=False))
    return int.from_bytes(state.digest()[:8], "little", signed=False)


def select_indices(pool_size: int, count: int, domain: str, parent_hash: bytes,
                   generation: int) -> list[int]:
    rng = random.Random(_selection_seed(domain, parent_hash, generation))
    return [rng.randrange(pool_size) for _ in range(count)]


def training_batch(mode: str, old_pool: TaskBatch, new_pool: TaskBatch, parent_hash: bytes,
                   generation: int, batch_size: int = 4) -> TaskBatch:
    if batch_size != 4:
        raise ValueError("Experiment 003C keeps the production V1 batch shape at 4")
    if mode == "old-only":
        return old_pool.take(select_indices(old_pool.size, 4, mode, parent_hash, generation))
    if mode == "new-only":
        return new_pool.take(select_indices(new_pool.size, 4, mode, parent_hash, generation))
    if mode == "replay":
        old = old_pool.take(select_indices(old_pool.size, 2, mode + ":old", parent_hash, generation))
        new = new_pool.take(select_indices(new_pool.size, 2, mode + ":new", parent_hash, generation))
        return combine_batches([old, new])
    raise ValueError(mode)
