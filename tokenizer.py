from dataclasses import dataclass

import numpy as np

from config import Config, EnvConfig
from env import EQUAL, PAD as ENV_PAD, Observation

PAD, T_MASK = 0, 1  #PAD: right-padding only; T_MASK: the target position under cmp
CMP_BASE = 2  #LESS, EQUAL, GREATER = env codes 0, 1, 2 at 2, 3, 4
SIZE_BASE = 5


@dataclass(frozen=True)
class Vocab:  #fixed from architecture constants, never from a run's n_min / n_max
    n_index: int  #model.n_index_logits: sizes 1..n_index, indices 0..n_index - 1
    v: int  #values 0..v - 1, shared by T and val feedback

    @property
    def index_base(self) -> int:
        return SIZE_BASE + self.n_index

    @property
    def value_base(self) -> int:
        return self.index_base + self.n_index

    @property
    def size(self) -> int:
        return self.value_base + self.v


def make_vocab(cfg: Config) -> Vocab:
    return Vocab(n_index=cfg.model.n_index_logits, v=cfg.env.v)


def _check(cond: np.ndarray | bool, msg: str) -> None:
    if not np.all(cond):
        raise ValueError(msg)


def encode(obs: Observation, vocab: Vocab) -> np.ndarray:
    size, k_max = obs.queries.shape
    _check((obs.n >= 1) & (obs.n <= vocab.n_index), f"N outside [1, {vocab.n_index}]")
    asked = np.arange(k_max)[None, :] < obs.steps[:, None]
    _check(~asked | ((obs.queries >= 0) & (obs.queries < obs.n[:, None])), "query outside [0, N)")
    tokens = np.full((size, 2 + 2 * k_max), PAD, dtype=np.int64)
    tokens[:, 0] = SIZE_BASE + obs.n - 1
    if obs.target is None:  #cmp: the target value is absent from the observation and from the sequence
        tokens[:, 1] = T_MASK
        _check(~asked | ((obs.feedback >= 0) & (obs.feedback <= 2)), "cmp feedback outside {LESS, EQUAL, GREATER}")
        fb = CMP_BASE + obs.feedback
    else:
        _check((obs.target >= 0) & (obs.target < vocab.v), f"target outside [0, {vocab.v})")
        tokens[:, 1] = vocab.value_base + obs.target
        _check(~asked | ((obs.feedback >= 0) & (obs.feedback < vocab.v)), f"val feedback outside [0, {vocab.v})")
        fb = vocab.value_base + obs.feedback
    tokens[:, 2::2] = np.where(asked, vocab.index_base + obs.queries, PAD)  #q_t at position 2t
    tokens[:, 3::2] = np.where(asked, fb, PAD)  #f_t at position 2t + 1, including the final feedback
    return tokens


def decode(tokens: np.ndarray, vocab: Vocab, cfg: EnvConfig) -> Observation:
    size, length = tokens.shape
    _check(length >= 2 and length % 2 == 0, f"sequence length {length} is not 2 + 2 * k_max")
    n = tokens[:, 0] - SIZE_BASE + 1
    _check((n >= 1) & (n <= vocab.n_index), "position 0 is not a size token")
    q, f = tokens[:, 2::2], tokens[:, 3::2]
    asked = q != PAD
    steps = asked.sum(axis=1)
    _check(asked == (np.arange(q.shape[1])[None, :] < steps[:, None]), "queries are not a contiguous prefix")
    _check((f != PAD) == asked, "feedback and queries disagree on episode length")
    queries = np.where(asked, q - vocab.index_base, ENV_PAD)
    if cfg.feedback == "cmp":
        _check(tokens[:, 1] == T_MASK, "cmp sequence without T_MASK at position 1")
        target = None
        feedback = np.where(asked, f - CMP_BASE, ENV_PAD)
        _check(~asked | ((feedback >= 0) & (feedback <= 2)), "cmp feedback token outside the comparison codes")
    else:
        target = tokens[:, 1] - vocab.value_base
        _check((target >= 0) & (target < vocab.v), "position 1 is not a value token")
        feedback = np.where(asked, f - vocab.value_base, ENV_PAD)
        _check(~asked | ((feedback >= 0) & (feedback < vocab.v)), "val feedback token outside the value range")
    _check(~asked | ((queries >= 0) & (queries < n[:, None])), "index token outside [0, N)")
    rows = np.arange(size)
    last = feedback[rows, np.maximum(steps - 1, 0)]
    hit = (steps > 0) & (last == (EQUAL if target is None else target))
    cap = np.array([cfg.query_cap(int(m)) for m in n])
    return Observation(n=n, target=target, queries=queries, feedback=feedback, steps=steps,
                       done=hit | (steps >= cap))


def decision_positions(k_max: int) -> np.ndarray:
    return 2 * np.arange(1, k_max + 1) - 1  #decision t reads position 2t - 1: f_{t-1}, or [T or T_MASK] for t = 1


def decision_mask(obs: Observation) -> np.ndarray:
    size, k_max = obs.queries.shape
    mask = np.zeros((size, 2 + 2 * k_max), dtype=bool)
    mask[:, decision_positions(k_max)] = np.arange(k_max)[None, :] < obs.steps[:, None]  #actions actually taken
    return mask


def next_decision_position(obs: Observation) -> np.ndarray:
    return 2 * obs.steps + 1  #the last real token; steps <= k_max keeps it inside the sequence
