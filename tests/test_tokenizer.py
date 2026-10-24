import dataclasses

import numpy as np
import pytest

from config import Config, EnvConfig, ModelConfig
from env import PAD as ENV_PAD, EpisodeBatch, Observation, SearchEnv, sample_batch
from tokenizer import (CMP_BASE, PAD, SIZE_BASE, T_MASK, Vocab, decision_mask, decision_positions, decode, encode,
                       make_vocab, next_decision_position)

VOCAB = make_vocab(Config())
SETTINGS = {
    "uniform": dict(distribution="uniform"),
    "mixture_trap": dict(distribution="mixture", adversarial="trap", p=0.3),
    "adversarial_clustered": dict(distribution="adversarial", adversarial="clustered"),
}


def _batch(rows: list[list[int]], target_idx: list[int], v: int = 256) -> EpisodeBatch:
    n = np.array([len(r) for r in rows])
    values = np.full((len(rows), int(n.max())), v, dtype=np.int64)
    for i, r in enumerate(rows):
        values[i, :len(r)] = r
    t = np.asarray(target_idx, dtype=np.int64)
    return EpisodeBatch(n=n, values=values, target_idx=t, target=values[np.arange(len(rows)), t],
                        adversarial=np.zeros(len(rows), dtype=bool))


def _trajectory(cfg: EnvConfig, batch: EpisodeBatch, seed: int) -> list[Observation]:
    rng = np.random.default_rng(seed)
    env = SearchEnv(cfg, batch)
    out = [env.observe()]
    while not env.done.all():
        env.step(rng.integers(0, batch.n))  #repeats included, so some rows reach the cap
        out.append(env.observe())
    return out


def _same(a: Observation, b: Observation) -> bool:
    for f in dataclasses.fields(Observation):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if (x is None) != (y is None) or (x is not None and not np.array_equal(x, y)):
            return False
    return True


@pytest.mark.parametrize("cap_mode", ["linear", "log"])
@pytest.mark.parametrize("setting", list(SETTINGS))
@pytest.mark.parametrize("feedback", ["cmp", "val"])
def test_round_trip(feedback: str, setting: str, cap_mode: str) -> None:
    cfg = EnvConfig(feedback=feedback, cap_mode=cap_mode, **SETTINGS[setting])
    batch = sample_batch(cfg, 400, np.random.default_rng(0))
    states = _trajectory(cfg, batch, 1)
    for obs in states:
        tokens = encode(obs, VOCAB)
        assert tokens.shape == (400, 2 + 2 * cfg.query_cap(cfg.n_max))
        assert _same(decode(tokens, VOCAB, cfg), obs)
    final = states[-1]
    assert final.done.all() and (final.steps == np.array([cfg.query_cap(int(n)) for n in final.n])).any()  #capped rows
    assert (final.steps < np.array([cfg.query_cap(int(n)) for n in final.n])).any()  #rows that hit


def test_hand_sequences() -> None:
    batch = _batch([[10, 20, 30, 40]], [2])
    size, index, value = (lambda n: SIZE_BASE + n - 1), (lambda i: VOCAB.index_base + i), (lambda x: VOCAB.value_base + x)
    less, equal, greater = CMP_BASE, CMP_BASE + 1, CMP_BASE + 2
    want = {
        "cmp": [size(4), T_MASK, index(0), less, index(3), greater, index(2), equal],
        "val": [size(4), value(30), index(0), value(10), index(3), value(40), index(2), value(30)],
    }
    for feedback, expected in want.items():
        cfg = EnvConfig(feedback=feedback)
        env = SearchEnv(cfg, batch)
        for q in (0, 3, 2):
            env.step(np.array([q]))
        tokens = encode(env.observe(), VOCAB)
        assert tokens[0, :8].tolist() == expected and np.all(tokens[0, 8:] == PAD)  #final feedback encoded, then PAD
        assert decode(tokens, VOCAB, cfg).done[0]


@pytest.mark.parametrize("setting", list(SETTINGS))
def test_cmp_never_carries_the_target(setting: str) -> None:
    cfg = EnvConfig(feedback="cmp", n_min=32, n_max=32, **SETTINGS[setting])  #fixed N so ranks transfer between arrays
    b1 = sample_batch(cfg, 300, np.random.default_rng(2))
    b2 = sample_batch(cfg, 300, np.random.default_rng(3))
    rows = np.arange(300)
    relabelled = EpisodeBatch(n=b1.n, values=b2.values, target_idx=b1.target_idx,  #same N and ranks, other values
                              target=b2.values[rows, b1.target_idx], adversarial=b1.adversarial)
    for c, val in ((cfg, False), (dataclasses.replace(cfg, feedback="val"), True)):
        s1, s2 = _trajectory(c, b1, 4), _trajectory(c, relabelled, 4)
        assert len(s1) == len(s2)
        differ = False
        for o1, o2 in zip(s1, s2):
            t1, t2 = encode(o1, VOCAB), encode(o2, VOCAB)
            if not val:
                assert np.array_equal(t1, t2)  #identical token sequences: nothing in them depends on the values
                assert np.all(t1[:, 1] == T_MASK) and not np.any(t1 >= VOCAB.value_base)
                assert not np.any(t1[:, 2:] == T_MASK)
            differ |= not np.array_equal(t1, t2)
        assert differ == val


def test_val_target_and_feedback_tokens() -> None:
    cfg = EnvConfig(feedback="val", **SETTINGS["mixture_trap"])
    batch = sample_batch(cfg, 300, np.random.default_rng(5))
    for obs in _trajectory(cfg, batch, 6):
        tokens = encode(obs, VOCAB)
        assert np.array_equal(tokens[:, 1], VOCAB.value_base + batch.target)
        asked = np.arange(obs.queries.shape[1])[None, :] < obs.steps[:, None]
        assert np.array_equal(tokens[:, 3::2][asked], VOCAB.value_base + obs.feedback[asked])
        assert not np.any(tokens == T_MASK)


@pytest.mark.parametrize("feedback", ["cmp", "val"])
def test_decision_positions(feedback: str) -> None:
    cfg = EnvConfig(feedback=feedback, **SETTINGS["uniform"])
    batch = sample_batch(cfg, 300, np.random.default_rng(7))
    k_max = cfg.query_cap(cfg.n_max)
    assert decision_positions(k_max).tolist() == list(range(1, 2 * k_max, 2))
    for obs in _trajectory(cfg, batch, 8):
        tokens = encode(obs, VOCAB)
        mask = decision_mask(obs)
        nxt = next_decision_position(obs)
        for i in range(300):
            steps = int(obs.steps[i])
            assert np.nonzero(mask[i])[0].tolist() == decision_positions(k_max)[:steps].tolist()
            for t in range(1, steps + 1):
                assert tokens[i, 2 * t] - VOCAB.index_base == obs.queries[i, t - 1]  #q_t right after its decision position
                read = tokens[i, 2 * t - 1]
                if t == 1:
                    assert read == (T_MASK if feedback == "cmp" else VOCAB.value_base + batch.target[i])
                else:
                    base = CMP_BASE if feedback == "cmp" else VOCAB.value_base
                    assert read - base == obs.feedback[i, t - 2]  #f_{t-1}
            assert nxt[i] == 2 * steps + 1 and tokens[i, nxt[i]] != PAD
            assert nxt[i] + 1 == tokens.shape[1] or np.all(tokens[i, nxt[i] + 1:] == PAD)


def test_vocab_layout() -> None:
    assert VOCAB.size == 389 and (PAD, T_MASK, CMP_BASE, SIZE_BASE) == (0, 1, 2, 5)
    ranges = [range(PAD, PAD + 1), range(T_MASK, T_MASK + 1), range(CMP_BASE, CMP_BASE + 3),
              range(SIZE_BASE, SIZE_BASE + VOCAB.n_index), range(VOCAB.index_base, VOCAB.index_base + VOCAB.n_index),
              range(VOCAB.value_base, VOCAB.value_base + VOCAB.v)]
    covered = [x for r in ranges for x in r]
    assert sorted(covered) == list(range(VOCAB.size))  #disjoint and complete
    for env in (EnvConfig(n_min=64), EnvConfig(n_min=4, n_max=8), EnvConfig(feedback="val", distribution="adversarial",
                                                                             adversarial="trap")):
        assert make_vocab(Config(env=env)) == VOCAB  #fixed across runs: C1 -> C2 continuation shares token ids
    assert make_vocab(Config(env=EnvConfig(n_max=32), model=ModelConfig(context_len=130))) == VOCAB


def test_decode_rejects_malformed() -> None:
    cmp_cfg, val_cfg = EnvConfig(), EnvConfig(feedback="val")
    batch = _batch([[10, 20, 30, 40]], [2])
    env = SearchEnv(cmp_cfg, batch)
    env.step(np.array([0]))
    env.step(np.array([3]))
    good = encode(env.observe(), VOCAB)
    decode(good, VOCAB, cmp_cfg)
    bad = []
    t = good.copy(); t[0, 1] = VOCAB.value_base + 30; bad.append((t, cmp_cfg))  #a target value under cmp
    bad.append((good, val_cfg))  #T_MASK where val needs a value
    t = good.copy(); t[0, 2], t[0, 3] = PAD, PAD; bad.append((t, cmp_cfg))  #gap before the second query
    t = good.copy(); t[0, 5] = PAD; bad.append((t, cmp_cfg))  #query without feedback
    t = good.copy(); t[0, 4] = VOCAB.index_base + 4; bad.append((t, cmp_cfg))  #index >= N
    t = good.copy(); t[0, 0] = T_MASK; bad.append((t, cmp_cfg))  #position 0 not a size token
    t = good.copy(); t[0, 3] = VOCAB.value_base + 1; bad.append((t, cmp_cfg))  #value token as cmp feedback
    bad.append((good[:, :-1], cmp_cfg))  #odd length
    for tokens, cfg in bad:
        with pytest.raises(ValueError):
            decode(tokens, VOCAB, cfg)


def test_encode_rejects_invalid_observations() -> None:
    obs = SearchEnv(EnvConfig(feedback="val"), _batch([[10, 20, 30, 40]], [2])).observe()
    with pytest.raises(ValueError):
        encode(dataclasses.replace(obs, queries=np.array([[4] + [ENV_PAD] * 63]), steps=np.array([1])), VOCAB)
    with pytest.raises(ValueError):
        encode(dataclasses.replace(obs, n=np.array([65])), VOCAB)
    with pytest.raises(ValueError):
        encode(dataclasses.replace(obs, target=np.array([256])), VOCAB)
    small = Vocab(n_index=64, v=128)
    with pytest.raises(ValueError):
        encode(obs, dataclasses.replace(small, v=20))  #target 30 outside a 20-value vocabulary
