import math
from fractions import Fraction

import numpy as np
import pytest

from cvar import cvar, quantile, surrogate_weights, tail_size, tail_stats


def _brute(costs: list[int], alpha: Fraction) -> tuple[Fraction, Fraction, list[Fraction]]:
    size = len(costs)
    k = alpha * size
    ru = {nu: nu + sum((max(c - nu, 0) for c in costs), Fraction(0)) / k for nu in map(Fraction, set(costs))}
    value = min(ru.values())  #convex, piecewise linear, kinks at the data points: the minimum is at one of them
    nu = Fraction(sorted(costs, reverse=True)[math.ceil(k) - 1])
    assert ru[nu] == value  #the upper alpha-quantile attains the Rockafellar-Uryasev minimum
    return value, nu, [(nu - c) / k if c > nu else Fraction(0) for c in costs]


def _check(costs: list[int], alpha: float, exact: Fraction | None = None) -> None:
    value, nu, weights = _brute(costs, Fraction(str(alpha)) if exact is None else exact)
    arr = np.array(costs, dtype=np.float64)
    stats = tail_stats(arr, alpha)
    assert abs(cvar(arr, alpha) - float(value)) < 1e-12 and abs(stats.cvar - float(value)) < 1e-12
    assert quantile(arr, alpha) == float(nu) == stats.nu
    w = surrogate_weights(arr, alpha)
    assert np.allclose(w, [float(x) for x in weights], rtol=0, atol=1e-12)
    assert np.array_equal(w != 0.0, arr > float(nu)) and np.all(w[arr <= float(nu)] == 0.0)  #exact zeros at ties
    assert np.array_equal(stats.tail, arr >= float(nu)) and np.array_equal(stats.nonzero, arr > float(nu))
    assert stats.strictly_worse == int((w != 0.0).sum()) == int(stats.nonzero.sum())
    assert stats.at_quantile == costs.count(int(nu)) and stats.zero_weight == (stats.strictly_worse == 0)
    assert abs(stats.cvar - (stats.nu - w.sum())) < 1e-12  #CVaR = nu + (1 / alpha B) sum (C - nu)^+


@pytest.mark.parametrize("costs, alpha, expected, nu, worse", [
    ([3] * 7 + [6] * 3, 0.1, Fraction(6), 6, 0),  #Section 8 flat region: the whole tail on the quantile atom
    ([3] * 12 + [6] * 3, 0.1, Fraction(6), 6, 0),  #same with k = 1.5: fractional mass, still all on the atom
    ([1, 2, 2, 2, 5, 9], 0.5, Fraction(16, 3), 2, 2),  #ties at the quantile with two episodes strictly worse
    (list(range(15)), 0.1, Fraction(14 + Fraction(1, 2) * 13, Fraction(3, 2)), 13, 1),  #k = 1.5: half of 13 counted
    ([5, 1, 4], 1.0, Fraction(10, 3), 1, 2),  #alpha = 1 is the mean
])
def test_hand_batches(costs: list[int], alpha: float, expected: Fraction, nu: int, worse: int) -> None:
    arr = np.array(costs, dtype=np.float64)
    stats = tail_stats(arr, alpha)
    assert abs(stats.cvar - float(expected)) < 1e-12 and stats.nu == nu and stats.strictly_worse == worse
    _check(costs, alpha)


def test_whole_tail_on_atom_gives_zero_weights() -> None:
    for costs, alpha in (([3] * 7 + [6] * 3, 0.1), ([3] * 12 + [6] * 3, 0.1), ([2] * 990 + [7] * 10, 0.01),
                         ([4] * 5, 0.4)):
        arr = np.array(costs, dtype=np.float64)
        stats = tail_stats(arr, alpha)
        assert stats.zero_weight and stats.strictly_worse == 0 and not stats.nonzero.any()
        assert np.all(surrogate_weights(arr, alpha) == 0.0)
        assert stats.cvar == float(max(costs)) and stats.tail.sum() == stats.at_quantile


def test_random_batches_against_brute_force() -> None:
    rng = np.random.default_rng(0)
    checked = 0
    for size in (1, 2, 7, 10, 15, 64, 100):
        for _ in range(20):
            costs = rng.integers(1, 8, size=size).tolist()  #few distinct values: many ties at the quantile
            for alpha in (0.05, 0.1, 0.25, 0.3, 0.5, 1.0):
                _check(costs, alpha)
            _check(costs, 1.0 / size, Fraction(1, size))  #the float 1 / size, against the exact alpha it stands for
            checked += 7
    assert checked > 900


def test_normalization_by_alpha_b() -> None:
    arr = np.array([10.0, 10.0, 8.0, 8.0, 8.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    w = surrogate_weights(arr, 0.5)  #k = 5: the five largest are 10, 10, 8, 8, 8, so nu = 8
    assert quantile(arr, 0.5) == 8.0 and np.allclose(w, [-0.4, -0.4] + [0.0] * 8)  #(8 - 10) / 5, not / 2
    w = surrogate_weights(arr, 0.6)  #k = 6: nu = 1, five episodes strictly worse
    assert np.allclose(w[:5], [-1.5, -1.5, -7 / 6, -7 / 6, -7 / 6]) and np.all(w[5:] == 0.0)


def test_properties() -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        arr = rng.integers(0, 30, size=int(rng.integers(1, 40))).astype(np.float64)
        assert abs(cvar(arr, 1.0) - arr.mean()) < 1e-12
        assert cvar(arr, 1.0 / arr.size) == arr.max() and quantile(arr, 1.0 / arr.size) == arr.max()
        values = [cvar(arr, a) for a in np.linspace(0.02, 1.0, 50)]
        assert all(x >= y - 1e-12 for x, y in zip(values, values[1:]))  #non-increasing in alpha
        assert cvar(arr, 0.1) >= arr.mean() - 1e-12


def test_tail_size_snapping() -> None:
    assert 0.07 * 100 != 7.0 and tail_size(0.07, 100) == 7.0
    assert 0.29 * 100 != 29.0 and tail_size(0.29, 100) == 29.0
    assert tail_size(0.1, 1000) == 100.0 and tail_size(0.05, 10_000) == 500.0 and tail_size(0.1, 15) == 1.5
    arr = np.arange(100, dtype=np.float64)
    assert quantile(arr, 0.07) == 93.0 and cvar(arr, 0.07) == 96.0  #without snapping ceil(k) = 8 and nu = 92


def test_bad_inputs() -> None:
    arr = np.array([1.0, 2.0])
    for alpha in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            cvar(arr, alpha)
    for bad in (np.array([]), np.array([[1.0, 2.0]]), np.array([1.0, np.nan]), np.array([np.inf])):
        for fn in (cvar, quantile, surrogate_weights, tail_stats):
            with pytest.raises(ValueError):
                fn(bad, 0.1)
