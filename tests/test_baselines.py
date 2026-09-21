"""Analytic, independent exhaustive, and sampling checks for bandit policies."""

from fractions import Fraction

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import beta

from jevbandits.baselines import (
    BASELINES,
    exact_q,
    ids_distribution,
    ids_statistics,
    knowledge_gradient,
    select_action,
)


def _exhaustive_fraction_q(successes, failures, h):
    """Uncached rational arithmetic reference for tiny trees."""
    if h == 0:
        return [Fraction(0)] * len(successes)
    result = []
    for arm in range(len(successes)):
        m = Fraction(1 + successes[arm], 2 + successes[arm] + failures[arm])
        sp, fm = list(successes), list(failures)
        sp[arm] += 1
        fm[arm] += 1
        result.append(
            m
            + m * max(_exhaustive_fraction_q(sp, failures, h - 1))
            + (1 - m) * max(_exhaustive_fraction_q(successes, fm, h - 1))
        )
    return result


def test_analytic_two_step_exploration_crossover():
    s, f = (10, 0), (8, 0)  # Beta(11,9), Beta(1,1).
    np.testing.assert_allclose(exact_q(s, f, 1), [0.55, 0.5])
    np.testing.assert_allclose(exact_q(s, f, 2), [1.1, 133 / 120])
    rng = np.random.default_rng(42)
    assert select_action("exact", s, f, 1, 1, rng) == 0
    assert select_action("exact", s, f, 1, 2, rng) == 1


@pytest.mark.parametrize("h", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("s,f", [((0, 0), (0, 0)), ((2, 0), (1, 4)), ((1, 2, 0), (2, 0, 1))])
def test_dp_agrees_with_exhaustive_rational_arithmetic(s, f, h):
    expected = [float(x) for x in _exhaustive_fraction_q(s, f, h)]
    np.testing.assert_allclose(exact_q(s, f, h), expected, rtol=0, atol=2e-14)


def test_dp_permutation_and_equal_arm_symmetry():
    s, f = (2, 8, 0), (4, 3, 1)
    order = [2, 0, 1]
    q = exact_q(s, f, 5)
    np.testing.assert_allclose(
        exact_q(tuple(s[i] for i in order), tuple(f[i] for i in order), 5),
        q[order], rtol=0, atol=2e-14,
    )
    same = exact_q((3, 3), (8, 8), 10)
    assert same[0] == same[1]


def test_dp_single_arm_value_and_return_not_shared():
    np.testing.assert_allclose(exact_q((2,), (3,), 20), [20 * 3 / 7])
    q = exact_q((0, 0), (0, 0), 2)
    q[0] = -99
    assert exact_q((0, 0), (0, 0), 2)[0] > 0


def test_ts_matches_independent_posterior_probability_integral():
    # Integral P(theta_0 > theta_1), independently evaluated by quadrature.
    s, f = (2, 1), (3, 4)
    p, error = quad(lambda x: beta.pdf(x, 3, 4) * beta.cdf(x, 2, 5), 0, 1)
    assert error < 1e-8
    rng = np.random.default_rng(741)
    n = 20_000
    observed = np.mean([select_action("ts", s, f, 1, 20, rng) == 0 for _ in range(n)])
    assert abs(observed - p) < 5 * np.sqrt(p * (1 - p) / n)


@pytest.mark.parametrize("policy", BASELINES)
def test_symmetric_first_decision_has_no_label_preference(policy):
    rng = np.random.default_rng(19)
    n = 3000
    choices = [select_action(policy, [0, 0, 0], [0, 0, 0], 1, 1, rng) for _ in range(n)]
    frequencies = np.bincount(choices, minlength=3) / n
    assert np.max(np.abs(frequencies - 1 / 3)) < 0.04


def test_bayes_ucb_uses_paper_c_zero_schedule():
    # t=2 means posterior medians; the old t+1 offset would select arm 1.
    s, f = [2, 0], [2, 0]
    rng = np.random.default_rng(52)
    actual = [select_action("bayes_ucb", s, f, 2, 20, rng) for _ in range(100)]
    assert set(actual) == {0, 1}
    # t=3 uses quantile 2/3, making the less concentrated arm optimistic.
    assert select_action("bayes_ucb", s, f, 3, 20, rng) == 1


def test_ucb_initialization_visits_every_unobserved_arm():
    s, f = np.zeros(5, dtype=int), np.zeros(5, dtype=int)
    rng = np.random.default_rng(27)
    chosen = []
    for t in range(1, 6):
        arm = select_action("ucb1", s, f, t, 10, rng)
        chosen.append(arm)
        f[arm] += 1
    assert len(set(chosen)) == 5


@pytest.mark.parametrize("policy", BASELINES)
def test_seeded_replay_and_input_immutability(policy):
    s, f = np.array([2, 1]), np.array([3, 0])
    a = [select_action(policy, s, f, 4, 10, np.random.default_rng(seed)) for seed in range(10)]
    b = [select_action(policy, s, f, 4, 10, np.random.default_rng(seed)) for seed in range(10)]
    assert a == b
    np.testing.assert_array_equal(s, [2, 1])
    np.testing.assert_array_equal(f, [3, 0])


@pytest.mark.parametrize("s,f", [([], []), ([1], [1, 2]), ([-1], [0]), ([0.5], [0]), ([np.nan], [0])])
def test_invalid_counts_rejected(s, f):
    with pytest.raises(ValueError):
        exact_q(s, f, 2)


@pytest.mark.parametrize("h", [-1, 1.5, 21])
def test_invalid_exact_horizon_rejected(h):
    with pytest.raises(ValueError):
        exact_q((0, 0), (0, 0), h)


def test_large_exact_problem_fails_instead_of_approximating():
    with pytest.raises(ValueError, match="state-space"):
        exact_q((0,) * 15, (0,) * 15, 10)


def test_knowledge_gradient_terminal_and_two_step_optimality():
    s, f = (10, 0), (8, 0)
    rng = np.random.default_rng(10)
    assert select_action("knowledge_gradient", s, f, 10, 10, rng) == 0
    assert select_action("knowledge_gradient", s, f, 9, 10, rng) == 1
    np.testing.assert_allclose(knowledge_gradient(s, f), [0, 7 / 120], atol=1e-14)
    for _ in range(50):
        s, f = rng.integers(0, 10, (2, 3))
        mean = (s + 1) / (s + f + 2)
        np.testing.assert_allclose(
            mean + knowledge_gradient(s, f) + mean.max(),
            exact_q(tuple(s), tuple(f), 2), atol=1e-14,
        )


def test_ids_uniform_prior_matches_analytic_information_and_regret():
    # Two independent uniforms: E max = 2/3, E[theta_i|i best]=2/3,
    # E[theta_i|other best]=1/3. Information = H(1/2) - H(2/3).
    stats = ids_statistics((0, 0), (0, 0), np.random.default_rng(123), samples=200_000)
    expected_information = np.log(2) + (2 / 3) * np.log(2 / 3) + (1 / 3) * np.log(1 / 3)
    np.testing.assert_allclose(stats["expected_regret"], [1 / 6] * 2, atol=0.002)
    np.testing.assert_allclose(stats["information_gain"], [expected_information] * 2, atol=0.002)
    np.testing.assert_allclose(stats["posterior_best_probabilities"], [0.5] * 2, atol=0.005)


def test_ids_mixture_optimization_against_dense_grid():
    delta, gain = np.array([0.1, 0.5]), np.array([0.001, 0.1])
    dist = ids_distribution(delta, gain, np.random.default_rng(1))
    assert 0 < dist[0] < 1
    assert dist.sum() == pytest.approx(1)
    achieved = (dist @ delta) ** 2 / (dist @ gain)
    grid = np.linspace(0, 1, 100_001)
    grid_ratio = (grid * delta[0] + (1 - grid) * delta[1]) ** 2 / (
        grid * gain[0] + (1 - grid) * gain[1]
    )
    assert achieved <= grid_ratio.min() + 1e-12


def test_ids_distribution_handles_zero_information_and_certain_optimum():
    rng = np.random.default_rng(1)
    np.testing.assert_array_equal(ids_distribution([0.2, 0.1], [0, 0], rng), [0, 1])
    np.testing.assert_array_equal(ids_distribution([0, 0.2], [0, 0.1], rng), [1, 0])
    np.testing.assert_array_equal(ids_distribution([0], [0], rng), [1])


def test_ids_statistics_finite_nonnegative_and_seeded():
    s, f = (100, 0, 3), (0, 100, 7)
    a = ids_statistics(s, f, np.random.default_rng(42))
    b = ids_statistics(s, f, np.random.default_rng(42))
    for key in ("information_gain", "expected_regret", "posterior_best_probabilities"):
        assert np.all(np.isfinite(a[key])) and np.all(a[key] >= 0)
        np.testing.assert_array_equal(a[key], b[key])
    assert sum(a["posterior_best_probabilities"]) == pytest.approx(1)
