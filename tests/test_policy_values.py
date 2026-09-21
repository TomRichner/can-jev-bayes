"""Independent checks of the analytic, no-simulation policy-value audit."""

from fractions import Fraction

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import beta

from jevbandits.baselines import select_action
from jevbandits.policy_values import (
    POLICIES,
    beta_greater_probability,
    policy_action_probabilities,
    policy_value,
    prior_value_table,
)


@pytest.mark.parametrize("parameters", [
    (1, 1, 1, 1), (3, 4, 2, 5), (20, 1, 1, 20), (1, 20, 20, 1),
    (11, 9, 1, 1), (19, 3, 15, 8), (5, 12, 5, 12),
])
def test_beta_comparison_matches_quadrature(parameters):
    a, b, c, d = parameters
    expected = quad(lambda x: beta.pdf(x, a, b) * beta.cdf(x, c, d), 0, 1, epsabs=1e-13)[0]
    np.testing.assert_allclose(beta_greater_probability(a, b, c, d), expected, atol=2e-14, rtol=1e-12)
    assert beta_greater_probability(a, b, c, d) + beta_greater_probability(c, d, a, b) == pytest.approx(1)


def _rational_greedy_value(state, remaining):
    if remaining == 0:
        return Fraction(0)
    means = [Fraction(s + 1, s + f + 2) for s, f in state]
    selected = [i for i, m in enumerate(means) if m == max(means)]
    result = Fraction(0)
    for arm in selected:
        plus, minus = list(state), list(state)
        s, f = state[arm]
        plus[arm], minus[arm] = (s + 1, f), (s, f + 1)
        m = means[arm]
        result += (
            m + m * _rational_greedy_value(plus, remaining - 1)
            + (1 - m) * _rational_greedy_value(minus, remaining - 1)
        ) / len(selected)
    return result


@pytest.mark.parametrize("horizon", [0, 1, 2, 3, 4])
def test_greedy_value_matches_uncached_rational_history_enumeration(horizon):
    for state in [((0, 0), (0, 0)), ((10, 8), (0, 0)), ((2, 1), (4, 3))]:
        s, f = zip(*state, strict=True)
        expected = float(_rational_greedy_value(state, horizon))
        assert policy_value("greedy", horizon, s, f) == pytest.approx(expected, abs=1e-14)


def test_analytic_tiny_prior_values():
    for policy in POLICIES:
        assert policy_value(policy, 0) == 0
        assert policy_value(policy, 1) == pytest.approx(0.5)
    assert policy_value("random", 2) == pytest.approx(1)
    assert policy_value("greedy", 2) == pytest.approx(13 / 12)
    assert policy_value("ts", 2) == pytest.approx(37 / 36)
    assert policy_value("exact", 2) == pytest.approx(13 / 12)


@pytest.mark.parametrize("horizon", [5, 10, 20])
def test_optimal_dominates_every_integrated_policy(horizon):
    optimal = policy_value("exact", horizon)
    for policy in POLICIES:
        assert policy_value(policy, horizon) <= optimal + 2e-13
    assert policy_value("random", horizon) == pytest.approx(horizon / 2)


def test_independent_optimal_recursion_matches_baseline_dp():
    from jevbandits.baselines import exact_q

    for horizon in [1, 2, 5, 10, 20]:
        assert policy_value("exact", horizon) == pytest.approx(
            max(exact_q((0, 0), (0, 0), horizon)), abs=2e-13
        )


@pytest.mark.parametrize("policy", [p for p in POLICIES if p != "ts"])
def test_action_probabilities_match_experimental_selector(policy):
    rng = np.random.default_rng(908)
    for remaining in [1, 2, 5, 10]:
        for _ in range(8):
            s, f = rng.integers(0, 8, (2, 2))
            probabilities = policy_action_probabilities(
                policy, tuple(s), tuple(f), horizon=10, remaining=remaining
            )
            observed = [select_action(policy, s, f, 11 - remaining, 10, rng) for _ in range(50)]
            assert all(probabilities[arm] > 0 for arm in observed)
            if probabilities[0] == 0.5:
                assert set(observed) == {0, 1}


def test_permutation_and_expected_regret_identity():
    for policy in POLICIES:
        a = policy_value(policy, 5, (1, 3), (4, 2))
        b = policy_value(policy, 5, (3, 1), (2, 4))
        assert a == b
    for row in prior_value_table((5,)):
        assert row["bayesian_regret"] == pytest.approx(10 / 3 - row["expected_reward"])
        assert row["optimality_gap"] >= -1e-14


def test_expected_sum_of_local_bellman_losses_equals_policy_value_gap():
    # Independent uncached enumeration of posterior-predictive histories.
    from jevbandits.baselines import exact_q

    def expected_loss(policy, state, remaining):
        if remaining == 0:
            return 0.0
        s, f = zip(*state, strict=True)
        p = policy_action_probabilities(policy, s, f, horizon=4, remaining=remaining)
        q = exact_q(s, f, remaining)
        result = 0.0
        for arm in range(2):
            mean = (s[arm] + 1) / (s[arm] + f[arm] + 2)
            plus, minus = list(state), list(state)
            plus[arm], minus[arm] = (s[arm] + 1, f[arm]), (s[arm], f[arm] + 1)
            result += p[arm] * (
                max(q) - q[arm]
                + mean * expected_loss(policy, plus, remaining - 1)
                + (1 - mean) * expected_loss(policy, minus, remaining - 1)
            )
        return result

    for policy in POLICIES:
        assert expected_loss(policy, [(0, 0), (0, 0)], 4) == pytest.approx(
            policy_value("exact", 4) - policy_value(policy, 4), abs=2e-14
        )
