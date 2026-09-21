"""Validation of one-arm stopping indices, separate from multi-arm optimality."""

from functools import lru_cache

import numpy as np
import pytest

from jevbandits.finite_index import (
    finite_ap_index,
    select_index_action,
    stopping_excess,
)


@pytest.mark.parametrize("a,b", [(1, 1), (11, 9), (1, 12), (20, 3)])
def test_one_pull_index_is_posterior_mean(a, b):
    result = finite_ap_index(a, b, 1)
    assert result.lower == result.upper == result.midpoint == a / (a + b)


@pytest.mark.parametrize("a,b", [(1, 1), (11, 9), (1, 12), (20, 3)])
def test_two_pull_index_matches_analytic_reservation_reward(a, b):
    mean, after_success = a / (a + b), (a + 1) / (a + b + 1)
    expected = mean * (1 + after_success) / (1 + mean)
    result = finite_ap_index(a, b, 2, 1e-10)
    assert result.lower - 1e-14 <= expected <= result.upper + 1e-14


def test_index_increases_with_horizon_and_uncertainty():
    values = [finite_ap_index(2, 3, h).midpoint for h in [1, 2, 5, 10, 20, 100]]
    assert np.all(np.diff(values) > 0)
    assert finite_ap_index(1, 1, 20).lower > finite_ap_index(20, 20, 20).upper


@pytest.mark.parametrize("a,b,h", [(1, 1, 100), (11, 9, 20), (2, 10, 5)])
def test_bisection_brackets_stop_value_root(a, b, h):
    result = finite_ap_index(a, b, h)
    assert result.upper - result.lower <= 1e-6
    assert stopping_excess(a, b, h, result.lower) >= -1e-12
    assert stopping_excess(a, b, h, result.upper) <= 1e-12
    tighter = finite_ap_index(a, b, h, 1e-10)
    assert result.lower - 1e-14 <= tighter.midpoint <= result.upper + 1e-14


def test_independent_full_known_safe_arm_dp_agrees_with_index_threshold():
    # This allows safe actions without permanent retirement. The stopping index
    # must still recover the first-action threshold in this special problem.
    @lru_cache(None)
    def value(a, b, h, safe):
        if h == 0:
            return 0.0
        m = a / (a + b)
        unknown = m + m * value(a + 1, b, h - 1, safe) + (1 - m) * value(a, b + 1, h - 1, safe)
        known = safe + value(a, b, h - 1, safe)
        return max(unknown, known)

    for a, b, h in [(1, 1, 2), (3, 5, 5), (11, 9, 6)]:
        index = finite_ap_index(a, b, h, 1e-10).midpoint
        for safe, direction in [(index - 1e-4, 1), (index + 1e-4, -1)]:
            m = a / (a + b)
            unknown = m + m * value(a + 1, b, h - 1, safe) + (1 - m) * value(a, b + 1, h - 1, safe)
            known = safe + value(a, b, h - 1, safe)
            assert direction * (unknown - known) > 0


def test_symmetric_ties_seeded_replay_and_input_immutability():
    successes, failures = np.array([0, 0]), np.array([0, 0])
    rng = np.random.default_rng(123)
    choices = [select_index_action(successes, failures, 10, rng)[0] for _ in range(1000)]
    assert abs(np.mean(choices) - 0.5) < 0.05
    a = select_index_action(successes, failures, 10, np.random.default_rng(18))
    b = select_index_action(successes, failures, 10, np.random.default_rng(18))
    assert a == b
    np.testing.assert_array_equal(successes, [0, 0])
    np.testing.assert_array_equal(failures, [0, 0])


def test_terminal_policy_is_greedy_and_declares_brackets():
    action, diagnostics = select_index_action([10, 0], [8, 0], 1, np.random.default_rng(1))
    assert action == 0
    assert diagnostics["index_lower"] == diagnostics["index_upper"] == [0.55, 0.5]
    assert diagnostics["index_ambiguous_ranking"] is False


@pytest.mark.parametrize("a,b,h,tol", [(0, 1, 2, 1e-6), (1, 1, 0, 1e-6), (1, 1, 2.5, 1e-6), (1, 1, 2, 0)])
def test_invalid_input(a, b, h, tol):
    with pytest.raises(ValueError):
        finite_ap_index(a, b, h, tol)
