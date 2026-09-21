"""Analytic and quadrature consistency tests for the IDS approximation audit."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import beta

from jevbandits.baselines import ids_distribution
from jevbandits.ids_audit import (
    binary_entropy,
    information_ratio,
    quadrature_ids_statistics,
    sensitivity_fixture,
)
from jevbandits.policy_values import beta_greater_probability


def test_uniform_prior_matches_analytic_information_regret_and_moments():
    stats = quadrature_ids_statistics([0, 0], [0, 0])
    np.testing.assert_allclose(stats["posterior_best_probabilities"], [0.5, 0.5], atol=1e-14)
    np.testing.assert_allclose(stats["expected_regret"], [1 / 6] * 2, atol=1e-14)
    np.testing.assert_allclose(stats["information_gain"], [np.log(2) - binary_entropy(2 / 3)] * 2, atol=1e-14)
    np.testing.assert_allclose(stats["conditional_first_moments"], [[1 / 3, 1 / 6], [1 / 6, 1 / 3]], atol=1e-14)


@pytest.mark.parametrize("s,f", [([2, 1], [3, 4]), ([20, 0], [0, 20]), ([59, 1], [39, 0])])
def test_quadrature_matches_finite_beta_probability_and_predictive_moments(s, f):
    stats = quadrature_ids_statistics(s, f)
    p = beta_greater_probability(s[0] + 1, f[0] + 1, s[1] + 1, f[1] + 1)
    assert stats["posterior_best_probabilities"][0] == pytest.approx(p, abs=1e-13)
    np.testing.assert_allclose(stats["conditional_first_moments"].sum(axis=0), stats["posterior_mean"], atol=1e-14)
    assert np.all(stats["information_gain"] >= 0)
    assert stats["entropy_kl_max_difference"] < 1e-13
    # Independent identity: E[max] = integral_0^1 [1-F1(x)F2(x)] dx.
    expected_max = quad(lambda x: 1 - beta.cdf(x, s[0] + 1, f[0] + 1)
                        * beta.cdf(x, s[1] + 1, f[1] + 1), 0, 1, epsabs=1e-13)[0]
    np.testing.assert_allclose(stats["expected_regret"] + stats["posterior_mean"], expected_max, atol=2e-13)


def test_arm_permutation_equivariance():
    first = quadrature_ids_statistics([11, 1], [8, 3])
    second = quadrature_ids_statistics([1, 11], [3, 8])
    for key in ["expected_regret", "information_gain", "posterior_best_probabilities"]:
        np.testing.assert_allclose(first[key][::-1], second[key], atol=1e-13)


def test_all_uniform_prior_mixtures_are_equally_optimal_not_tv_identifiable():
    stats = quadrature_ids_statistics([0, 0], [0, 0])
    d, g = stats["expected_regret"], stats["information_gain"]
    values = [information_ratio([p, 1 - p], d, g) for p in [0, 0.25, 0.5, 1]]
    np.testing.assert_allclose(values, values[0], atol=1e-13)
    selected = ids_distribution(d, g, np.random.default_rng(1))
    assert information_ratio(selected, d, g) == pytest.approx(values[0])


def test_information_ratio_boundary_conventions():
    assert information_ratio([1, 0], [0, 1], [0, 1]) == 0
    assert information_ratio([1, 0], [1, 0], [0, 1]) == float("inf")


@pytest.mark.parametrize("s,f", [([0, 10], [1, 10]), ([10, 0], [8, 0]), ([2, 4], [1, 7])])
def test_reference_optimizer_dominates_independent_dense_mixture_grid(s, f):
    stats = quadrature_ids_statistics(s, f)
    d, g = stats["expected_regret"], stats["information_gain"]
    p = ids_distribution(d, g, np.random.default_rng(82))
    grid = np.linspace(0, 1, 10001)
    objective = (grid * d[0] + (1 - grid) * d[1]) ** 2 / (grid * g[0] + (1 - grid) * g[1])
    assert information_ratio(p, d, g) <= objective.min() + 1e-12


def test_bounded_sensitivity_smoke_is_seeded_and_scores_true_objective():
    a = sensitivity_fixture("test:fixture", [10, 0], [8, 0], [128, 2048], 2)
    b = sensitivity_fixture("test:fixture", [10, 0], [8, 0], [128, 2048], 2)
    assert a == b
    assert len(a["results"]) == 4
    for row in a["results"]:
        expected = information_ratio(row["mixture"], a["reference"]["expected_regret"],
                                     a["reference"]["information_gain"])
        assert row["true_information_ratio"] == expected
        assert row["excess_information_ratio"] >= 0
        assert row["bayesian_h10_one_step_loss"] >= -1e-12


def test_rare_best_category_diagnostic():
    result = sensitivity_fixture("test:rare", [20, 0], [0, 20], [128], 2)
    assert min(result["reference"]["posterior_best_probabilities"]) < 1e-10
    assert all(row["rare_best_category"] and row["missed_best_category"] for row in result["results"])
