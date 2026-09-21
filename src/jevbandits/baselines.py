"""Independent Beta(1,1)-Bernoulli policies and finite-horizon control.

Counts are observed successes/failures, excluding prior pseudo-counts. The
one-based ``t`` is the current decision, and ``horizon`` includes all decisions.
Bayes-UCB reproduces Kaufmann et al. (2012), Algorithm 1 with experimental c=0:
posterior quantile 1 - 1/t. This differs from their theorem's c >= 5 schedule.
"""

from functools import lru_cache
from math import comb
from numbers import Integral

import numpy as np
from scipy.special import xlogy
from scipy.stats import beta

BASELINES = ("random", "greedy", "ts", "bayes_ucb", "ucb1", "exact", "knowledge_gradient", "ids")
IDS_SAMPLES = 2048
TIE_ATOL = 1e-12
MAX_EXACT_HORIZON = 20
MAX_EXACT_STATE_BOUND = 250_000


def _counts(successes, failures):
    successes = np.asarray(successes, dtype=float)
    failures = np.asarray(failures, dtype=float)
    if (
        successes.ndim != 1
        or successes.size == 0
        or successes.shape != failures.shape
    ):
        raise ValueError("Counts must be nonempty, equal-length one-dimensional arrays")
    for counts in (successes, failures):
        if (
            not np.all(np.isfinite(counts))
            or np.any(counts < 0)
            or np.any(counts != np.floor(counts))
        ):
            raise ValueError("Counts must be finite nonnegative integers")
    return successes, failures


def _argmax_random(values, rng):
    """Seeded uniform tie-breaking, with only roundoff-sized absolute tolerance."""
    values = np.asarray(values)
    candidates = np.flatnonzero(np.isclose(values, values.max(), rtol=0, atol=TIE_ATOL))
    return int(rng.choice(candidates))


def select_action(name, successes, failures, t, horizon, rng) -> int:
    """Select a zero-based arm without accessing hidden reward parameters.

    UCB1 selects uniformly among unobserved arms, which induces a random initial
    permutation. Thereafter its index is empirical mean + sqrt(2 log(t)/n).
    Other policies do not force initialization. ``exact`` uses the number of
    decisions remaining, ``horizon - t + 1``. Historical observations may precede
    the experiment, so counts need not sum to ``t - 1``.
    """
    successes, failures = _counts(successes, failures)
    if (
        not isinstance(t, Integral)
        or not isinstance(horizon, Integral)
        or not 1 <= t <= horizon
    ):
        raise ValueError("Require integer 1 <= t <= horizon")
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline {name!r}; expected one of {BASELINES}")
    if name == "random":
        return int(rng.integers(successes.size))

    alpha, beta_parameter = successes + 1, failures + 1
    if name == "greedy":
        values = alpha / (alpha + beta_parameter)
    elif name == "ts":
        values = rng.beta(alpha, beta_parameter)
    elif name == "bayes_ucb":
        values = beta.ppf(1 - 1 / t, alpha, beta_parameter)
    elif name == "ucb1":
        n = successes + failures
        unobserved = np.flatnonzero(n == 0)
        if unobserved.size:
            return int(rng.choice(unobserved))
        values = successes / n + np.sqrt(2 * np.log(t) / n)
    elif name == "knowledge_gradient":
        values = alpha / (alpha + beta_parameter) + (horizon - t) * knowledge_gradient(
            successes, failures
        )
    elif name == "ids":
        statistics = ids_statistics(successes, failures, rng)
        distribution = ids_distribution(
            statistics["expected_regret"], statistics["information_gain"], rng
        )
        return int(rng.choice(successes.size, p=distribution))
    else:
        values = exact_q(tuple(successes), tuple(failures), horizon - t + 1)
    return _argmax_random(values, rng)


def knowledge_gradient(successes, failures) -> np.ndarray:
    """Expected one-observation improvement in the largest posterior mean.

    Selecting m + (h-1)*KG is a one-step lookahead approximation for h > 2,
    not the exact finite-horizon optimum. It is terminal-greedy and agrees with
    exact action rankings at h=2. See Russo and Van Roy (2018), Section 4.3.
    """
    successes, failures = _counts(successes, failures)
    alpha, total = successes + 1, successes + failures + 2
    mean = alpha / total
    current_best = mean.max()
    improvements = np.empty(len(mean))
    for arm in range(len(mean)):
        after_success, after_failure = mean.copy(), mean.copy()
        after_success[arm] = (alpha[arm] + 1) / (total[arm] + 1)
        after_failure[arm] = alpha[arm] / (total[arm] + 1)
        improvements[arm] = max(
            0.0,
            mean[arm] * after_success.max()
            + (1 - mean[arm]) * after_failure.max()
            - current_best,
        )
    return improvements


def _binary_entropy(p):
    return -xlogy(p, p) - xlogy(1 - p, 1 - p)


def ids_statistics(successes, failures, rng, samples=IDS_SAMPLES) -> dict:
    """Monte Carlo SampleIR estimates, Russo--Van Roy (2018), Algorithm 4.

    Draw ``samples`` independent joint posterior models. All expectations use
    that same empirical posterior so regret and information remain coherent.
    Information is I(optimal arm; next binary reward), measured in nats. It is
    computed analytically conditional on each posterior model: no extra reward
    draws. Unsampled optimal-arm categories have zero empirical probability;
    this finite-sample limitation must be acknowledged in reported results.
    """
    successes, failures = _counts(successes, failures)
    if not isinstance(samples, Integral) or samples < 2:
        raise ValueError("samples must be an integer >= 2")
    draws = rng.beta(successes + 1, failures + 1, size=(samples, len(successes)))
    winners = np.argmax(draws, axis=1)
    best_probability = np.bincount(winners, minlength=len(successes)) / samples
    predictive_mean = draws.mean(axis=0)
    expected_regret = np.mean(draws.max(axis=1)[:, None] - draws, axis=0)
    conditional_entropy = np.zeros(len(successes))
    for arm in np.flatnonzero(best_probability):
        conditional_mean = draws[winners == arm].mean(axis=0)
        conditional_entropy += best_probability[arm] * _binary_entropy(conditional_mean)
    information_gain = np.maximum(0, _binary_entropy(predictive_mean) - conditional_entropy)
    return {
        "expected_regret": expected_regret,
        "information_gain": information_gain,
        "posterior_best_probabilities": best_probability,
        "posterior_sample_mean": predictive_mean,
        "samples": samples,
    }


def ids_distribution(expected_regret, information_gain, rng) -> np.ndarray:
    """Minimize the estimated information ratio over all two-arm mixtures.

    Proposition 6 of Russo--Van Roy (2018) guarantees a minimizer supported on
    at most two arms. Enumerate each pure action and each feasible stationary
    point of (d_j+p(d_i-d_j))^2/(g_j+p(g_i-g_j)). Ties among distinct candidate
    minimizers are seeded random. Zero estimated regret has ratio zero,
    including 0/0; if all information vanishes choose minimum estimated regret.
    """
    delta = np.asarray(expected_regret, dtype=float)
    gain = np.asarray(information_gain, dtype=float)
    if (
        delta.ndim != 1 or delta.size == 0 or delta.shape != gain.shape
        or not np.all(np.isfinite(delta)) or not np.all(np.isfinite(gain))
        or np.any(delta < 0) or np.any(gain < 0)
    ):
        raise ValueError("Regret and information must be equal nonempty nonnegative vectors")
    k = len(delta)
    if gain.max() <= 1e-15:
        result = np.zeros(k)
        result[_argmax_random(-delta, rng)] = 1
        return result

    def ratio(d, g):
        if d == 0:
            return 0.0
        return d * d / g if g > 0 else float("inf")

    candidates = [(ratio(delta[i], gain[i]), i, i, 1.0) for i in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            a, b = delta[i] - delta[j], delta[j]
            c, d = gain[i] - gain[j], gain[j]
            if a * c != 0:
                p = (b * c - 2 * a * d) / (a * c)
                if 0 < p < 1:
                    candidates.append((ratio(b + a * p, d + c * p), i, j, p))
    values = np.array([candidate[0] for candidate in candidates])
    minimum = values.min()
    ties = np.flatnonzero(np.isclose(values, minimum, rtol=1e-10, atol=1e-14))
    _, i, j, p = candidates[int(rng.choice(ties))]
    result = np.zeros(k)
    result[i] += p
    result[j] += 1 - p
    return result


def _updated(state, arm, outcome):
    changed = list(state)
    success, failure = changed[arm]
    changed[arm] = (success + outcome, failure + 1 - outcome)
    # Values are invariant under arm permutation; canonicalization shares work.
    return tuple(sorted(changed))


def _q_value(state, arm, remaining):
    success, failure = state[arm]
    mean = (success + 1) / (success + failure + 2)
    if remaining == 1:
        return mean
    return (
        mean
        + mean * _value(_updated(state, arm, 1), remaining - 1)
        + (1 - mean) * _value(_updated(state, arm, 0), remaining - 1)
    )


@lru_cache(maxsize=300_000)
def _value(state, remaining):
    """Full-state Bellman value, cached after arm-label canonicalization."""
    if remaining == 0:
        return 0.0
    return max(_q_value(state, arm, remaining) for arm in range(len(state)))


def exact_q(successes: tuple, failures: tuple, remaining: int) -> np.ndarray:
    """Return finite-horizon Bayes-optimal action values in original arm order.

    This exhausts the Bellman recursion in float64, without simulation or index
    approximations. At zero remaining decisions all action values are defined
    as zero. A combinatorial bound and h <= 20 protect against accidental large
    jobs; unsupported problems fail explicitly rather than use approximations.
    The bound C(h+2K,2K) counts all allocations of up to h outcomes to 2K counts;
    reachable states form a subset. Cached values are scalar and bounded in
    number; every call returns a fresh array, safe for caller mutation.
    """
    successes, failures = _counts(successes, failures)
    if not isinstance(remaining, Integral) or remaining < 0:
        raise ValueError("remaining must be a nonnegative integer")
    if remaining > MAX_EXACT_HORIZON:
        raise ValueError(f"Exact solver supports remaining <= {MAX_EXACT_HORIZON}")
    k = len(successes)
    if comb(remaining + 2 * k, 2 * k) > MAX_EXACT_STATE_BOUND:
        raise ValueError("Exact problem exceeds the configured state-space bound")
    if remaining == 0:
        return np.zeros(k)
    state = tuple((int(s), int(f)) for s, f in zip(successes, failures, strict=True))
    return np.array([_q_value(state, arm, remaining) for arm in range(k)])


def clear_exact_cache():
    """Release cached Bellman values between unrelated experiment panels."""
    _value.cache_clear()


def exact_cache_info():
    """Expose cache diagnostics for provenance and feasibility checks."""
    return _value.cache_info()
