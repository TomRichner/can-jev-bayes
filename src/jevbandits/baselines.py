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
from scipy.stats import beta

BASELINES = ("random", "greedy", "ts", "bayes_ucb", "ucb1", "exact")
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
    else:
        values = exact_q(tuple(successes), tuple(failures), horizon - t + 1)
    return _argmax_random(values, rng)


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
