"""Independent analytic policy-value audit for two Beta-Bernoulli arms.

The recursion integrates outcomes and policy randomization, not hidden means
sampled by simulation. There is no Monte Carlo. Arithmetic is float64 and Beta
quantiles use SciPy. The implementation deliberately does not call the tested
baseline action selector or its optimal-value recursion.
"""

from functools import lru_cache
from numbers import Integral

import numpy as np
from scipy.special import betaln, logsumexp
from scipy.stats import beta

POLICIES = ("random", "greedy", "ts", "bayes_ucb", "ucb1", "knowledge_gradient", "exact")
MAX_HORIZON = 20
TIE_ATOL = 1e-12


def beta_greater_probability(alpha_x, beta_x, alpha_y, beta_y) -> float:
    r"""P(X>Y) for independent integer-parameter Beta random variables.

    Uses the finite sum

    sum_{i=0}^{alpha_x-1} B(alpha_y+i,beta_x+beta_y)
        / ((beta_x+i) B(1+i,beta_x) B(alpha_y,beta_y)).

    Terms are accumulated in log space. The formula is checked independently
    by numerical quadrature in the test suite. Only positive integer parameters
    are accepted; small floating-point excursions are clipped to [0,1].
    """
    parameters = (alpha_x, beta_x, alpha_y, beta_y)
    if any(not isinstance(x, Integral) or x < 1 for x in parameters):
        raise ValueError("Beta parameters must be positive integers")
    i = np.arange(alpha_x, dtype=float)
    log_terms = (
        betaln(alpha_y + i, beta_x + beta_y)
        - np.log(beta_x + i)
        - betaln(1 + i, beta_x)
        - betaln(alpha_y, beta_y)
    )
    return float(np.clip(np.exp(logsumexp(log_terms)), 0, 1))


def _validate(policy, successes, failures, horizon, remaining):
    if policy not in POLICIES:
        raise ValueError(f"Expected one of {POLICIES}")
    if (
        not isinstance(horizon, Integral) or not 0 <= horizon <= MAX_HORIZON
        or not isinstance(remaining, Integral) or not 0 <= remaining <= horizon
    ):
        raise ValueError(f"Require integer 0 <= remaining <= horizon <= {MAX_HORIZON}")
    if len(successes) != 2 or len(failures) != 2:
        raise ValueError("The analytic audit supports exactly two arms")
    if any(not isinstance(x, Integral) or x < 0 for x in (*successes, *failures)):
        raise ValueError("Observed counts must be nonnegative integers")
    return tuple(zip(successes, failures, strict=True))


def _next_state(state, arm, reward):
    updated = list(state)
    s, f = updated[arm]
    updated[arm] = (s + reward, f + 1 - reward)
    return tuple(sorted(updated))


def _greedy_probabilities(scores):
    scores = np.asarray(scores)
    selected = np.isclose(scores, scores.max(), rtol=0, atol=TIE_ATOL)
    return selected / selected.sum()


def _probabilities(policy, state, horizon, remaining):
    s = np.array([arm[0] for arm in state], dtype=float)
    f = np.array([arm[1] for arm in state], dtype=float)
    n, alpha = s + f, s + 1
    means = alpha / (n + 2)
    t = horizon - remaining + 1
    if policy == "random":
        return np.array([0.5, 0.5])
    if policy == "ts":
        p = beta_greater_probability(
            state[0][0] + 1, state[0][1] + 1,
            state[1][0] + 1, state[1][1] + 1,
        )
        return np.array([p, 1 - p])
    if policy == "greedy":
        scores = means
    elif policy == "bayes_ucb":
        scores = beta.ppf(1 - 1 / t, alpha, f + 1)
    elif policy == "ucb1":
        if np.any(n == 0):
            return (n == 0) / np.sum(n == 0)
        scores = s / n + np.sqrt(2 * np.log(t) / n)
    elif policy == "knowledge_gradient":
        improvement = np.zeros(2)
        for arm in range(2):
            other = means[1 - arm]
            after_success = max(other, (alpha[arm] + 1) / (n[arm] + 3))
            after_failure = max(other, alpha[arm] / (n[arm] + 3))
            improvement[arm] = max(
                0, means[arm] * after_success + (1 - means[arm]) * after_failure - max(means)
            )
        scores = means + (remaining - 1) * improvement
    else:
        scores = _q_values("exact", state, horizon, remaining)
    return _greedy_probabilities(scores)


def _q_values(policy, state, horizon, remaining):
    values = []
    for arm, (s, f) in enumerate(state):
        mean = (s + 1) / (s + f + 2)
        values.append(
            mean
            + mean * _value(policy, _next_state(state, arm, 1), horizon, remaining - 1)
            + (1 - mean) * _value(policy, _next_state(state, arm, 0), horizon, remaining - 1)
        )
    return np.array(values)


@lru_cache(maxsize=200_000)
def _value(policy, state, horizon, remaining):
    if remaining == 0:
        return 0.0
    q = _q_values(policy, state, horizon, remaining)
    if policy == "exact":
        return float(q.max())
    probabilities = _probabilities(policy, state, horizon, remaining)
    return float(probabilities @ q)


def policy_action_probabilities(
    policy, successes=(0, 0), failures=(0, 0), *, horizon, remaining=None
) -> np.ndarray:
    """Integrated policy choice probabilities in the supplied arm order.

    All non-TS policies use the same 1e-12 absolute index tie tolerance as the
    experimental baselines. TS integrates independent continuous Beta draws;
    probability-zero exact sample ties are immaterial. The numerical selector
    also treats sample differences <=1e-12 as ties, whose effect is below the
    precision reported for this audit.
    """
    remaining = horizon if remaining is None else remaining
    state = _validate(policy, successes, failures, horizon, remaining)
    if remaining == 0:
        raise ValueError("There is no action with zero decisions remaining")
    return _probabilities(policy, state, horizon, remaining)


def policy_value(
    policy, horizon, successes=(0, 0), failures=(0, 0), *, remaining=None
) -> float:
    """Bayesian expected total future reward for a fully specified policy.

    Default initial state is two independent Beta(1,1) priors. Optional observed
    counts permit a conditional audit; remaining includes the current pull.
    Horizon still sets the experiment clock for Bayes-UCB and UCB1. Canonical
    arm ordering memoizes symmetric states without changing the policy law.
    """
    remaining = horizon if remaining is None else remaining
    state = _validate(policy, successes, failures, horizon, remaining)
    return _value(policy, tuple(sorted(state)), horizon, remaining)


def prior_value_table(horizons=(5, 10, 20)) -> list[dict]:
    """Auditable prior expected reward, Bayesian regret, and optimality loss."""
    rows = []
    for horizon in horizons:
        optimal = policy_value("exact", horizon)
        for policy in POLICIES:
            reward = policy_value(policy, horizon)
            rows.append({
                "horizon": horizon,
                "policy": policy,
                "expected_reward": reward,
                "bayesian_regret": horizon * 2 / 3 - reward,
                "optimality_gap": optimal - reward,
            })
    return rows


def clear_policy_value_cache():
    _value.cache_clear()
