"""Public observations only. Hidden environment parameters never enter this module."""

import hashlib
import json

import numpy as np
from scipy.stats import beta

MODEL = "jev-1.13.0"
TASK = {
    "environment": "Independent stationary Bernoulli bandit: each pull earns 0 or 1. Unknown arm success probabilities stay fixed. Only selected-arm outcomes are observed.",
    "prior": "Before observations each arm independently has a Beta(1,1) prior.",
    "protocol": "Each question is a separate decision with its own observation in its instructions. Arm labels carry no reward information. Remaining pulls include the current pull. There are no rewards after that budget ends.",
}
OBJECTIVES = {
    "neutral": "Choose the arm to pull now to maximize expected total reward over all remaining pulls, including this one.",
    "explore": "Choose the arm to pull now to maximize expected total reward over all remaining pulls, including this one. Pulling an uncertain arm may improve later choices. Explore only when the expected future benefit justifies the immediate reward tradeoff.",
    "regret": "Choose the arm to pull now to minimize expected cumulative regret over the remaining pulls relative to always pulling the arm with the highest fixed success probability. Those probabilities are unknown; observations may help later decisions.",
    "thompson": "Choose an arm using the principle of Thompson sampling: an arm should be selected in proportion to its posterior probability of having the highest success probability. Use the observations and Bayesian posterior information. The overall task is to maximize expected total reward over the remaining pulls.",
    "ucb": "Choose the arm with the largest supplied Bayes-UCB index, breaking equal-index ties arbitrarily. The index is computed from this episode's observations to balance reward and exploration.",
    "recommendation": "Choose the recommended arm. The supplied recommendation is from an exact Bayesian calculation maximizing expected total reward over the remaining pulls, given these observations and priors.",
}


def stable_seed(*parts):
    raw = json.dumps(parts, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def observation(
    successes,
    failures,
    remaining,
    representation="counts",
    q_values=None,
    indices=None,
    label_order=None,
):
    s, f = np.asarray(successes, dtype=float), np.asarray(failures, dtype=float)
    if (
        s.ndim != 1
        or len(s) < 2
        or s.shape != f.shape
        or not np.isfinite(s).all()
        or not np.isfinite(f).all()
        or np.any(s < 0)
        or np.any(f < 0)
        or np.any(s != np.floor(s))
        or np.any(f != np.floor(f))
    ):
        raise ValueError("invalid counts")
    s, f = s.astype(int), f.astype(int)
    if not isinstance(remaining, (int, np.integer)) or remaining < 1:
        raise ValueError("remaining must include at least one pull")
    if representation not in {
        "counts",
        "means",
        "bayes",
        "dp_values",
        "recommendation",
        "ucb",
    }:
        raise ValueError("unknown representation")
    order = list(range(len(s))) if label_order is None else list(label_order)
    if sorted(order) != list(range(len(s))):
        raise ValueError("invalid label permutation")
    a, b = s + 1, f + 1
    means = a / (a + b)
    sd = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    intervals = beta.ppf(np.array([0.025, 0.975])[:, None], a, b)
    arms = []
    for display, physical in enumerate(order):
        item = {
            "id": f"arm_{display:02d}",
            "successes": int(s[physical]),
            "failures": int(f[physical]),
        }
        if representation != "counts":
            item["posterior_mean"] = round(float(means[physical]), 6)
        if representation in {"bayes", "dp_values", "recommendation", "ucb"}:
            item.update(
                posterior_alpha=int(a[physical]),
                posterior_beta=int(b[physical]),
                posterior_sd=round(float(sd[physical]), 6),
                credible_interval_95=[
                    round(float(x), 6) for x in intervals[:, physical]
                ],
            )
        if representation == "dp_values":
            if q_values is None:
                raise ValueError("DP assistance requires computed action values")
            item["expected_total_reward_if_chosen_then_optimal"] = round(
                float(q_values[physical]), 8
            )
        if representation == "ucb":
            if indices is None:
                raise ValueError("UCB assistance requires computed indices")
            item["bayes_ucb_index"] = round(float(indices[physical]), 8)
        arms.append(item)
    result = {"remaining_pulls_including_this_one": int(remaining), "arms": arms}
    if representation != "counts":
        result["summary_definition"] = (
            "Posterior summaries concern the unknown success probability, not the noise of the next binary outcome. Counts and Beta(1,1) priors determine these summaries."
        )
    if representation == "recommendation":
        if q_values is None:
            raise ValueError("recommendation requires computed action values")
        best = np.flatnonzero(
            np.isclose(q_values, np.max(q_values), rtol=0, atol=1e-10)
        )
        result["recommended_arm"] = f"arm_{order.index(int(best[0])):02d}"
    return result


def question(obs, framing="explore"):
    return {
        "type": "choice",
        "instructions": {"question": OBJECTIVES[framing], "observation": obs},
        "criteria": {a["id"]: f"Pull {a['id']} now." for a in obs["arms"]},
    }


def validate_answer(answer, ids):
    if answer.get("type") != "choice" or answer.get("choice") not in ids:
        raise ValueError("invalid choice response")
    p = answer.get("probabilities", {})
    if set(p) != set(ids):
        raise ValueError("probability labels mismatch")
    values = np.array([p[x] for x in ids], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("non-finite or out-of-range probabilities")
    mass = float(values.sum())
    if mass <= 0 or abs(mass - 1) > max(0.02, 0.005 * len(ids)) + 1e-8:
        raise ValueError(f"invalid probability mass {mass}")
    chosen = list(ids).index(answer["choice"])
    # Preserve the backend choice even if rounded probabilities favor another
    # arm; the caller records this observed API inconsistency explicitly.
    confidence = float(answer.get("confidence", float("nan")))
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("invalid confidence")
    return chosen, values / mass, mass
