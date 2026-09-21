"""Finite-horizon average-productivity index by stopping-value calibration.

This is a horizon-aware multi-arm heuristic, not the joint-state Bayesian
optimum. Each index solves one unknown arm against a constant alternative.
Run the optional posthoc offline panel with ``python -m jevbandits.finite_index``.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from numbers import Integral
from pathlib import Path
from time import perf_counter

import numpy as np

DEFAULT_TOLERANCE = 1e-6
POLICY_NAME = "finite_ap_index"


@dataclass(frozen=True, slots=True)
class IndexBracket:
    """Index enclosure from monotone stopping calibration, in float64."""

    lower: float
    upper: float
    midpoint: float
    iterations: int
    tolerance: float


def _validate(a, b, horizon):
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        raise ValueError("Beta posterior parameters must be finite and positive")
    if not isinstance(horizon, Integral) or not 1 <= horizon <= 1000:
        raise ValueError("Require integer 1 <= horizon <= 1000")


def _posterior_triangles(a, b, horizon):
    return [(a + np.arange(depth + 1)) / (a + b + depth) for depth in range(horizon)]


def _continuation_value(triangles, safe_reward):
    # At depth j, element s denotes s successes and j-s failures.
    next_value = np.zeros(len(triangles) + 1)
    for mean in reversed(triangles):
        continuation = (
            mean - safe_reward
            + mean * next_value[1:]
            + (1 - mean) * next_value[:-1]
        )
        next_value = np.maximum(0, continuation)
    return float(continuation[0])  # Starting the unknown arm is compulsory.


def stopping_excess(a, b, horizon, safe_reward) -> float:
    """Excess value of starting this arm, with optimal later retirement.

    Positive values favor trying the unknown arm over the constant reward.
    Retirement permits the constant reward on every remaining pull. This is
    C_h, not max(0,C_h), so it is negative above the reservation index.
    """
    _validate(a, b, horizon)
    if not np.isfinite(safe_reward) or not 0 <= safe_reward <= 1:
        raise ValueError("safe_reward must be in [0,1]")
    return _continuation_value(_posterior_triangles(a, b, horizon), safe_reward)


@lru_cache(maxsize=200_000)
def finite_ap_index(a, b, horizon, tolerance=DEFAULT_TOLERANCE) -> IndexBracket:
    """Approximate the single-arm finite-horizon AP index to a bracket width.

    Bracket endpoints satisfy C_h(lower)>=0 and C_h(upper)<=0 up to ordinary
    floating-point rounding. The maximum midpoint error is half the final
    bracket width; no global multi-arm optimality is implied. Only final
    brackets are cached, not triangular work arrays.
    """
    _validate(a, b, horizon)
    if not np.isfinite(tolerance) or not 1e-12 <= tolerance <= 1e-2:
        raise ValueError("Require tolerance in [1e-12,1e-2]")
    mean = a / (a + b)
    if horizon == 1:
        return IndexBracket(mean, mean, mean, 0, tolerance)
    lower, upper, iterations = mean, 1.0, 0
    triangles = _posterior_triangles(a, b, horizon)
    while upper - lower > tolerance:
        midpoint = (lower + upper) / 2
        if _continuation_value(triangles, midpoint) >= 0:
            lower = midpoint
        else:
            upper = midpoint
        iterations += 1
    return IndexBracket(lower, upper, (lower + upper) / 2, iterations, tolerance)


def select_index_action(successes, failures, remaining, rng, tolerance=DEFAULT_TOLERANCE):
    """Return arm and numerical diagnostics; uniform midpoint-index ties.

    Identical posterior states produce identical cached brackets. Distinct
    states can have overlapping enclosures; these decisions are flagged for
    tolerance sensitivity analysis rather than silently declared exact ranks.
    """
    s, f = np.asarray(successes), np.asarray(failures)
    if (
        s.ndim != 1 or s.size == 0 or s.shape != f.shape
        or not np.all(np.isfinite(s)) or not np.all(np.isfinite(f))
        or np.any(s < 0) or np.any(f < 0)
        or np.any(s != np.floor(s)) or np.any(f != np.floor(f))
    ):
        raise ValueError("Counts must be equal nonempty nonnegative integer vectors")
    brackets = [finite_ap_index(int(si) + 1, int(fi) + 1, remaining, tolerance)
                for si, fi in zip(s, f, strict=True)]
    values = np.array([x.midpoint for x in brackets])
    ties = np.flatnonzero(np.isclose(values, values.max(), atol=1e-12, rtol=0))
    action = int(rng.choice(ties))
    overlapping_distinct = [
        i for i, bracket in enumerate(brackets)
        if i != action and bracket.upper >= brackets[action].lower
        and (s[i], f[i]) != (s[action], f[action])
    ]
    return action, {
        "index_midpoints": values.tolist(),
        "index_lower": [x.lower for x in brackets],
        "index_upper": [x.upper for x in brackets],
        "index_ambiguous_ranking": bool(overlapping_distinct),
    }


def run_offline(run_dir, experiments, tolerance=DEFAULT_TOLERANCE, n_override=None):
    """Additional posthoc policy on frozen worlds; never changes core manifest."""
    from .experiments import Episode, append_json, read_jsonl

    run_dir = Path(run_dir)
    source_manifest = run_dir / "manifest.json"
    original = json.loads(source_manifest.read_text())
    for filename in ["experiments.py", "baselines.py", "prompts.py"]:
        actual = hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()
        if actual != original["source_sha256"][filename]:
            raise ValueError(f"Frozen scientific source changed: {filename}")
    if n_override is not None and (not isinstance(n_override, Integral) or n_override < 1):
        raise ValueError("Episode cap must be a positive integer")
    metadata = {
        "policy": POLICY_NAME,
        "status": "posthoc exploratory classical comparator; no Jev calls",
        "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        "method_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tolerance": tolerance,
        "tie_rule": "uniform among midpoint maxima within absolute 1e-12",
        "rank_uncertainty": "flag overlapping brackets from distinct posterior states",
        "arithmetic": "float64",
        "prior": "independent Beta(1,1)",
    }
    metadata_path = run_dir / "index_manifest.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("AP-index provenance changed; use a separate run directory")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    output = run_dir / "episodes_index.jsonl"
    done = {(x["episode_id"], x["policy"]) for x in read_jsonl(output)}
    start, completed = perf_counter(), 0
    for experiment in experiments:
        design = original["online"][experiment]
        n = design["n"] if n_override is None else min(n_override, design["n"])
        for family in design["families"]:
            for k in design["ks"]:
                for index in range(n):
                    episode = Episode(experiment, family, k, index, design["horizon"], POLICY_NAME)
                    if (episode.episode_id, POLICY_NAME) in done:
                        continue
                    for turn in range(1, episode.horizon + 1):
                        action, diagnostics = select_index_action(
                            episode.s, episode.f, episode.horizon - turn + 1,
                            episode.rng(turn), tolerance,
                        )
                        episode.step(turn, action)
                        episode.trace[-1].update(diagnostics)
                    record = episode.summary()
                    record["posthoc_exploratory"] = True
                    record["index_tolerance"] = tolerance
                    record["index_ambiguous_ranking_fraction"] = float(np.mean([
                        x["index_ambiguous_ranking"] for x in episode.trace
                    ]))
                    append_json(output, [record])
                    completed += 1
                    if completed % 10 == 0:
                        print(json.dumps({
                            "experiment": experiment, "completed_this_invocation": completed,
                            "elapsed_seconds": round(perf_counter() - start, 2),
                            "index_cache": finite_ap_index.cache_info()._asdict(),
                        }), flush=True)
    print(json.dumps({"completed_this_invocation": completed,
                      "elapsed_seconds": perf_counter() - start}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="artifacts/overnight_v2")
    parser.add_argument("--experiments", nargs="+", default=["e3_exact_online", "e4_scaling", "e5_strategy"])
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--n", type=int, help="Optional cap for feasibility probes; completed episodes resume")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.benchmark:
        for a, b in [(1, 1), (10, 10), (19, 3)]:
            start = perf_counter()
            bracket = finite_ap_index(a, b, 100, args.tolerance)
            print(json.dumps({"a": a, "b": b, "horizon": 100,
                              "seconds": perf_counter() - start, **asdict(bracket)}))
    else:
        run_offline(args.run_dir, args.experiments, args.tolerance, args.n)


if __name__ == "__main__":
    main()
