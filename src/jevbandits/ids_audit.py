"""Independent two-arm quadrature audit of Monte Carlo IDS information ratios.

This audits numerical approximation, not Jev or the scientific run protocol.
Full sensitivity runs are offline and resumable at fixture boundaries.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import quad
from scipy.special import betainc, kl_div, xlogy
from scipy.stats import beta

from .baselines import exact_q, ids_distribution, ids_statistics

QUAD_EPSABS = 1e-13
QUAD_EPSREL = 1e-11


def binary_entropy(probability):
    p = np.asarray(probability)
    return -xlogy(p, p) - xlogy(1 - p, 1 - p)


def information_ratio(distribution, expected_regret, information_gain) -> float:
    """Squared posterior expected regret per nat, with explicit boundary rules."""
    p, delta, gain = map(np.asarray, (distribution, expected_regret, information_gain))
    regret, information = float(p @ delta), float(p @ gain)
    if regret == 0:
        return 0.0  # Includes the defined 0/0 convention.
    if information <= 0:
        return float("inf")
    return regret * regret / information


def quadrature_ids_statistics(successes, failures) -> dict:
    r"""Two-arm Beta posterior IDS moments by one-dimensional quadrature.

    w_i = integral f_i(x) F_j(x) dx,
    c_ii = integral x f_i(x) F_j(x) dx,
    c_ij = integral f_i(x) m_j I_x(alpha_j+1,beta_j) dx.

    c_ij denotes E[theta_j * 1(best=i)]. Conditional Bernoulli means are
    c_ij/w_i. The information gain is equivalently conditional KL or predictive
    entropy minus weighted conditional entropy. KL is used for the returned
    result because subtraction of almost equal entropies loses precision when
    one best-arm category is rare. This is numerical quadrature, not symbolic
    exactness; error estimates and consistency residuals are returned.
    """
    s, f = np.asarray(successes, dtype=float), np.asarray(failures, dtype=float)
    if (
        s.shape != (2,) or f.shape != (2,)
        or not np.all(np.isfinite(s)) or not np.all(np.isfinite(f))
        or np.any(s < 0) or np.any(f < 0)
        or np.any(s != np.floor(s)) or np.any(f != np.floor(f))
    ):
        raise ValueError("Require two nonnegative integer success/failure counts")
    a, b = s + 1, f + 1
    mean = a / (a + b)
    weights, moments = np.zeros(2), np.zeros((2, 2))
    error_estimates = []

    def integrate(function):
        value, error = quad(function, 0, 1, epsabs=QUAD_EPSABS, epsrel=QUAD_EPSREL, limit=250)
        error_estimates.append(error)
        return value

    for i in range(2):
        j = 1 - i
        weights[i] = integrate(lambda x, i=i, j=j: beta.pdf(x, a[i], b[i]) * betainc(a[j], b[j], x))
        moments[i, i] = integrate(lambda x, i=i, j=j: x * beta.pdf(x, a[i], b[i]) * betainc(a[j], b[j], x))
        moments[i, j] = integrate(
            lambda x, i=i, j=j: beta.pdf(x, a[i], b[i]) * mean[j] * betainc(a[j] + 1, b[j], x)
        )
    mass_residual = float(weights.sum() - 1)
    moment_residual = moments.sum(axis=0) - mean
    if abs(mass_residual) > 1e-10 or np.max(np.abs(moment_residual)) > 1e-10:
        raise ArithmeticError("Quadrature moment consistency failed")
    if np.any(weights <= 0):
        raise ArithmeticError("A best-arm probability was numerically lost; use higher precision")
    # Correct only quadrature-sized integration residuals to preserve moments.
    weights = weights / weights.sum()
    moments = moments * (mean / moments.sum(axis=0))[None, :]
    conditional_mean = moments / weights[:, None]
    if np.any(conditional_mean < -1e-9) or np.any(conditional_mean > 1 + 1e-9):
        raise ArithmeticError("Conditional mean outside [0,1]; quadrature is unreliable")
    conditional_mean = np.clip(conditional_mean, 0, 1)
    entropy_information = binary_entropy(mean) - weights @ binary_entropy(conditional_mean)
    # Sum of two generalized KL terms equals Bernoulli KL and is nonnegative.
    conditional_kl = kl_div(conditional_mean, mean) + kl_div(1 - conditional_mean, 1 - mean)
    information = weights @ conditional_kl
    expected_best = np.trace(moments)
    delta = expected_best - mean
    if np.any(delta < -1e-10):
        raise ArithmeticError("Negative expected regret beyond numerical tolerance")
    return {
        "expected_regret": np.maximum(0, delta),
        "information_gain": np.maximum(0, information),
        "posterior_best_probabilities": weights,
        "posterior_mean": mean,
        "conditional_first_moments": moments,
        "conditional_means": conditional_mean,
        "entropy_information": entropy_information,
        "quadrature_max_error_estimate": max(error_estimates),
        "raw_probability_mass_residual": mass_residual,
        "raw_first_moment_max_residual": float(np.max(np.abs(moment_residual))),
        "entropy_kl_max_difference": float(np.max(np.abs(entropy_information - information))),
    }


def sensitivity_fixture(fixture_id, successes, failures, sample_counts, repeats, seed=20260920):
    """Assess Monte Carlo policy mixtures under quadrature-derived objectives."""
    from .prompts import stable_seed

    reference = quadrature_ids_statistics(successes, failures)
    delta, information = reference["expected_regret"], reference["information_gain"]
    optimal_mixture = ids_distribution(delta, information, np.random.default_rng(
        stable_seed(seed, "ids_audit", fixture_id, "quadrature_optimizer")
    ))
    optimal_ratio = information_ratio(optimal_mixture, delta, information)
    q = exact_q(tuple(successes), tuple(failures), 10)
    records = []
    for samples in sample_counts:
        for repetition in range(repeats):
            rng = np.random.default_rng(stable_seed(seed, "ids_audit", fixture_id, samples, repetition))
            estimate = ids_statistics(successes, failures, rng, samples=samples)
            mixture = ids_distribution(estimate["expected_regret"], estimate["information_gain"], rng)
            achieved = information_ratio(mixture, delta, information)
            raw_excess = achieved - optimal_ratio
            if raw_excess < -1e-9 * max(1, abs(optimal_ratio)):
                raise ArithmeticError("Estimated policy beats the reference objective beyond tolerance")
            records.append({
                "samples": samples, "repetition": repetition,
                "mixture": mixture.tolist(), "true_information_ratio": achieved,
                "raw_excess_information_ratio": raw_excess,
                "excess_information_ratio": max(0.0, raw_excess),
                "rare_best_category": bool(samples * reference["posterior_best_probabilities"].min() < 1),
                "missed_best_category": bool(np.any(estimate["posterior_best_probabilities"] == 0)),
                "estimated_all_information_zero": bool(np.max(estimate["information_gain"]) <= 1e-15),
                "bayesian_h10_one_step_loss": float(q.max() - mixture @ q),
            })
    serializable_reference = {key: value.tolist() if isinstance(value, np.ndarray) else value
                              for key, value in reference.items()}
    return {
        "fixture_id": fixture_id, "successes": list(successes), "failures": list(failures),
        "reference": serializable_reference,
        "optimal_information_ratio": optimal_ratio,
        "reference_mixture": optimal_mixture.tolist(),
        "reference_bayesian_h10_one_step_loss": float(q.max() - optimal_mixture @ q),
        "results": records,
    }


def write_report(output_dir):
    """Generate a descriptive sensitivity report from completed fixtures only."""
    from .experiments import read_jsonl

    output_dir = Path(output_dir)
    fixtures = read_jsonl(output_dir / "sensitivity.jsonl")
    rows = [{"fixture_id": fixture["fixture_id"], **row}
            for fixture in fixtures for row in fixture["results"]]
    if not rows:
        return
    lines = [
        "# Monte Carlo IDS numerical sensitivity", "",
        f"Completed fixtures: **{len(fixtures)}**; Monte Carlo replicates: **{len(rows)}**.", "",
        ("Excess information ratio is evaluated with quadrature-derived true posterior moments, "
        "relative to the minimum of that same objective. It is not total-variation distance "
        "to an arbitrary optimal mixture. Fixtures may repeat posterior states; these are "
        "descriptive numerical diagnostics, not independent environment effect estimates."), "",
        "| Posterior draws | Median excess | 95th percentile | Mean excess | Maximum excess | Missing category | Rare-category rows |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for samples in sorted({row["samples"] for row in rows}):
        group = [row for row in rows if row["samples"] == samples]
        excess = np.array([row["excess_information_ratio"] for row in group])
        lines.append(
            f"| {samples} | {np.median(excess):.8g} | {np.quantile(excess, .95):.8g} | "
            f"{excess.mean():.8g} | {excess.max():.8g} | "
            f"{np.mean([r['missed_best_category'] for r in group]):.2%} | "
            f"{np.mean([r['rare_best_category'] for r in group]):.2%} |"
        )
    lines.extend(["", ("A rare category has fewer than one expected posterior sample "
                  "at the indicated draw count. Missing categories can force an "
                  "empirical zero-information fallback even when the true posterior "
                  "retains uncertainty."), "", "## Largest excess-ratio cases at 2,048 draws", ""])
    worst = sorted((r for r in rows if r["samples"] == 2048),
                   key=lambda r: r["excess_information_ratio"], reverse=True)[:10]
    for row in worst:
        lines.append(f"- {row['fixture_id']}, replicate {row['repetition']}: "
                     f"excess {row['excess_information_ratio']:.8g}; "
                     f"mixture {row['mixture']}; missing category {row['missed_best_category']}.")
    maximum_error = max(x["reference"]["quadrature_max_error_estimate"] for x in fixtures)
    maximum_residual = max(x["reference"]["raw_first_moment_max_residual"] for x in fixtures)
    lines.extend(["", (f"Maximum quadrature error estimate: {maximum_error:.3g}. "
                  f"Maximum raw first-moment residual: {maximum_residual:.3g}."), "",
                  ("The records also contain expected one-step Bayes-optimal decision loss at "
                  "horizon 10. That is a separate planning objective; minimizing the IDS "
                  "information ratio need not minimize this loss."), ""])
    (output_dir / "report.md").write_text("\n".join(lines))


def run_sensitivity(output_dir, sample_counts=(128, 2048, 32768), repeats=30, fixture_count=100):
    from .experiments import append_json, fixtures, read_jsonl

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not 1 <= fixture_count <= 100 or repeats < 1 or any(n < 2 for n in sample_counts):
        raise ValueError("Require 1..100 fixtures per study, positive repetitions, and sample sizes >=2")
    study_fixtures = [(f"{experiment}:{index}", successes, failures)
                      for experiment in ["e1_horizon", "e2_assistance"]
                      for index, (successes, failures) in enumerate(fixtures(experiment, fixture_count))]
    metadata = {
        "purpose": "posthoc numerical audit of frozen MC IDS, no API calls",
        "sample_counts": list(sample_counts), "repetitions": repeats,
        "fixtures_per_study": fixture_count, "master_seed": 20260920,
        "fixture_sha256": hashlib.sha256(json.dumps(study_fixtures).encode()).hexdigest(),
        "quadrature_epsabs": QUAD_EPSABS, "quadrature_epsrel": QUAD_EPSREL,
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ["ids_audit.py", "baselines.py", "experiments.py", "prompts.py"]},
    }
    manifest = output_dir / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError("Audit configuration/source changed; choose a new output directory")
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    output = output_dir / "sensitivity.jsonl"
    done = {row["fixture_id"] for row in read_jsonl(output)}
    start = perf_counter()
    for index, (fixture_id, successes, failures) in enumerate(study_fixtures):
        if fixture_id in done:
            continue
        result = sensitivity_fixture(fixture_id, successes, failures, sample_counts, repeats)
        append_json(output, [result])
        print(json.dumps({"completed_fixture": fixture_id, "position": index + 1,
                          "total": len(study_fixtures), "elapsed_seconds": round(perf_counter() - start, 2)}), flush=True)
    write_report(output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/overnight_v2/ids_audit")
    parser.add_argument("--samples", type=int, nargs="+", default=[128, 2048, 32768])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--fixtures-per-study", type=int, default=100)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        write_report(args.output_dir)
    else:
        run_sensitivity(args.output_dir, args.samples, args.repeats, args.fixtures_per_study)


if __name__ == "__main__":
    main()
