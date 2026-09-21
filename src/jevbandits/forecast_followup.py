"""E7: separately ask predictive and best-arm probability questions.

This is not an action policy. Its targets are analytic conditional probabilities,
so squared-error/KL measure excess proper-scoring risk without noisy future labels.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad_vec
from scipy.special import xlogy
from scipy.stats import beta

from .client import JevClient
from .experiments import SEED, append_json, read_jsonl
from .prompts import MODEL, TASK, observation, stable_seed, validate_answer

EXPERIMENT = "e7_probability_forecasts"
CLIP = 1e-6


def best_probabilities(successes, failures):
    """Independent Beta best-arm probabilities by adaptive vector quadrature."""
    a, b = np.asarray(successes) + 1, np.asarray(failures) + 1
    k = len(a)

    def integrand(x):
        cdf = beta.cdf(x, a, b)
        density = beta.pdf(x, a, b)
        return density * np.array([np.prod(np.delete(cdf, i)) for i in range(k)])

    probabilities, error = quad_vec(integrand, 0, 1, epsabs=1e-10, epsrel=1e-10)
    if (
        error > 1e-7
        or abs(probabilities.sum() - 1) > 1e-7
        or np.any(probabilities < -1e-12)
    ):
        raise ValueError("Posterior best-arm quadrature failed accuracy check")
    probabilities = np.maximum(0, probabilities)
    return probabilities / probabilities.sum()


def scoring(reference, prediction):
    q, p = np.asarray(reference, dtype=float), np.asarray(prediction, dtype=float)
    if q.shape != p.shape or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Invalid forecast")
    if abs(q.sum() - 1) > 1e-6 or abs(p.sum() - 1) > 1e-6:
        raise ValueError("Forecast distributions must sum to one")
    clipped = np.clip(p, CLIP, 1)
    clipped /= clipped.sum()
    return {
        "excess_brier": float(np.sum((p - q) ** 2)),
        "excess_log_loss": float(np.sum(xlogy(q, q) - xlogy(q, clipped))),
        "zero_prediction_positive_reference": int(np.sum((p == 0) & (q > 0))),
        "max_absolute_error": float(np.max(np.abs(p - q))),
    }


class ForecastClient(JevClient):
    """Use the same audited transport/ledger, adding the documented Noul type."""

    def _unpack(self, jobs, body, request_id, elapsed):
        if body.get("model") != MODEL:
            raise ValueError("Returned model version mismatch")
        if set(body.get("answers", {})) != {f"q{i}" for i in range(len(jobs))}:
            raise ValueError("Question response IDs mismatch")
        records = []
        for i, job in enumerate(jobs):
            answer = body["answers"][f"q{i}"]
            if job["question"]["type"] == "noul":
                value = answer.get("noul")
                if (
                    answer.get("type") != "noul"
                    or not isinstance(value, (float, int))
                    or not np.isfinite(value)
                    or not 0 <= value <= 1
                ):
                    raise ValueError("Invalid Noul probability")
                action, p, mass = None, [float(value), float(1 - value)], 1.0
            else:
                action, values, mass = validate_answer(
                    answer, list(job["question"]["criteria"])
                )
                p = values.tolist()
            record = {
                "action": action,
                "probabilities": p,
                "probability_mass": mass,
                "raw_answer": answer,
                "request_id": request_id,
                "latency_seconds": elapsed,
                "question_sha256": hashlib.sha256(
                    json.dumps(job["question"], sort_keys=True).encode()
                ).hexdigest(),
            }
            self.record(job["id"], record)
            records.append(record)
        return records


def build_jobs(n=100):
    jobs, metadata = [], []
    for k in [2, 3, 5, 10, 15]:
        rng = np.random.default_rng(stable_seed(SEED, EXPERIMENT, k))
        for index in range(n):
            total = rng.choice([0, 2, 5, 10, 20], k)
            s = rng.integers(0, total + 1)
            f = total - s
            means = (s + 1) / (total + 2)
            best = best_probabilities(s, f)
            fixture = f"{EXPERIMENT}:{k}:{index}"
            for representation in ["counts", "means", "bayes"]:
                obs = observation(s, f, 1, representation)
                for repeat in [0, 1]:
                    best_question = {
                        "type": "choice",
                        "instructions": {
                            "question": "Which arm has the highest fixed but unknown success probability? Express uncertainty about the identity of that arm under the independent Beta(1,1) priors and observed outcomes. This is a belief about the latent best arm, not a recommendation for the next action.",
                            "observation": obs,
                        },
                        "criteria": {
                            a[
                                "id"
                            ]: f"{a['id']} has the largest true success probability."
                            for a in obs["arms"]
                        },
                    }
                    base = {
                        "experiment": EXPERIMENT,
                        "fixture_id": fixture,
                        "k": k,
                        "representation": representation,
                        "repeat": repeat,
                        "successes": s.tolist(),
                        "failures": f.tolist(),
                    }
                    decision_id = f"{fixture}:{representation}:{repeat}:best"
                    jobs.append({"id": decision_id, "question": best_question})
                    metadata.append(
                        {
                            **base,
                            "decision_id": decision_id,
                            "target": "best_arm",
                            "arm": None,
                            "reference": best.tolist(),
                            "simple_baseline": (means / means.sum()).tolist(),
                        }
                    )
                    for arm, record in enumerate(obs["arms"]):
                        event_question = {
                            "type": "noul",
                            "instructions": {
                                "question": f"On the next pull of {record['id']}, will the reward be 1? Use the stated Beta(1,1) prior and observed outcomes. The question concerns a random future outcome, including its uncertainty.",
                                "observation": {"arm": record},
                            },
                            "criteria": {
                                "true": "The next pull of this arm yields reward 1.",
                                "false": "The next pull of this arm yields reward 0.",
                            },
                        }
                        decision_id = (
                            f"{fixture}:{representation}:{repeat}:reward:{arm}"
                        )
                        empirical = float(s[arm] / total[arm]) if total[arm] else 0.5
                        jobs.append({"id": decision_id, "question": event_question})
                        metadata.append(
                            {
                                **base,
                                "decision_id": decision_id,
                                "target": "next_reward",
                                "arm": arm,
                                "reference": [float(means[arm]), float(1 - means[arm])],
                                "simple_baseline": [empirical, 1 - empirical],
                            }
                        )
    return jobs, metadata


def prepare(run_dir, n=100):
    jobs, metadata = build_jobs(n)
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload_hash = hashlib.sha256(json.dumps(jobs, sort_keys=True).encode()).hexdigest()
    record = {
        "experiment": EXPERIMENT,
        "n_fixtures_per_k": n,
        "ks": [2, 3, 5, 10, 15],
        "representations": ["counts", "means", "bayes"],
        "repeats": 2,
        "questions": len(jobs),
        "payload_sha256": payload_hash,
        "log_clip": CLIP,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependency_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ["client.py", "prompts.py", "experiments.py"]
        },
        "shared_state": TASK,
        "model": MODEL,
        "seed": SEED,
        "question_target_distinction": "predictive reward vs latent best arm; no action-policy reward claims",
    }
    manifest = path / "forecast_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != record:
        raise ValueError("Forecast protocol/source changed; use a new run directory")
    manifest.write_text(json.dumps(record, indent=2))
    return jobs, metadata


async def execute(run_dir, cap=18, n=100):
    jobs, metadata = prepare(run_dir, n)
    output = Path(run_dir) / "forecasts.jsonl"
    done = {r["decision_id"] for r in read_jsonl(output)}
    permutation = np.random.default_rng(
        stable_seed(SEED, EXPERIMENT, "order")
    ).permutation(len(jobs))
    indices = [int(i) for i in permutation if jobs[i]["id"] not in done]
    client = ForecastClient(run_dir, cap=cap)
    try:
        for start in range(0, len(indices), 256):
            batch = indices[start : start + 256]
            answers = await client.evaluate([jobs[i] for i in batch])
            records = []
            for index, answer in zip(batch, answers, strict=True):
                record = {
                    **metadata[index],
                    "prediction": answer["probabilities"],
                    "raw_answer": answer["raw_answer"],
                    "request_id": answer["request_id"],
                }
                record.update(scoring(record["reference"], record["prediction"]))
                record["simple_baseline_scores"] = scoring(
                    record["reference"], record["simple_baseline"]
                )
                records.append(record)
            append_json(output, records)
            print(
                json.dumps(
                    {
                        "completed": min(start + 256, len(indices)),
                        "total": len(indices),
                        "project_cost": client.accounting()[
                            "project_cost_and_reserves_usd"
                        ],
                    }
                ),
                flush=True,
            )
    finally:
        await client.close()


def report(run_dir, output_dir):
    from .report import bootstrap_mean, markdown_table, paired_effect

    source, output = Path(run_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(read_jsonl(source / "forecasts.jsonl"))
    # Every fixture contributes equal weight after its arms and repeats are averaged.
    metrics = ["excess_brier", "excess_log_loss", "max_absolute_error"]
    units = data.groupby(
        ["target", "k", "fixture_id", "representation"], as_index=False
    )[metrics].mean()
    summary = []
    effects = []
    for (target, k, representation), group in units.groupby(
        ["target", "k", "representation"]
    ):
        for metric in metrics:
            mean, low, high = bootstrap_mean(
                group[metric],
                seed=stable_seed(EXPERIMENT, target, int(k), representation, metric),
            )
            summary.append(
                {
                    "target": target,
                    "k": int(k),
                    "representation": representation,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n_fixtures": len(group),
                }
            )
    for (target, k), group in units.groupby(["target", "k"]):
        for left, right in [
            ("means", "counts"),
            ("bayes", "counts"),
            ("bayes", "means"),
        ]:
            for metric in metrics:
                effect = paired_effect(
                    group,
                    left,
                    right,
                    metric,
                    ["fixture_id"],
                    policy_col="representation",
                )
                effects.append({"target": target, "k": int(k), **effect.__dict__})
    summary = pd.DataFrame(summary)
    effects = pd.DataFrame(effects)
    baseline_rows = []
    for record in data.to_dict("records"):
        baseline_rows.append(
            {
                "target": record["target"],
                "k": record["k"],
                "fixture_id": record["fixture_id"],
                **record["simple_baseline_scores"],
            }
        )
    baseline_units = (
        pd.DataFrame(baseline_rows)
        .groupby(["target", "k", "fixture_id"], as_index=False)[metrics]
        .mean()
    )
    baseline_summary = []
    for (target, k), group in baseline_units.groupby(["target", "k"]):
        for metric in metrics:
            mean, low, high = bootstrap_mean(
                group[metric],
                seed=stable_seed(EXPERIMENT, "baseline", target, int(k), metric),
            )
            baseline_summary.append(
                {
                    "target": target,
                    "k": int(k),
                    "metric": metric,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n_fixtures": len(group),
                }
            )
    baseline_summary = pd.DataFrame(baseline_summary)
    summary.to_csv(output / "forecast_summary.csv", index=False)
    effects.to_csv(output / "forecast_paired_effects.csv", index=False)
    baseline_summary.to_csv(output / "forecast_simple_baselines.csv", index=False)
    # Reliability against analytic conditional probabilities, not fabricated outcome labels.
    scalar = []
    for _, r in data.iterrows():
        for p, q in zip(r.prediction, r.reference, strict=True):
            scalar.append(
                {
                    "target": r.target,
                    "representation": r.representation,
                    "prediction": p,
                    "reference": q,
                }
            )
    scalars = pd.DataFrame(scalar)
    scalars["bin"] = np.minimum(9, (scalars.prediction * 10).astype(int))
    reliability = scalars.groupby(
        ["target", "representation", "bin"], as_index=False
    ).agg(
        prediction=("prediction", "mean"),
        reference=("reference", "mean"),
        n=("prediction", "size"),
    )
    reliability.to_csv(output / "conditional_reliability.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, target in zip(axes, ["next_reward", "best_arm"], strict=True):
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        for representation in ["counts", "means", "bayes"]:
            z = reliability[
                (reliability.target == target)
                & (reliability.representation == representation)
            ]
            ax.plot(z.prediction, z.reference, "o-", label=representation)
        ax.set(
            xlabel="Jev probability",
            ylabel="Analytic conditional probability",
            title=target.replace("_", " "),
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend()
    fig.savefig(output / "forecast_reliability.png", dpi=180)
    plt.close(fig)
    selected = summary[summary.metric == "excess_brier"]
    text = [
        "# E7: probability forecasts, separate from action selection",
        f"Recorded {len(data):,} questions; independent fixtures per K and target are the resampling units. Arms and API repeats are averaged within fixtures.",
        "The targets are analytic posterior predictive reward probabilities and quadrature-computed probabilities of being the latent best arm. Excess Brier risk is the sum of squared errors between distributions. Excess log loss is KL(reference || prediction), clipping predictions to 1e-6 and renormalizing. For a binary event the reported two-class Brier excess equals twice the scalar squared error.",
        "## Excess Brier risk",
        markdown_table(selected),
        "## Simple forecast controls",
        markdown_table(baseline_summary[baseline_summary.metric == "excess_brier"]),
        "## Interpretation limits",
        "These are belief questions, not action recommendations. Errors cannot be directly substituted for online regret. The simple reward baseline is the unsmoothed empirical rate (0.5 if unobserved); the simple best-arm baseline normalizes posterior means, which is not a Bayesian best-arm distribution. Analytic references are the proper-scoring optimum by construction. Paired intervals are exploratory, pointwise, and unadjusted for multiplicity.",
        "![Conditional probability reliability](forecast_reliability.png)",
        "Reliability curves pool individual probability components descriptively; scalar components and arms are dependent. Inferential tables weight independent fixtures equally. No outcome labels were simulated or silently treated as observed.",
    ]
    (output / "report.md").write_text("\n\n".join(text) + "\n")
    return output / "report.md"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run", "report"])
    parser.add_argument("--run-dir", default="artifacts/forecast_v1")
    parser.add_argument("--output", default="reports/forecast_v1")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--cap", type=float, default=18)
    args = parser.parse_args()
    if args.command == "prepare":
        jobs, _ = prepare(args.run_dir, args.n)
        print(f"Prepared {len(jobs)} questions; no API calls")
    elif args.command == "run":
        asyncio.run(execute(args.run_dir, args.cap, args.n))
    else:
        print(report(args.run_dir, args.output))


if __name__ == "__main__":
    main()
