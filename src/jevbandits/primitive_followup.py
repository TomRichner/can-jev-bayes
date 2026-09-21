"""E10 forecast-target/primitive bridge and known-probability controls."""

import argparse
import asyncio
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .experiments import SEED, append_json, read_jsonl
from .forecast_followup import CLIP, ForecastClient, best_probabilities, scoring
from .prompts import MODEL, TASK, observation, stable_seed

EXPERIMENT = "e10_primitive_bridge"
KS = (2, 5, 15)
N = 30
REPEATS = 2
REPRESENTATIONS = ("counts", "bayes", "oracle_probs_full", "oracle_probs_only")
PRIMARY = (("next_reward", "choice", "noul"), ("best_arm", "noul", "choice"))
ROOT = Path(__file__).resolve().parents[2]
FROZEN_FILES = (
    "src/jevbandits/primitive_followup.py",
    "tests/test_primitive_followup.py",
    "docs/e10_primitive_protocol.md",
    "src/jevbandits/forecast_followup.py",
    "src/jevbandits/client.py",
    "src/jevbandits/experiments.py",
    "src/jevbandits/baselines.py",
    "src/jevbandits/prompts.py",
    "uv.lock",
)
EVENT_FIELDS = {
    "next_reward": "exact_probability_next_reward_is_one",
    "best_arm": "exact_probability_true_mean_is_largest",
}
FORECAST_CONTEXT = (
    "This is a probability forecast, not a recommendation to pull an arm. "
    "Use the independent Beta(1,1) priors and the stated observations when present. "
    "When an exact event probability is supplied, it is the known conditional "
    "probability of the named event and should be reported directly."
)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def fixture_panel():
    panel = []
    for k in KS:
        rng = np.random.default_rng(stable_seed(SEED, EXPERIMENT, "fixtures", k))
        for index in range(N):
            total = rng.choice([0, 2, 5, 10, 20], k)
            s = rng.integers(0, total + 1)
            f = total - s
            panel.append(
                {
                    "fixture_id": f"{EXPERIMENT}:{k}:{index}",
                    "k": k,
                    "index": index,
                    "successes": s.tolist(),
                    "failures": f.tolist(),
                    "next_reward": ((s + 1) / (total + 2)).tolist(),
                    "best_arm": best_probabilities(s, f).tolist(),
                }
            )
    return panel


def target_observation(fixture, representation, target):
    if representation not in REPRESENTATIONS or target not in EVENT_FIELDS:
        raise ValueError("Unknown E10 representation or target")
    if representation == "oracle_probs_only":
        obs = {"arms": [{"id": f"arm_{i:02d}"} for i in range(fixture["k"])]}
    else:
        obs = observation(
            fixture["successes"],
            fixture["failures"],
            1,
            "counts" if representation == "counts" else "bayes",
        )
        # Forecasts have no decision horizon; keep the evidence fields themselves.
        obs.pop("remaining_pulls_including_this_one")
    if representation.startswith("oracle_probs"):
        for arm, value in zip(obs["arms"], fixture[target], strict=True):
            arm[EVENT_FIELDS[target]] = value
    obs["forecast_context"] = FORECAST_CONTEXT
    obs["target_event"] = (
        "The next binary reward from the named arm is 1."
        if target == "next_reward"
        else "The named arm's fixed unknown success probability is strictly largest among these arms. "
        "Continuous independent Beta posteriors give ties probability zero."
    )
    return obs


def make_question(fixture, representation, target, primitive, arm=None):
    if primitive not in ("choice", "noul"):
        raise ValueError("Unknown primitive")
    obs = target_observation(fixture, representation, target)
    if target == "best_arm" and primitive == "choice":
        if arm is not None:
            raise ValueError("Best-arm Choice is a joint categorical question")
        prompt = (
            "Which arm's fixed unknown success probability is largest? Report probabilities for the identities of the best arm. "
            + FORECAST_CONTEXT
        )
        criteria = {
            a["id"]: f"{a['id']}'s fixed unknown success probability is largest."
            for a in obs["arms"]
        }
    else:
        if arm is None or not 0 <= arm < fixture["k"]:
            raise ValueError("Binary forecasts require a valid arm")
        label = f"arm_{arm:02d}"
        event = (
            f"the next pull of {label} yields reward 1"
            if target == "next_reward"
            else f"{label}'s fixed unknown success probability is largest among these arms"
        )
        prompt = (
            f"Is it true that {event}? Report the probability that this event is true. "
            + FORECAST_CONTEXT
        )
        criteria = {
            "true": f"Yes: {event}.",
            "false": f"No: it is not true that {event}.",
        }
        obs["queried_arm"] = label
    return {
        "type": primitive,
        "instructions": {"question": prompt, "observation": obs},
        "criteria": criteria,
    }


def build_design(panel):
    jobs, contexts = [], []
    for fixture in panel:
        for representation in REPRESENTATIONS:
            for repeat in range(REPEATS):
                for target in EVENT_FIELDS:
                    for primitive in ("noul", "choice"):
                        arms = (
                            [None]
                            if target == "best_arm" and primitive == "choice"
                            else range(fixture["k"])
                        )
                        for arm in arms:
                            identity = f"{fixture['fixture_id']}:{representation}:{repeat}:{target}:{primitive}:{arm}"
                            jobs.append(
                                {
                                    "id": identity,
                                    "question": make_question(
                                        fixture, representation, target, primitive, arm
                                    ),
                                }
                            )
                            reference = (
                                fixture[target]
                                if arm is None
                                else [fixture[target][arm]]
                            )
                            contexts.append(
                                {
                                    "experiment": EXPERIMENT,
                                    "fixture_id": fixture["fixture_id"],
                                    "decision_id": identity,
                                    "k": fixture["k"],
                                    "representation": representation,
                                    "repeat": repeat,
                                    "target": target,
                                    "primitive": primitive,
                                    "arm": arm,
                                    "reference_event_probabilities": reference,
                                }
                            )
    order = np.random.default_rng(
        stable_seed(SEED, EXPERIMENT, "execution")
    ).permutation(len(jobs))
    return [jobs[i] for i in order], [contexts[i] for i in order]


def manifest_content(panel, jobs, contexts):
    return {
        "experiment": EXPERIMENT,
        "version": 1,
        "master_seed": SEED,
        "ks": list(KS),
        "fixtures_per_k": N,
        "repeats": REPEATS,
        "representations": list(REPRESENTATIONS),
        "questions": len(jobs),
        "fixtures": len(panel),
        "model": MODEL,
        "shared_state": TASK,
        "fixture_sha256": digest(panel),
        "ordered_jobs_sha256": digest(jobs),
        "contexts_sha256": digest(contexts),
        "primary": {
            "representation": "bayes",
            "contrasts_target_left_right": PRIMARY,
            "metric": "mean_binary_excess_brier",
            "pooling": "equal K weights",
            "bootstrap": "10000 paired whole fixtures within K",
            "pointwise_quantiles": [0.025, 0.975],
            "bonferroni_quantiles": [0.0125, 0.9875],
            "normalization": "none for raw Noul best-arm marginal probabilities",
            "incomplete": "complete fixture conditions only; disclose missing pairs; incomplete pool descriptive",
        },
        "secondary_log_clip": CLIP,
        "budget": {
            "default_shared_cap_usd": 18,
            "rough_incremental_estimate_usd": 0.6,
            "gate": "parent budget check after E9; no automatic credit purchase",
        },
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in FROZEN_FILES
        },
    }


def verify_manifest(run_dir, panel, jobs, contexts):
    path = Path(run_dir)
    if not (path / "manifest.json").exists():
        raise ValueError("E10 requires prepare before run")
    expected = json.loads(json.dumps(manifest_content(panel, jobs, contexts)))
    existing = json.loads((path / "manifest.json").read_text())
    for key, value in expected.items():
        if existing.get(key) != value:
            raise ValueError(f"Frozen E10 {key} changed; use a new namespace/amendment")
    if json.loads((path / "fixtures.json").read_text()) != panel:
        raise ValueError("Frozen E10 fixture file changed")
    return existing


def prepare(run_dir):
    panel = fixture_panel()
    jobs, contexts = build_design(panel)
    path = Path(run_dir)
    if (path / "manifest.json").exists():
        return verify_manifest(path, panel, jobs, contexts)
    if path.exists() and any(path.iterdir()):
        raise ValueError("Preparation requires an empty new run directory")
    content = manifest_content(panel, jobs, contexts)
    content["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    path.mkdir(parents=True, exist_ok=True)
    (path / "fixtures.json").write_text(json.dumps(panel, indent=2) + "\n")
    (path / "manifest.json").write_text(json.dumps(content, indent=2) + "\n")
    return content


def event_scores(reference, prediction):
    """Common binary proper-risk scale, including unnormalized marginal vectors."""
    q, p = np.asarray(reference, dtype=float), np.asarray(prediction, dtype=float)
    if q.ndim != 1 or not q.size or q.shape != p.shape:
        raise ValueError("Event vectors must have matching nonempty shapes")
    if (
        not np.isfinite(q).all()
        or not np.isfinite(p).all()
        or np.any(q < 0)
        or np.any(q > 1)
        or np.any(p < 0)
        or np.any(p > 1)
    ):
        raise ValueError("Event probabilities must be finite in [0,1]")
    binary = [scoring([qi, 1 - qi], [pi, 1 - pi]) for qi, pi in zip(q, p, strict=True)]
    return {
        "mean_binary_excess_brier": float(2 * np.mean((p - q) ** 2)),
        "mean_binary_excess_log_loss": float(
            np.mean([r["excess_log_loss"] for r in binary])
        ),
        "mean_absolute_error": float(np.mean(np.abs(p - q))),
    }


def score(context, answer):
    p = (
        answer["probabilities"]
        if context["arm"] is None
        else [answer["probabilities"][0]]
    )
    return {
        **context,
        "prediction_event_probabilities": p,
        **event_scores(context["reference_event_probabilities"], p),
        "raw_answer": answer["raw_answer"],
        "probability_mass": answer["probability_mass"],
        "request_id": answer["request_id"],
    }


async def execute(client, jobs, contexts, max_questions=None):
    output = client.run_dir / "forecasts.jsonl"
    done = {r["decision_id"] for r in read_jsonl(output)}
    pending = [
        (job, context)
        for job, context in zip(jobs, contexts, strict=True)
        if job["id"] not in done
    ]
    if max_questions is not None:
        pending = pending[:max_questions]
    for start in range(0, len(pending), 256):
        chunk = pending[start : start + 256]
        answers = await client.evaluate([job for job, _ in chunk])
        if len(answers) != len(chunk):
            raise ValueError("Missing E10 answers; no partial chunk will be scored")
        append_json(
            output,
            [
                score(context, answer)
                for (_, context), answer in zip(chunk, answers, strict=True)
            ],
        )
        print(
            json.dumps(
                {
                    "experiment": EXPERIMENT,
                    "completed": len(done) + start + len(chunk),
                    "planned": len(jobs),
                    "accounting": client.accounting(),
                }
            ),
            flush=True,
        )


def fixture_units(records):
    """Only complete arm/repeat packages enter the independent-fixture analysis."""
    metrics = [
        "mean_binary_excess_brier",
        "mean_binary_excess_log_loss",
        "mean_absolute_error",
    ]
    columns = ["fixture_id", "k", "representation", "target", "primitive"]
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=columns + metrics), pd.DataFrame()
    if frame.decision_id.duplicated().any():
        raise ValueError("Duplicate E10 decisions")
    units, incomplete = [], []
    for key, group in frame.groupby(columns, dropna=False):
        _fixture, k, _representation, target, primitive = key
        arms = (
            [None] if target == "best_arm" and primitive == "choice" else range(int(k))
        )
        expected = {(repeat, arm) for repeat in range(REPEATS) for arm in arms}
        observed = {
            (int(row["repeat"]), None if pd.isna(row["arm"]) else int(row["arm"]))
            for row in group.to_dict("records")
        }
        identity = dict(zip(columns, key, strict=True))
        if observed != expected or len(group) != len(expected):
            incomplete.append(
                {**identity, "observed": len(group), "expected": len(expected)}
            )
            continue
        units.append(
            {**identity, **{metric: float(group[metric].mean()) for metric in metrics}}
        )
    return pd.DataFrame(units, columns=columns + metrics), pd.DataFrame(incomplete)


def bootstrap_contrast(
    units, target, left, right, *, samples=10000, ks=KS, expected_n=N
):
    """Resample matched fixture differences independently within each K stratum."""
    metric = "mean_binary_excess_brier"
    selected = units[(units.representation == "bayes") & (units.target == target)]
    if selected.duplicated(["fixture_id", "primitive"]).any():
        raise ValueError("Duplicate primitive/fixture units")
    rng = np.random.default_rng(
        stable_seed(SEED, EXPERIMENT, "primary", target, left, right)
    )
    cells, draws, means = [], [], []
    for k in ks:
        group = selected[selected.k == k]
        a = group[group.primitive == left].set_index("fixture_id")[metric]
        b = group[group.primitive == right].set_index("fixture_id")[metric]
        paired = pd.concat([a.rename("left"), b.rename("right")], axis=1).dropna()
        differences = (paired.left - paired.right).to_numpy(dtype=float)
        if not np.isfinite(differences).all():
            raise ValueError("Nonfinite paired risk")
        cells.append(
            {
                "k": k,
                "n_left": len(a),
                "n_right": len(b),
                "n_pairs": len(differences),
                "expected": expected_n,
            }
        )
        if len(differences):
            means.append(float(differences.mean()))
            draws.append(
                differences[
                    rng.integers(0, len(differences), size=(samples, len(differences)))
                ].mean(axis=1)
            )
    complete = all(c["n_pairs"] == expected_n for c in cells)
    estimable = len(draws) == len(ks) and all(c["n_pairs"] >= 2 for c in cells)
    pooled = np.mean(draws, axis=0) if estimable else None
    quantiles = (
        np.quantile(pooled, [0.025, 0.975, 0.0125, 0.9875]).tolist()
        if estimable
        else [None] * 4
    )
    return {
        "target": target,
        "left": left,
        "right": right,
        "representation": "bayes",
        "metric": metric,
        "mean_difference": float(np.mean(means)) if len(means) == len(ks) else None,
        "ci95_low": quantiles[0],
        "ci95_high": quantiles[1],
        "ci97_5_low": quantiles[2],
        "ci97_5_high": quantiles[3],
        "complete": complete,
        "status": "prespecified full design"
        if complete
        else "incomplete descriptive only",
        "cells": cells,
    }


def coherence_records(records):
    """Secondary best-arm Noul coherence; never changes the primary forecasts."""
    frame = pd.DataFrame(
        [r for r in records if r["target"] == "best_arm" and r["primitive"] == "noul"]
    )
    if frame.empty:
        return []
    result = []
    for (fixture, k, representation, repeat), group in frame.groupby(
        ["fixture_id", "k", "representation", "repeat"]
    ):
        if len(group) != k or set(group.arm) != set(range(k)):
            continue
        group = group.sort_values("arm")
        p = np.array([r[0] for r in group.prediction_event_probabilities])
        q = np.array([r[0] for r in group.reference_event_probabilities])
        mass = float(p.sum())
        result.append(
            {
                "fixture_id": fixture,
                "k": int(k),
                "representation": representation,
                "repeat": int(repeat),
                "sum_probabilities": mass,
                "absolute_sum_error": abs(mass - 1),
                "normalization_defined": mass > 0,
                "normalized_mean_binary_excess_brier": event_scores(q, p / mass)[
                    "mean_binary_excess_brier"
                ]
                if mass > 0
                else None,
            }
        )
    return result


def secondary_effects(units, *, samples=10000):
    """Per-K, per-representation primitive contrasts; exploratory 95% only."""
    result = []
    metric = "mean_binary_excess_brier"
    for representation in REPRESENTATIONS:
        for target, left, right in PRIMARY:
            for k in KS:
                group = units[
                    (units.representation == representation)
                    & (units.target == target)
                    & (units.k == k)
                ]
                a = group[group.primitive == left].set_index("fixture_id")[metric]
                b = group[group.primitive == right].set_index("fixture_id")[metric]
                paired = pd.concat(
                    [a.rename("left"), b.rename("right")], axis=1
                ).dropna()
                delta = (paired.left - paired.right).to_numpy(dtype=float)
                low, high = None, None
                if len(delta) >= 2:
                    rng = np.random.default_rng(
                        stable_seed(
                            SEED, EXPERIMENT, "secondary", representation, target, k
                        )
                    )
                    draws = delta[
                        rng.integers(0, len(delta), size=(samples, len(delta)))
                    ].mean(axis=1)
                    low, high = np.quantile(draws, [0.025, 0.975]).tolist()
                result.append(
                    {
                        "representation": representation,
                        "target": target,
                        "k": k,
                        "left": left,
                        "right": right,
                        "n_left": len(a),
                        "n_right": len(b),
                        "n_pairs": len(delta),
                        "expected": N,
                        "mean_difference": float(delta.mean()) if len(delta) else None,
                        "ci95_low": low,
                        "ci95_high": high,
                        "status": "exploratory pointwise; not primary family",
                    }
                )
    return result


def report(run_dir, output_dir):
    from .report import markdown_table

    records = read_jsonl(Path(run_dir) / "forecasts.jsonl")
    if not records:
        raise ValueError("No E10 forecasts to report")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    units, incomplete = fixture_units(records)
    units.to_csv(output / "fixture_risks.csv", index=False)
    incomplete.to_csv(output / "incomplete_conditions.csv", index=False)
    primary = [bootstrap_contrast(units, *contrast) for contrast in PRIMARY]
    (output / "primary_effects.json").write_text(
        json.dumps(primary, indent=2, allow_nan=False) + "\n"
    )
    table = pd.DataFrame(
        [{k: v for k, v in effect.items() if k != "cells"} for effect in primary]
    )
    table.to_csv(output / "primary_effects.csv", index=False)
    completeness = pd.DataFrame(
        [
            {"target": effect["target"], **cell}
            for effect in primary
            for cell in effect["cells"]
        ]
    )
    completeness.to_csv(output / "primary_completeness.csv", index=False)
    coherence = pd.DataFrame(coherence_records(records))
    coherence.to_csv(output / "best_noul_coherence.csv", index=False)
    secondary = pd.DataFrame(secondary_effects(units))
    secondary.to_csv(output / "secondary_primitive_effects.csv", index=False)
    summary = units.groupby(
        ["target", "primitive", "representation", "k"], as_index=False
    ).agg(
        mean_binary_excess_brier=("mean_binary_excess_brier", "mean"),
        mean_absolute_error=("mean_absolute_error", "mean"),
        n_fixtures=("fixture_id", "size"),
    )
    summary.to_csv(output / "risk_summary.csv", index=False)
    text = [
        "# E10: forecast target and elicitation format",
        f"Recorded {len(records):,} questions; {len(units):,} complete fixture/condition packages; {len(incomplete):,} observed incomplete packages. Planned: 16,560 questions and 30 fixtures per K = 2, 5, 15. Entirely absent conditions appear in primary completeness counts, not the observed-incomplete table.",
        "The common endpoint is mean binary excess Brier risk: 2 times mean squared event-probability error across arms and repeats. The best-arm Choice vector receives 2/K times its squared-error sum. Separate Noul best-arm probabilities remain unnormalized for every primary calculation. This is excess conditional proper-scoring risk, not realized reward or empirical outcome accuracy.",
        "## Prespecified primary contrasts under full Bayesian summaries",
        markdown_table(table),
        markdown_table(completeness),
        "Differences are left minus right; lower risk is better. Bootstrap whole paired fixtures within K, then weight the three K strata equally. Pointwise 95% and Bonferroni 97.5% intervals are shown for the two-contrast family; the latter target simultaneous 95% coverage subject to bootstrap approximation. Incomplete-design estimates are descriptive. A zero-crossing interval is not equivalence.",
        "## Descriptive risks",
        markdown_table(summary),
        "Per-K primitive contrasts for every representation, with exploratory pointwise 95% intervals and missing-pair counts, are in secondary_primitive_effects.csv. They are outside the two overall primary contrasts.",
        "## Interpretation limits",
        "Reward Choice and Noul use identical binary event wording and rubric, with the type field changed. Best-arm Choice elicits one joint categorical distribution, whereas Noul elicits separate marginal events; this contrast includes question scope and joint-coherence demands as well as primitive. Known-probability controls test following explicit event probabilities, with and without counts and moments. These bundled changes cannot identify an internal architectural mechanism or prove that all joint inference fails. No action recommendation is tested.",
        "Raw Noul probability sums and optional normalized best-arm risks are secondary outputs in best_noul_coherence.csv. Zero-sum vectors have undefined normalized risk and are not replaced with a uniform distribution. Secondary log risk clips each binary distribution at 1e-6 and renormalizes, as in E7; this clipping does not affect Brier risk. Repeated calls and arms are dependent and are averaged within independent fixtures.",
    ]
    (output / "report.md").write_text("\n\n".join(text) + "\n")
    return output / "report.md"


async def run(args):
    panel = fixture_panel()
    jobs, contexts = build_design(panel)
    verify_manifest(args.run_dir, panel, jobs, contexts)
    client = ForecastClient(args.run_dir, cap=args.cap)
    try:
        await execute(client, jobs, contexts, args.max_questions)
    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "report"])
    parser.add_argument("--run-dir", default="artifacts/primitive_v1")
    parser.add_argument("--output", default="reports/primitive_v1")
    parser.add_argument("--cap", type=float, default=18)
    parser.add_argument(
        "--max-questions",
        type=int,
        help="Pause after this many pending questions; does not change N",
    )
    args = parser.parse_args()
    if args.max_questions is not None and args.max_questions < 1:
        parser.error("--max-questions must be positive")
    if args.command == "prepare":
        print(json.dumps(prepare(args.run_dir), indent=2))
    elif args.command == "run":
        asyncio.run(run(args))
    else:
        print(report(args.run_dir, args.output))


if __name__ == "__main__":
    main()
