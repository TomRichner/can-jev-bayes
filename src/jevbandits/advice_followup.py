"""Pre-registered E6 advice-binding follow-up; prepare and run are separate steps."""

import argparse
import asyncio
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from .baselines import exact_q
from .client import JevClient
from .experiments import SEED, append_json, fixtures, read_jsonl
from .prompts import MODEL, OBJECTIVES, TASK, observation, question, stable_seed

EXPERIMENT = "e6_advice_binding"
FORMATS = (
    "nested",
    "path_explicit",
    "inline",
    "per_option",
    "advice_only",
    "dp_values",
)
LABELS = {"original": ("arm_00", "arm_01"), "letters": ("A", "B")}
HORIZONS = (1, 2, 10)
TOLERANCE = 1e-10
ROOT = Path(__file__).resolve().parents[2]
FROZEN_FILES = (
    "src/jevbandits/advice_followup.py",
    "tests/test_advice_followup.py",
    "docs/e6_advice_protocol.md",
    "src/jevbandits/prompts.py",
    "src/jevbandits/experiments.py",
    "src/jevbandits/baselines.py",
    "src/jevbandits/client.py",
    "uv.lock",
)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def fixture_panel():
    """Generate states without consulting any Jev results or response artifacts."""
    panel = [
        {"cohort": "random", "index": i, "successes": s, "failures": f}
        for i, (s, f) in enumerate(fixtures("e6_advice_random", 100))
    ]
    rng = np.random.default_rng(stable_seed(SEED, EXPERIMENT, "targeted"))
    seen = set()
    attempts = 0
    while len(seen) < 100:
        attempts += 1
        if attempts > 100000:
            raise RuntimeError("Targeted rejection sampler exhausted its fixed guard")
        n = int(rng.integers(10, 81))
        # Uniform integer successes among those with posterior mean in [.51,.68].
        lo, hi = math.ceil(0.51 * (n + 2) - 1), math.floor(0.68 * (n + 2) - 1)
        s = int(rng.integers(lo, hi + 1))
        if (s, n - s) in seen:
            continue
        q = exact_q((s, 0), (n - s, 0), 10)
        if q[1] - q[0] <= 1e-4:
            continue
        panel.append(
            {
                "cohort": "targeted",
                "index": len(seen),
                "successes": [s, 0],
                "failures": [n - s, 0],
                "accepted_draw": attempts,
            }
        )
        seen.add((s, n - s))
    return panel


def make_question(s, f, horizon, format_name, label_scheme, reversal, q_values):
    if (
        format_name not in FORMATS
        or label_scheme not in LABELS
        or reversal not in (0, 1)
    ):
        raise ValueError("Unknown E6 condition")
    order = [0, 1] if reversal == 0 else [1, 0]
    labels = LABELS[label_scheme]
    q = np.asarray(q_values)
    gold = int(np.flatnonzero(np.isclose(q, q.max(), atol=TOLERANCE, rtol=0))[0])
    recommended = labels[order.index(gold)]
    representation = "dp_values" if format_name == "dp_values" else "recommendation"
    obs = observation(s, f, horizon, representation, q_values=q, label_order=order)
    for item, label in zip(obs["arms"], labels):
        item["id"] = label
    if format_name != "dp_values":
        obs["recommended_arm"] = recommended
    framing = "explore" if format_name == "dp_values" else "recommendation"
    if format_name == "advice_only":
        obs = {
            "remaining_pulls_including_this_one": horizon,
            "arms": [{"id": label} for label in labels],
            "recommended_arm": recommended,
        }
    result = question(obs, framing)
    if format_name == "path_explicit":
        result["instructions"]["question"] = (
            "Read observation.recommended_arm. Choose the criterion whose ID exactly "
            "matches that value. " + OBJECTIVES["recommendation"]
        )
    elif format_name == "inline":
        result["instructions"]["question"] = (
            f"Choose EXACTLY {recommended}. " + OBJECTIVES["recommendation"]
        )
    elif format_name == "per_option":
        result["criteria"] = {
            label: f"Pull {label} now. "
            + (
                "This is the recommended arm; choose this option."
                if label == recommended
                else "This is not the recommended arm; do not choose this option."
            )
            for label in labels
        }
    return result, gold, recommended, order


def build_design(panel=None):
    panel = fixture_panel() if panel is None else panel
    jobs, contexts = [], []
    for fixture in panel:
        s, f = fixture["successes"], fixture["failures"]
        for horizon in HORIZONS:
            q = exact_q(tuple(s), tuple(f), horizon)
            for format_name in FORMATS:
                for label_scheme in LABELS:
                    for reversal in (0, 1):
                        payload, gold, recommended, order = make_question(
                            s, f, horizon, format_name, label_scheme, reversal, q
                        )
                        for repeat in (0, 1):
                            fixture_id = (
                                f"{EXPERIMENT}:{fixture['cohort']}:{fixture['index']}"
                            )
                            identity = f"{fixture_id}:{horizon}:{format_name}:{label_scheme}:{reversal}:{repeat}"
                            jobs.append({"id": identity, "question": payload})
                            contexts.append(
                                {
                                    "experiment": EXPERIMENT,
                                    "fixture_id": fixture_id,
                                    "cohort": fixture["cohort"],
                                    "decision_id": identity,
                                    "horizon": horizon,
                                    "format": format_name,
                                    "representation": format_name,
                                    "framing": "advice_binding",
                                    "policy": f"{format_name}_{label_scheme}",
                                    "label_scheme": label_scheme,
                                    "label_order": reversal,
                                    "repeat": repeat,
                                    "successes": s,
                                    "failures": f,
                                    "q_values": q.tolist(),
                                    "order": order,
                                    "recommended_canonical_action": gold,
                                    "recommended_arm": recommended,
                                    "advice_present": format_name != "dp_values",
                                }
                            )
    permutation = np.random.default_rng(
        stable_seed(SEED, EXPERIMENT, "execution")
    ).permutation(len(jobs))
    return [jobs[i] for i in permutation], [contexts[i] for i in permutation]


def manifest_content(panel, jobs):
    return {
        "experiment": EXPERIMENT,
        "version": 1,
        "master_seed": SEED,
        "targeted_seed": stable_seed(SEED, EXPERIMENT, "targeted"),
        "random_seed": stable_seed(SEED, "e6_advice_random", "fixtures"),
        "execution_seed": stable_seed(SEED, EXPERIMENT, "execution"),
        "model": MODEL,
        "shared_state": TASK,
        "objectives": OBJECTIVES,
        "formats": list(FORMATS),
        "label_schemes": LABELS,
        "horizons": list(HORIZONS),
        "repeats": 2,
        "assignments": 2,
        "decisions": len(jobs),
        "fixture_sha256": digest(panel),
        "ordered_jobs_sha256": digest(jobs),
        "action_rule": "backend_choice",
        "tie_tolerance": TOLERANCE,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in FROZEN_FILES
        },
    }


def prepare(run_dir):
    panel = fixture_panel()
    jobs, _ = build_design(panel)
    content = json.loads(json.dumps(manifest_content(panel, jobs)))
    path = Path(run_dir)
    if (path / "manifest.json").exists():
        verify_manifest(path, panel, jobs)
        return content
    if path.exists() and any(path.iterdir()):
        raise ValueError("Preparation requires an empty new run directory")
    path.mkdir(parents=True, exist_ok=True)
    (path / "fixtures.json").write_text(json.dumps(panel, indent=2) + "\n")
    content["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    (path / "manifest.json").write_text(json.dumps(content, indent=2) + "\n")
    return content


def verify_manifest(run_dir, panel, jobs):
    path = Path(run_dir)
    if not (path / "manifest.json").exists():
        raise ValueError(
            "Run requires a previously prepared manifest; use --prepare first"
        )
    existing = json.loads((path / "manifest.json").read_text())
    expected = json.loads(json.dumps(manifest_content(panel, jobs)))
    for key, value in expected.items():
        if existing.get(key) != value:
            raise ValueError(
                f"Frozen E6 {key} changed; a new namespace/amendment is required"
            )
    if json.loads((path / "fixtures.json").read_text()) != panel:
        raise ValueError("Frozen E6 fixture file changed")


def score(context, answer):
    row = context.copy()
    order = row.pop("order")
    action = order[answer["action"]]
    q = np.asarray(row["q_values"])
    s, f = np.asarray(row["successes"]), np.asarray(row["failures"])
    means = (s + 1) / (s + f + 2)
    p = np.zeros(2)
    p[order] = answer["probabilities"]
    row.update(
        action=answer["action"],
        canonical_action=action,
        chosen_arm=LABELS[row["label_scheme"]][answer["action"]],
        probabilities=p.tolist(),
        exact_loss=float(q.max() - q[action]),
        optimal_agreement=bool(q.max() - q[action] <= TOLERANCE),
        advice_adherence=bool(action == row["recommended_canonical_action"]),
        distribution_advice_adherence=float(p[row["recommended_canonical_action"]]),
        distribution_exact_loss=float(q.max() - p @ q),
        posterior_mean_greedy=bool(means.max() - means[action] <= TOLERANCE),
        prediction_entropy=float(-sum(x * np.log(x) for x in p if x > 0)),
        request_id=answer["request_id"],
        raw_answer=answer["raw_answer"],
        choice_probability_gap=answer.get("choice_probability_gap", 0),
        choice_mismatch=bool(answer.get("choice_probability_gap", 0) > 1e-8),
        probability_mass=answer["probability_mass"],
    )
    return row


async def execute(client, jobs, contexts, max_decisions=None):
    """Resume the fixed permutation. A limit pauses collection, never changes design."""
    output = client.run_dir / "diagnostics.jsonl"
    done = {r["decision_id"] for r in read_jsonl(output)}
    pending = [(j, c) for j, c in zip(jobs, contexts) if j["id"] not in done]
    if max_decisions is not None:
        pending = pending[:max_decisions]
    for start in range(0, len(pending), 256):
        chunk = pending[start : start + 256]
        answers = await client.evaluate([j for j, _ in chunk])
        if len(answers) != len(chunk):
            raise ValueError("Missing E6 answers; no partial chunk will be scored")
        append_json(output, [score(c, a) for (_, c), a in zip(chunk, answers)])
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


async def run(args):
    if args.prepare:
        print(json.dumps(prepare(args.run_dir), indent=2))
        return
    panel = fixture_panel()
    jobs, contexts = build_design(panel)
    verify_manifest(args.run_dir, panel, jobs)
    client = JevClient(
        args.run_dir,
        cap=args.cap,
        rate=args.rate,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
    )
    try:
        await execute(client, jobs, contexts, args.max_decisions)
    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prepare", action="store_true", help="Freeze only; no API client"
    )
    mode.add_argument("--run", action="store_true", help="Execute the prepared design")
    parser.add_argument("--run-dir", default="artifacts/followup_advice_v1")
    parser.add_argument("--cap", type=float, default=18.0)
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-decisions", type=int, help="Pause after this many pending decisions"
    )
    args = parser.parse_args()
    if args.max_decisions is not None and args.max_decisions < 1:
        parser.error("--max-decisions must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
