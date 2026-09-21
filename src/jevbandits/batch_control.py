"""Frozen validation audit of standalone versus mixed E6 question batching."""

import argparse
import asyncio
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import advice_followup as e6
from .client import JevClient
from .experiments import SEED, append_json, read_jsonl
from .prompts import MODEL, TASK, stable_seed

EXPERIMENT = "batch_control"
ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "artifacts/followup_advice_v1"
BLOCK_SIZE = 16
MODES = ("alone", "mixed")
BOOTSTRAPS = 10000
FROZEN_FILES = (
    *e6.FROZEN_FILES,
    "src/jevbandits/batch_control.py",
    "tests/test_batch_control.py",
    "docs/batch_control_protocol.md",
    "src/jevbandits/__init__.py",
    "pyproject.toml",
    "artifacts/followup_advice_v1/manifest.json",
    "artifacts/followup_advice_v1/fixtures.json",
)


def source_panel():
    """Verify prepared E6 inputs; never read its responses."""
    panel = e6.fixture_panel()
    jobs, _ = e6.build_design(panel)
    e6.verify_manifest(SOURCE_DIR, panel, jobs)
    return [f for f in panel if f["index"] < 10]


def build_design(panel):
    source_jobs, source_contexts = e6.build_design(panel)
    selected = sorted(
        ((j, c) for j, c in zip(source_jobs, source_contexts) if c["horizon"] == 10),
        key=lambda pair: pair[0]["id"],
    )
    if len(selected) % BLOCK_SIZE:
        raise ValueError("Source design must contain complete 16-question blocks")
    rng = np.random.default_rng(stable_seed(SEED, EXPERIMENT, "execution"))
    selected = [selected[i] for i in rng.permutation(len(selected))]
    blocks = len(selected) // BLOCK_SIZE
    first_modes = np.array([i % 2 for i in range(blocks)])
    rng.shuffle(first_modes)
    jobs, contexts = [], []
    for block in range(blocks):
        entries = selected[block * BLOCK_SIZE : (block + 1) * BLOCK_SIZE]
        for mode in (MODES[first_modes[block]], MODES[1 - first_modes[block]]):
            for position, (source, context) in enumerate(entries):
                identity = f"{EXPERIMENT}:{mode}:{source['id']}"
                jobs.append({"id": identity, "question": source["question"]})
                contexts.append(
                    dict(
                        context,
                        experiment=EXPERIMENT,
                        source_experiment=context["experiment"],
                        source_decision_id=source["id"],
                        decision_id=identity,
                        batch_mode=mode,
                        block=block,
                        block_position=position,
                        first_mode=MODES[first_modes[block]],
                    )
                )
    return jobs, contexts


def manifest_content(panel, jobs, contexts):
    return {
        "experiment": EXPERIMENT,
        "version": 1,
        "purpose": "batching validation audit",
        "model": MODEL,
        "shared_state": TASK,
        "master_seed": SEED,
        "execution_seed": stable_seed(SEED, EXPERIMENT, "execution"),
        "selection": "E6 fixture indices 0..9 in each cohort; horizon 10 only",
        "formats": list(e6.FORMATS),
        "label_schemes": list(e6.LABELS),
        "assignments": 2,
        "repeats": 2,
        "batch_sizes": {"alone": 1, "mixed": 16},
        "target_questions": len(jobs) // 2,
        "decisions": len(jobs),
        "fixture_sha256": e6.digest(panel),
        "ordered_jobs_sha256": e6.digest(jobs),
        "ordered_contexts_sha256": e6.digest(contexts),
        "question_sha256": {j["id"]: e6.digest(j["question"]) for j in jobs},
        "bootstrap_resamples": BOOTSTRAPS,
        "project_cap_usd": 18.0,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in FROZEN_FILES
        },
    }


def prepare(run_dir):
    panel = source_panel()
    jobs, contexts = build_design(panel)
    path = Path(run_dir)
    if (path / "manifest.json").exists():
        return verify_manifest(path, panel, jobs, contexts)
    if path.exists() and any(path.iterdir()):
        raise ValueError("Preparation requires an empty new run directory")
    manifest = manifest_content(panel, jobs, contexts)
    manifest["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    path.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("fixtures", panel),
        ("jobs", jobs),
        ("contexts", contexts),
        ("manifest", manifest),
    ):
        (path / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    return manifest


def verify_manifest(run_dir, panel, jobs, contexts):
    path = Path(run_dir)
    if not (path / "manifest.json").exists():
        raise ValueError("Run requires a previously prepared manifest; use --prepare")
    manifest = json.loads((path / "manifest.json").read_text())
    for key, value in manifest_content(panel, jobs, contexts).items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Frozen batch control {key} changed; new namespace required"
            )
    for name, expected in (("fixtures", panel), ("jobs", jobs), ("contexts", contexts)):
        if json.loads((path / f"{name}.json").read_text()) != expected:
            raise ValueError(f"Frozen batch control {name} file changed")
    return manifest


async def execute(client, jobs, contexts, max_decisions=None):
    """Pause only between intact mode-blocks; never re-form mixed batches on resume."""
    output = client.run_dir / "diagnostics.jsonl"
    done_rows = read_jsonl(output)
    done = {r["decision_id"] for r in done_rows}
    expected = {j["id"] for j in jobs}
    if len(done) != len(done_rows) or not done <= expected:
        raise ValueError("Duplicate or foreign diagnostic decision IDs")
    collected = 0
    for start in range(0, len(jobs), BLOCK_SIZE):
        chunk, context_chunk = (
            jobs[start : start + BLOCK_SIZE],
            contexts[start : start + BLOCK_SIZE],
        )
        if len(chunk) != BLOCK_SIZE:
            raise ValueError("Incomplete mode-block")
        pending = [j for j in chunk if j["id"] not in done]
        if not pending:
            continue
        if max_decisions is not None and collected + len(pending) > max_decisions:
            break
        mode = context_chunk[0]["batch_mode"]
        if any(
            c["batch_mode"] != mode or c["block"] != context_chunk[0]["block"]
            for c in context_chunk
        ):
            raise ValueError("Mode-block metadata mismatch")
        client.batch_size = 1 if mode == "alone" else BLOCK_SIZE
        cached = [client.get_record(j["id"]) for j in chunk]
        if (
            mode == "mixed"
            and any(c is not None for c in cached)
            and any(c is None for c in cached)
        ):
            # A crash may interrupt _unpack after its durable HTTP response was saved.
            # Replay the original complete response, preserving the 16-question request.
            request_ids = {c["request_id"] for c in cached if c is not None}
            original_identity = {
                "ids": [j["id"] for j in chunk],
                "payload": {
                    "model": MODEL,
                    "state": TASK,
                    "questions": {f"q{i}": j["question"] for i, j in enumerate(chunk)},
                },
            }
            original_request_id = hashlib.sha256(
                json.dumps(original_identity, sort_keys=True).encode()
            ).hexdigest()
            if (
                request_ids != {original_request_id}
                or not client.db.execute(
                    "SELECT 1 FROM requests WHERE id=? AND response IS NOT NULL",
                    (next(iter(request_ids)),),
                ).fetchone()
            ):
                raise ValueError("Partial mixed cache has no durable complete response")
            answers = await client._request(chunk)
        else:
            answers = await client.evaluate(chunk)
        if len(answers) != BLOCK_SIZE:
            raise ValueError("Missing batch-control answers")
        if mode == "mixed" and len({a["request_id"] for a in answers}) != 1:
            raise ValueError("Mixed block did not use exactly one HTTP request")
        rows = [
            e6.score(c, a)
            for j, c, a in zip(chunk, context_chunk, answers)
            if j["id"] not in done
        ]
        append_json(output, rows)
        done.update(r["decision_id"] for r in rows)
        collected += len(rows)
        print(
            json.dumps(
                {
                    "experiment": EXPERIMENT,
                    "completed": len(done),
                    "planned": len(jobs),
                    "batch_mode": mode,
                    "accounting": client.accounting(),
                }
            ),
            flush=True,
        )


def paired_observations(rows):
    pairs = defaultdict(dict)
    for row in rows:
        pair = pairs[row["source_decision_id"]]
        if row["batch_mode"] in pair:
            raise ValueError("Duplicate paired decision")
        pair[row["batch_mode"]] = row
    observations = []
    repeat_groups = defaultdict(dict)
    for identity, modes in pairs.items():
        if set(modes) != set(MODES):
            continue
        alone, mixed = modes["alone"], modes["mixed"]
        common = {k: alone[k] for k in ("fixture_id", "cohort", "format")}
        p, q = np.array(alone["probabilities"]), np.array(mixed["probabilities"])
        metrics = {
            "probability_tv": float(np.abs(p - q).sum() / 2),
            "optimal_agreement_delta": float(mixed["optimal_agreement"])
            - float(alone["optimal_agreement"]),
            "distribution_exact_loss_delta": mixed["distribution_exact_loss"]
            - alone["distribution_exact_loss"],
        }
        if alone["advice_present"]:
            metrics["advice_adherence_delta"] = float(
                mixed["advice_adherence"]
            ) - float(alone["advice_adherence"])
            metrics["distribution_advice_adherence_delta"] = (
                mixed["distribution_advice_adherence"]
                - alone["distribution_advice_adherence"]
            )
        observations.extend(
            dict(common, metric=metric, value=value)
            for metric, value in metrics.items()
        )
        repeat_groups[identity.rsplit(":", 1)[0]][alone["repeat"]] = (common, p, q)
    for repeats in repeat_groups.values():
        if set(repeats) != {0, 1}:
            continue
        common, p0, q0 = repeats[0]
        _, p1, q1 = repeats[1]
        metrics = {
            "repeat_mean_probability_tv": np.abs((p0 + p1 - q0 - q1) / 2).sum() / 2,
            "alone_repeat_probability_tv": np.abs(p0 - p1).sum() / 2,
            "mixed_repeat_probability_tv": np.abs(q0 - q1).sum() / 2,
        }
        observations.extend(
            dict(common, metric=k, value=float(v)) for k, v in metrics.items()
        )
    return observations, sum(set(p) == set(MODES) for p in pairs.values())


def summarize(rows, expected_contexts):
    observations, n_pairs = paired_observations(rows)
    expected_rows = []
    for context in expected_contexts:
        # All scalar outcomes have the same availability rules as real scores.
        expected_rows.append(
            dict(
                context,
                probabilities=[0.5, 0.5],
                optimal_agreement=True,
                distribution_exact_loss=0.0,
                advice_adherence=True,
                distribution_advice_adherence=0.5,
            )
        )
    expected_obs, expected_pairs = paired_observations(expected_rows)
    groups, expected_groups = (
        defaultdict(lambda: defaultdict(list)),
        defaultdict(lambda: defaultdict(list)),
    )
    for records, target in ((observations, groups), (expected_obs, expected_groups)):
        for record in records:
            for format_name in ("all", record["format"]):
                target[(record["cohort"], format_name, record["metric"])][
                    record["fixture_id"]
                ].append(record["value"])
    estimates, fixture_means = [], {}
    for key, expected in sorted(expected_groups.items()):
        observed = groups[key]
        complete = {
            fixture: float(np.mean(observed[fixture]))
            for fixture, values in expected.items()
            if len(observed.get(fixture, [])) == len(values)
        }
        fixture_means[key] = complete
        estimates.append(bootstrap_record(key, complete, len(expected)))
    interactions = []
    for (cohort, format_name, metric), values in sorted(fixture_means.items()):
        if format_name in ("all", "nested") or metric.endswith("repeat_probability_tv"):
            continue
        reference = fixture_means[(cohort, "nested", metric)]
        paired = {f: v - reference[f] for f, v in values.items() if f in reference}
        interactions.append(bootstrap_record((cohort, format_name, metric), paired, 10))
    return {
        "experiment": EXPERIMENT,
        "completed_decisions": len(rows),
        "planned_decisions": len(expected_contexts),
        "matched_questions": n_pairs,
        "planned_matched_questions": expected_pairs,
        "complete": len(rows) == len(expected_contexts),
        "estimates": estimates,
        "format_minus_nested_contrasts": interactions,
        "interpretation": "Descriptive validation audit, not a new mechanism claim. Signed differences are mixed minus alone. TV includes service variation. Repeat-mean TV and within-mode repeat TV are secondary diagnostics, not noise-corrected causal effects. Fixture bootstrap intervals do not capture cross-fixture dependence induced by mixed requests. Ten fixtures per cohort; exploratory unadjusted intervals, no equivalence or significance claim.",
    }


def bootstrap_record(key, values, planned):
    cohort, format_name, metric = key
    array = np.array([values[f] for f in sorted(values)])
    result = {
        "cohort": cohort,
        "format": format_name,
        "metric": metric,
        "complete_fixtures": len(array),
        "planned_fixtures": planned,
        "mean": None,
        "ci95": None,
    }
    if len(array):
        rng = np.random.default_rng(stable_seed(SEED, EXPERIMENT, "bootstrap", *key))
        boot = array[rng.integers(0, len(array), size=(BOOTSTRAPS, len(array)))].mean(
            axis=1
        )
        result.update(
            mean=float(array.mean()), ci95=np.quantile(boot, [0.025, 0.975]).tolist()
        )
    return result


def report(run_dir, contexts):
    path = Path(run_dir)
    rows = read_jsonl(path / "diagnostics.jsonl")
    expected = {c["decision_id"]: c for c in contexts}
    if len({r["decision_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate diagnostic IDs")
    for row in rows:
        context = expected.get(row["decision_id"])
        if context is None or any(
            row.get(k) != v for k, v in context.items() if k != "order"
        ):
            raise ValueError("Diagnostic context differs from frozen design")
    result = summarize(rows, contexts)
    (path / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Batch control validation audit",
        "",
        result["interpretation"],
        "",
        f"Completed {len(rows)}/{len(contexts)} decisions; matched {result['matched_questions']}/{result['planned_matched_questions']} questions.",
        "",
        "Only fixtures complete for each metric contribute to its estimate.",
        "",
        "| Cohort | Format | Metric | Fixtures | Mean | 95% fixture bootstrap |",
        "|---|---|---|---:|---:|---|",
    ]
    for record in result["estimates"]:
        ci = record["ci95"]
        lines.append(
            f"| {record['cohort']} | {record['format']} | {record['metric']} | {record['complete_fixtures']} | "
            + (
                f"{record['mean']:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] |"
                if ci
                else "— | — |"
            )
        )
    lines.extend(
        [
            "",
            "Format-minus-nested contrasts, including exploratory mode × format differences, are in report.json.",
        ]
    )
    (path / "report.md").write_text("\n".join(lines) + "\n")
    return result


async def run(args):
    if args.prepare:
        manifest = prepare(args.run_dir)
        print(
            json.dumps(
                {k: v for k, v in manifest.items() if k != "question_sha256"}, indent=2
            )
        )
        return
    panel = source_panel()
    jobs, contexts = build_design(panel)
    verify_manifest(args.run_dir, panel, jobs, contexts)
    if args.report:
        result = report(args.run_dir, contexts)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ("estimates", "format_minus_nested_contrasts")
                },
                indent=2,
            )
        )
        return
    if Path(args.run_dir).resolve().parent != ROOT / "artifacts":
        raise ValueError(
            "Live audit must use a project artifacts sibling directory for global ledger accounting"
        )
    client = JevClient(
        args.run_dir,
        cap=args.cap,
        rate=args.rate,
        concurrency=args.concurrency,
        batch_size=1,
    )
    try:
        await execute(client, jobs, contexts, args.max_decisions)
    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "run", "report"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--run-dir", default="artifacts/batch_control_v1")
    parser.add_argument("--cap", type=float, default=18.0)
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--max-decisions",
        type=int,
        help="Pause at mode-block boundary, at most this many new rows",
    )
    args = parser.parse_args()
    if not 0 < args.cap <= 18:
        parser.error("--cap must be positive and no larger than the global $18 cap")
    if args.rate <= 0 or args.concurrency < 1:
        parser.error("Rate and concurrency must be positive")
    if args.max_decisions is not None and args.max_decisions < BLOCK_SIZE:
        parser.error("--max-decisions must allow at least one 16-decision mode-block")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
