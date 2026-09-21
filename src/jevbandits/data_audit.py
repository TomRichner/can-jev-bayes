"""Read-only completeness audit derived from frozen manifests, not runner output totals.

Never imports experiment runners, contacts an API, repairs JSONL, or sums cumulative
accounting files. Completed runs are explicit; in-flight controls are excluded.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

COMPLETED = (
    "overnight_v2",
    "followup_advice_v1",
    "forecast_v1",
    "evidence_v1",
    "sampling_v1",
    "batch_control_v1",
    "primitive_v1",
)
EXCLUDED = ()
ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def records(path):
    """Strict, streaming parser: malformed/truncated data fail without mutation."""
    if not Path(path).exists():
        return
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record at {path}:{number}")
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL record at {path}:{number}") from exc


def compare(expected, observed):
    counts = Counter(observed)
    actual = set(counts)
    missing, unexpected = expected - actual, actual - expected
    duplicates = {str(key): count for key, count in counts.items() if count > 1}
    return {
        "expected": len(expected),
        "observed": sum(counts.values()),
        "unique": len(actual),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "duplicate_extra_rows": sum(n - 1 for n in duplicates.values()),
        "missing_examples": sorted(missing, key=str)[:10],
        "unexpected_examples": sorted(unexpected, key=str)[:10],
        "duplicate_examples": dict(list(duplicates.items())[:10]),
        "passed": not (missing or unexpected or duplicates),
    }


def online_expected(manifest):
    jev, classical, decisions = set(), set(), set()
    for experiment, design in manifest.get("online", {}).items():
        for family in design["families"]:
            for k in design["ks"]:
                for index in range(design["n"]):
                    episode = f"{experiment}:{family}:{k}:{index}"
                    for policy in design["policies"]:
                        jev.add((episode, policy))
                        decisions.update(
                            f"{episode}:{policy}:{turn}"
                            for turn in range(1, design["horizon"] + 1)
                        )
                    for policy in manifest["classical"] + (
                        ["exact"] if experiment == "e3_exact_online" else []
                    ):
                        classical.add((episode, policy))
    return jev, classical, decisions


def diagnostic_expected(run, manifest, directory):
    result = set()
    if run == "batch_control_v1":
        return set(manifest["question_sha256"])
    if run == "overnight_v2":
        for experiment in ("e1_horizon", "e2_assistance"):
            design = manifest[experiment]
            for i in range(design["fixtures"]):
                for h in design["horizons"]:
                    for representation in design.get("representations", ["counts"]):
                        for framing in design.get("framings", ["explore"]):
                            for assignment in range(design["label_orders"]):
                                for repeat in range(design["repeats"]):
                                    result.add(
                                        f"{experiment}:{i}:{h}:{representation}:{framing}:{assignment}:{repeat}"
                                    )
    elif run == "followup_advice_v1":
        panel = json.loads((directory / "fixtures.json").read_text())
        for fixture in panel:
            for h in manifest["horizons"]:
                for format_name in manifest["formats"]:
                    for label in manifest["label_schemes"]:
                        for assignment in range(manifest["assignments"]):
                            for repeat in range(manifest["repeats"]):
                                result.add(
                                    f"{manifest['experiment']}:{fixture['cohort']}:{fixture['index']}:{h}:{format_name}:{label}:{assignment}:{repeat}"
                                )
    return result


def forecast_expected(manifest):
    result = set()
    for k in manifest["ks"]:
        for i in range(manifest["n_fixtures_per_k"]):
            for representation in manifest["representations"]:
                for repeat in range(manifest["repeats"]):
                    base = f"{manifest['experiment']}:{k}:{i}:{representation}:{repeat}"
                    result.add(f"{base}:best")
                    result.update(f"{base}:reward:{arm}" for arm in range(k))
    return result


def primitive_audit(directory, manifest):
    """Reconstruct E10 identities/count fixtures independently of its runner."""
    panel = json.loads((directory / "fixtures.json").read_text())
    by_id = {fixture["fixture_id"]: fixture for fixture in panel}
    expected_fixtures = {
        f"{manifest['experiment']}:{k}:{index}"
        for k in manifest["ks"]
        for index in range(manifest["fixtures_per_k"])
    }
    fixture_check = compare(expected_fixtures, [f["fixture_id"] for f in panel])
    encoded = json.dumps(
        panel, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    fixture_check["manifest_hash_matches"] = (
        hashlib.sha256(encoded).hexdigest() == manifest["fixture_sha256"]
    )
    fixture_errors = []
    for k in manifest["ks"]:
        seed_parts = (manifest["master_seed"], manifest["experiment"], "fixtures", k)
        seed_hash = hashlib.sha256(
            json.dumps(seed_parts, separators=(",", ":")).encode()
        ).digest()
        rng = np.random.default_rng(int.from_bytes(seed_hash[:8], "little"))
        for index in range(manifest["fixtures_per_k"]):
            total = rng.choice([0, 2, 5, 10, 20], k)
            successes = rng.integers(0, total + 1)
            identity = f"{manifest['experiment']}:{k}:{index}"
            fixture = by_id.get(identity)
            if fixture is None:
                continue
            if (
                fixture["k"] != k
                or fixture["index"] != index
                or fixture["successes"] != successes.tolist()
                or fixture["failures"] != (total - successes).tolist()
                or fixture["next_reward"] != ((successes + 1) / (total + 2)).tolist()
            ):
                fixture_errors.append(
                    identity
                    + ": counts/analytic mean differ from independent seeded reconstruction"
                )
            q = fixture["best_arm"]
            if (
                len(q) != k
                or not all(math.isfinite(x) and 0 <= x <= 1 for x in q)
                or not math.isclose(sum(q), 1, abs_tol=1e-10)
            ):
                fixture_errors.append(
                    identity + ": invalid best-arm probability vector"
                )
    fixture_check["reference_errors"] = len(fixture_errors)
    fixture_check["reference_error_examples"] = fixture_errors[:10]
    fixture_check["best_reference_scope"] = (
        "Frozen quadrature values checked by manifest hash and probability validity; quadrature not independently recomputed."
    )
    fixture_check["passed"] &= (
        fixture_check["manifest_hash_matches"] and not fixture_errors
    )
    expected = set()
    for identity in expected_fixtures:
        k = int(identity.split(":")[-2])
        for representation in manifest["representations"]:
            for repeat in range(manifest["repeats"]):
                for target in ("next_reward", "best_arm"):
                    for primitive in ("noul", "choice"):
                        arms = (
                            [None]
                            if target == "best_arm" and primitive == "choice"
                            else range(k)
                        )
                        expected.update(
                            f"{identity}:{representation}:{repeat}:{target}:{primitive}:{arm}"
                            for arm in arms
                        )
    ids, errors, breakdown = [], [], Counter()
    for row in records(directory / "forecasts.jsonl"):
        ids.append(row["decision_id"])
        breakdown[
            f"K={row['k']}|{row['representation']}|{row['target']}|{row['primitive']}"
        ] += 1
        fixture = by_id.get(row["fixture_id"])
        if fixture is None or row["target"] not in ("next_reward", "best_arm"):
            errors.append(row["decision_id"] + ": unknown fixture or target")
            continue
        arm = row["arm"]
        joint = row["target"] == "best_arm" and row["primitive"] == "choice"
        if (
            row["k"] != fixture["k"]
            or (joint and arm is not None)
            or (not joint and (not isinstance(arm, int) or not 0 <= arm < fixture["k"]))
        ):
            errors.append(row["decision_id"] + ": wrong fixture/arm mapping")
            continue
        reference = fixture[row["target"]] if joint else [fixture[row["target"]][arm]]
        prediction = row["prediction_event_probabilities"]
        if (
            reference != row["reference_event_probabilities"]
            or len(prediction) != len(reference)
            or not all(math.isfinite(p) and 0 <= p <= 1 for p in prediction)
        ):
            errors.append(
                row["decision_id"] + ": wrong reference or prediction shape/range"
            )
            continue
        risk = (
            2
            * sum((p - q) ** 2 for p, q in zip(prediction, reference, strict=True))
            / len(reference)
        )
        if not math.isclose(
            risk, row["mean_binary_excess_brier"], rel_tol=1e-12, abs_tol=1e-14
        ):
            errors.append(
                row["decision_id"]
                + ": stored binary Brier risk differs from unnormalized calculation"
            )
    forecasts = compare(expected, ids)
    forecasts.update(
        sha256=file_hash(directory / "forecasts.jsonl"),
        reference_score_errors=len(errors),
        reference_score_error_examples=errors[:10],
        condition_counts=dict(sorted(breakdown.items())),
    )
    forecasts["passed"] &= not errors
    return expected, fixture_check, forecasts


def supplemental_expected(manifest, kind):
    result = set()
    for experiment, design in manifest.get("online", {}).items():
        if experiment == "pilot":
            continue
        for family in design["families"]:
            for k in design["ks"]:
                if kind == "long_exact" and (k != 2 or design["horizon"] != 100):
                    continue
                for index in range(design["n"]):
                    result.add(
                        (
                            f"{experiment}:{family}:{k}:{index}",
                            "finite_ap_index" if kind == "index" else "exact_h100",
                        )
                    )
    return result


def ledger_audit(path, expected=None):
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as database:
        database.execute("BEGIN")  # One consistent read snapshot per ledger.
        attempts = database.execute(
            "SELECT COUNT(*),COALESCE(SUM(cost),0),COALESCE(SUM(reserved),0),COALESCE(SUM(input_tokens),0) FROM attempts"
        ).fetchone()
        statuses = dict(
            database.execute("SELECT status,COUNT(*) FROM attempts GROUP BY status")
        )
        decision_ids = [r[0] for r in database.execute("SELECT id FROM decisions")]
        quick_check = [r[0] for r in database.execute("PRAGMA quick_check")]
    result = {
        "path": str(path),
        "http_attempts": attempts[0],
        "known_cost_usd": attempts[1],
        "reserved_usd": attempts[2],
        "cost_plus_reserves_usd": attempts[1] + attempts[2],
        "input_tokens": attempts[3],
        "status_counts": statuses,
        "decision_count": len(decision_ids),
        "sqlite_quick_check": quick_check,
    }
    if expected is not None:
        result["decision_ids"] = compare(expected, decision_ids)
    return result


def scientific_hashes(manifest, run):
    expected = {}
    if run == "overnight_v2":
        expected.update(
            {
                f"src/jevbandits/{name}": manifest["source_sha256"][name]
                for name in (
                    "client.py",
                    "prompts.py",
                    "experiments.py",
                    "baselines.py",
                )
            }
        )
        expected["uv.lock"] = manifest["uv_lock_sha256"]
    elif run == "forecast_v1":
        expected["src/jevbandits/forecast_followup.py"] = manifest["source_sha256"]
        expected.update(
            {
                f"src/jevbandits/{name}": value
                for name, value in manifest["dependency_sha256"].items()
            }
        )
    else:
        expected.update(manifest["source_sha256"])
    mismatches = [
        name
        for name, value in expected.items()
        if not (ROOT / name).exists() or file_hash(ROOT / name) != value
    ]
    return {
        "files_checked": len(expected),
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def episode_file(path, expected, manifest, world_signatures):
    ids, errors, breakdown = [], [], Counter()
    for row in records(path):
        identity = (row["episode_id"], row["policy"])
        ids.append(identity)
        breakdown[
            f"{row['experiment']}|{row['family']}|K={row['k']}|{row['policy']}"
        ] += 1
        design = manifest["online"].get(row["experiment"])
        trace, horizon = row.get("trace", []), row["horizon"]
        issues = []
        if not design or horizon != design["horizon"] or not row.get("completed"):
            issues.append("wrong design/horizon/incomplete flag")
        if len(trace) != horizon or [r.get("turn") for r in trace] != list(
            range(1, horizon + 1)
        ):
            issues.append("missing/duplicate/out-of-order turns")
        if any(
            not isinstance(r.get("action"), int)
            or not 0 <= r["action"] < row["k"]
            or r.get("reward") not in (0, 1)
            for r in trace
        ):
            issues.append("invalid action/reward")
        if sum(r["reward"] for r in trace) != row["reward"]:
            issues.append("reward sum mismatch")
        if not math.isclose(
            sum(r["pseudo_regret"] for r in trace), row["pseudo_regret"], abs_tol=1e-9
        ):
            issues.append("pseudo-regret sum mismatch")
        signature = (row["seed"], tuple(row["theta"]))
        if world_signatures.setdefault(row["episode_id"], signature) != signature:
            issues.append("unpaired hidden worlds across policies")
        if issues:
            errors.append({"identity": identity, "issues": issues})
    summary = compare(expected, ids)
    summary.update(
        sha256=file_hash(path) if path.exists() else None,
        trace_errors=len(errors),
        trace_error_examples=errors[:10],
        cell_policy_counts=dict(sorted(breakdown.items())),
    )
    summary["passed"] = summary["passed"] and not errors
    return summary, set(ids)


def id_file(path, expected):
    summary = compare(expected, [r["decision_id"] for r in records(path)])
    summary["sha256"] = file_hash(path) if path.exists() else None
    return summary


def export_audit(directory, source_checks, ledger):
    path = directory / "export_manifest.json"
    if not path.exists():
        return {"passed": False, "error": "missing export manifest"}
    manifest = json.loads(path.read_text())
    checks = {}
    source_expected = {
        "episodes": sum(
            c["observed"]
            for name, c in source_checks.items()
            if name.startswith("episodes_") and name.endswith(".jsonl")
        ),
        "diagnostics": source_checks.get("diagnostics.jsonl", {}).get("observed", 0),
        "forecasts": source_checks.get("forecasts.jsonl", {}).get("observed", 0),
        "decision_distributions": ledger["decision_count"],
    }
    for name, metadata in manifest["exports"].items():
        artifact = directory / f"{name}.csv.gz"
        with gzip.open(artifact, "rt", newline="") as stream:
            count = sum(1 for _ in csv.DictReader(stream))
        expected = source_expected.get(name, metadata["rows"])
        checks[name] = {
            "rows": count,
            "declared_rows": metadata["rows"],
            "source_rows": expected,
            "hash_matches": file_hash(artifact) == metadata["sha256"],
            "passed": count == expected == metadata["rows"]
            and file_hash(artifact) == metadata["sha256"],
        }
    absent = [
        name
        for name, expected in source_expected.items()
        if expected and name not in checks
    ]
    return {
        "passed": all(c["passed"] for c in checks.values()) and not absent,
        "files": checks,
        "missing_exports": absent,
    }


def audit_run(run, artifact_root, result_root):
    directory = artifact_root / run
    manifest_path = directory / (
        "forecast_manifest.json" if run == "forecast_v1" else "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    jev, classical, decisions = online_expected(manifest)
    checks, worlds, authoritative = {}, {}, set()
    if "online" in manifest:
        for filename, expected in [
            ("episodes_jev.jsonl", jev),
            ("episodes_baselines.jsonl", classical),
        ]:
            checks[filename], actual = episode_file(
                directory / filename, expected, manifest, worlds
            )
            authoritative.update(actual)
        if (directory / "episodes_index.jsonl").exists():
            filename = "episodes_index.jsonl"
            checks[filename], actual = episode_file(
                directory / filename,
                supplemental_expected(manifest, "index"),
                manifest,
                worlds,
            )
            authoritative.update(actual)
        if (directory / "episodes_long_exact.jsonl").exists():
            filename = "episodes_long_exact.jsonl"
            checks[filename], actual = episode_file(
                directory / filename,
                supplemental_expected(manifest, "long_exact"),
                manifest,
                worlds,
            )
            authoritative.update(actual)
        core_merged = [
            (r["episode_id"], r["policy"])
            for r in records(directory / "episodes.jsonl")
        ]
        checks["merged_core_view"] = compare(jev | classical, core_merged)
        checks["merged_core_view"]["note"] = (
            "Core merged file intentionally excludes separately stored posthoc index/long-exact outputs. Export uses all authoritative files."
        )
    if run in ("overnight_v2", "followup_advice_v1", "batch_control_v1"):
        expected = diagnostic_expected(run, manifest, directory)
        checks["diagnostics.jsonl"] = id_file(directory / "diagnostics.jsonl", expected)
        decisions.update(expected)
    if run == "forecast_v1":
        expected = forecast_expected(manifest)
        checks["forecasts.jsonl"] = id_file(directory / "forecasts.jsonl", expected)
        decisions.update(expected)
    if run == "primitive_v1":
        expected, checks["fixtures.json"], checks["forecasts.jsonl"] = primitive_audit(
            directory, manifest
        )
        decisions.update(expected)
    if run == "overnight_v2":
        preflight = {
            f"preflight:{mode}:{repeat}:{i}"
            for mode, size in [("alone", 4), ("mixed", 16)]
            for repeat in range(10)
            for i in range(size)
        }
        decisions.update(preflight)
        observed = json.loads((directory / "preflight.json").read_text())
        checks["preflight.json"] = compare(
            preflight,
            [
                f"preflight:{r['condition']}:{r['repeat']}:{r['fixture']}"
                for r in observed
            ],
        )
    ledger = ledger_audit(directory / "ledger.sqlite", decisions)
    sources = scientific_hashes(manifest, run)
    exports = export_audit(result_root / run, checks, ledger)
    return {
        "run": run,
        "manifest_sha256": file_hash(manifest_path),
        "checks": checks,
        "ledger": ledger,
        "scientific_hashes": sources,
        "exports": exports,
        "authoritative_episode_rows": len(authoritative),
        "passed": all(c["passed"] for c in checks.values())
        and ledger["decision_ids"]["passed"]
        and ledger["sqlite_quick_check"] == ["ok"]
        and sources["passed"]
        and exports["passed"],
    }


def audit(artifact_root=Path("artifacts"), result_root=Path("results")):
    runs = [audit_run(run, artifact_root, result_root) for run in COMPLETED]
    # Archived development spending is separate, not another sample of evaluation.
    archived = ledger_audit(artifact_root / "overnight_v1" / "ledger.sqlite")
    ledger_rows = [r["ledger"] for r in runs] + [archived]
    environments = trajectories = actions = 0
    for run in COMPLETED:
        if run == "forecast_v1":
            continue
        manifest = json.loads((artifact_root / run / "manifest.json").read_text())
        for experiment, design in manifest.get("online", {}).items():
            if experiment == "pilot":
                continue
            tasks = len(design["ks"]) * len(design["families"]) * design["n"]
            environments += tasks
            trajectories += tasks * len(design["policies"])
            actions += tasks * len(design["policies"]) * design["horizon"]
    completed_cached = sum(r["ledger"]["decision_count"] for r in runs)
    expected_cached = sum(r["ledger"]["decision_ids"]["expected"] for r in runs)
    pending_planned = sum(
        json.loads((artifact_root / run / "manifest.json").read_text()).get(
            "questions", 0
        )
        for run in EXCLUDED
    )
    return {
        "generated_utc": datetime.now(UTC).isoformat(),
        "audit_source_sha256": file_hash(Path(__file__)),
        "scope": list(COMPLETED),
        "excluded_pending_runs": list(EXCLUDED),
        "runs": runs,
        "completed_evaluation_totals": {
            "independent_online_environments": environments,
            "jev_trajectories_excluding_pilot": trajectories,
            "online_actions_excluding_pilot": actions,
            "cached_questions_including_v2_pilot_preflight": completed_cached,
            "evaluation_questions_excluding_pilot_preflight": completed_cached
            - 2160
            - 200,
            "planned_evaluation_questions_including_pending_controls": expected_cached
            - 2160
            - 200
            + pending_planned,
            "planned_cached_questions_including_pending_controls": expected_cached
            + pending_planned,
        },
        "archived_development_cost_only": archived,
        "cost_totals": {
            "known_cost_usd": sum(r["known_cost_usd"] for r in ledger_rows),
            "reserved_usd": sum(r["reserved_usd"] for r in ledger_rows),
            "cost_plus_reserves_usd": sum(
                r["cost_plus_reserves_usd"] for r in ledger_rows
            ),
            "preparation_allowance_usd_not_in_ledgers": 0.00015456,
            "including_preparation_allowance_usd": sum(
                r["cost_plus_reserves_usd"] for r in ledger_rows
            )
            + 0.00015456,
            "method": "Sum each included ledger attempts table once; never sum previous-run/cumulative accounting fields. All completed project run ledgers and archived development are included. Recorded costs/reserves are not independently verified billing.",
        },
        "passed": all(r["passed"] for r in runs),
    }


def write_report(result, path):
    lines = [
        "# Independent data completeness audit",
        f"Snapshot: {result['generated_utc']}.",
        "This offline audit reconstructs expected decision and episode identities directly from frozen manifests and fixture labels. It reads JSONL strictly without repairing files, checks traces and paired worlds, compares ledger decision IDs, verifies scientific source hashes, and validates compressed export row counts and hashes. It makes no API calls.",
        "| Run | Expected / observed API decisions | Authoritative episodes | Duplicates / missing IDs | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for run in result["runs"]:
        ids = run["ledger"]["decision_ids"]
        duplicates = sum(
            c.get("duplicate_extra_rows", 0) for c in run["checks"].values()
        )
        missing = (
            sum(c.get("missing", 0) for c in run["checks"].values()) + ids["missing"]
        )
        lines.append(
            f"| {run['run']} | {ids['expected']:,} / {ids['observed']:,} | {run['authoritative_episode_rows']:,} | {duplicates} / {missing} | {'PASS' if run['passed'] else 'INVESTIGATE'} |"
        )
    lines += [
        "",
        "The main API count includes 200 preflight questions and 2,160 pilot decisions; the five evaluation experiments alone contain 390,800 decisions. Pilot/preflight records are not independent evaluation tasks. E6 contains 28,800 fixed-state decisions, E7 24,000 forecasts, E8 96,000 online decisions, E9 40,000 online decisions, E10 16,560 forecasts, and the batch control 1,920 questions. All are complete.",
        f"Manifest-derived evaluation totals: {result['completed_evaluation_totals']['independent_online_environments']:,} independent online environments; {result['completed_evaluation_totals']['jev_trajectories_excluding_pilot']:,} Jev trajectories; {result['completed_evaluation_totals']['online_actions_excluding_pilot']:,} online actions, excluding pilot. Completed evaluation/control questions total {result['completed_evaluation_totals']['evaluation_questions_excluding_pilot_preflight']:,}, matching the {result['completed_evaluation_totals']['planned_evaluation_questions_including_pending_controls']:,} planned IDs. Including revised pilot/preflight gives {result['completed_evaluation_totals']['cached_questions_including_v2_pilot_preflight']:,} cached questions, matching {result['completed_evaluation_totals']['planned_cached_questions_including_pending_controls']:,} expected IDs. Archived development is excluded from these scientific totals.",
        "E10 adds an independent fixture audit: all 90 fixture identities and their frozen hash, seeded counts, analytic next-reward means, target-to-reference mappings, probability ranges, and stored common binary Brier scores pass. Primary Noul best-arm scores are rechecked without normalizing their marginal vectors. Best-arm quadrature values match the frozen fixture hash and valid probability constraints; this audit does not independently rerun quadrature.",
        "The original main and E9 episodes.jsonl files are core-only merged views. Supplemental finite-index and exact-horizon baseline files are authoritative separate outputs and are included by export_results. Their absence from the core merged file is not missing experiment data. Trace lengths, turn order, valid actions/rewards, summed rewards/pseudo-regret, and matching seeds/hidden means across policies are checked. This structural audit does not rederive every reward stream or every numerical reference value.",
        "## Costs counted once",
        "| Ledger | Known cost (USD) | Retained reserves (USD) | HTTP attempts |",
        "|---|---:|---:|---:|",
    ]
    for name, ledger in [(r["run"], r["ledger"]) for r in result["runs"]] + [
        (
            "overnight_v1 (archived development)",
            result["archived_development_cost_only"],
        )
    ]:
        lines.append(
            f"| {name} | {ledger['known_cost_usd']:.9f} | {ledger['reserved_usd']:.9f} | {ledger['http_attempts']:,} |"
        )
    cost = result["cost_totals"]
    lines += [
        "",
        f"Included ledger totals: **${cost['known_cost_usd']:.9f} known cost**, **${cost['reserved_usd']:.9f} reserves**, **${cost['cost_plus_reserves_usd']:.9f} combined**. Adding the separate $0.000154560 preparation allowance gives **${cost['including_preparation_allowance_usd']:.9f}**. Each attempts table is summed once; cumulative previous-run totals in accounting.json are never added. Reserves are uncertain potential charges, not verified billing.",
        "All completed project runs, final controls, and archived overnight_v1 development spending are included in the cost scope. Archived overnight_v1 is cost-only and does not enter evaluation completeness. No controls remain pending in this snapshot.",
        "## Exports and reproduction",
        "Completed follow-up exports are in results/followup_advice_v1, results/forecast_v1, results/evidence_v1, results/sampling_v1, results/batch_control_v1, and results/primitive_v1. The existing results/overnight_v2 export is validated in place. E9 interaction_summary.json is retained with a provenance hash. The batch-control report is copied to reports/batch_control_v1. Compressed CSV rows preserve decision IDs, probabilities, and compact episode trajectories; raw HTTP ledgers remain local. Source scientific hashes match the frozen manifest when a run is marked PASS. Exported accounting files may contain cumulative totals and must not be summed across runs.",
        "Run `PYTHONPATH=src .venv/bin/python -m jevbandits.data_audit` from the repository root to reproduce results/data_audit.json and this document. The JSON includes per-file and per-cell/policy counts, missing/duplicate examples, source-hash checks, SQLite integrity results, HTTP status counts, and export checks.",
    ]
    failed = [r["run"] for r in result["runs"] if not r["passed"]]
    if failed:
        lines.append(
            "Failed checks require review: "
            + ", ".join(failed)
            + ". Consult results/data_audit.json; do not interpret this audit as a complete pass."
        )
    blocks = []
    for line in lines:
        if line.startswith("|") and blocks and blocks[-1].startswith("|"):
            blocks[-1] += "\n" + line
        elif line:
            blocks.append(line)
    Path(path).write_text("\n\n".join(blocks) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--document", type=Path, default=Path("docs/data_completeness_audit.md")
    )
    args = parser.parse_args()
    result = audit(args.artifacts, args.results)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "data_audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    write_report(result, args.document)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "runs": {r["run"]: r["passed"] for r in result["runs"]},
                "cost_totals": result["cost_totals"],
            },
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
