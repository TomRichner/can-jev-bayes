"""E9 fresh confirmation of the sampling-by-arm-count interaction; preparation never opens an API client."""

import argparse
import asyncio
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from .baselines import select_action
from .client import JevClient
from .experiments import (
    CLASSICAL,
    SEED,
    Episode,
    append_json,
    environment,
    merge_episodes,
    read_jsonl,
)
from .finite_index import DEFAULT_TOLERANCE, POLICY_NAME, select_index_action
from .prompts import MODEL, OBJECTIVES, TASK, observation, question, stable_seed

EXPERIMENT = "e9_sampling_regimes"
POLICIES = {
    "bayes_direct": ("bayes", "explore", "direct"),
    "bayes_sample": ("bayes", "explore", "sample"),
}
DESIGN = {
    "ks": [2, 10],
    "families": ["prior"],
    "n": 100,
    "horizon": 100,
    "policies": list(POLICIES),
}
BASELINES = [*CLASSICAL, POLICY_NAME]
PRIMARY_CONTRAST = "(sample-direct at K=10) - (sample-direct at K=2)"
ROOT = Path(__file__).resolve().parents[2]
FROZEN_FILES = (
    "src/jevbandits/sampling_followup.py",
    "tests/test_sampling_followup.py",
    "docs/e9_sampling_protocol.md",
    "src/jevbandits/prompts.py",
    "src/jevbandits/experiments.py",
    "src/jevbandits/baselines.py",
    "src/jevbandits/client.py",
    "src/jevbandits/finite_index.py",
    "uv.lock",
)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class SamplingEpisode(Episode):
    """Use the original Episode payload, streams, and per-turn action RNG."""

    def summary(self):
        record = super().summary()
        if self.policy in POLICIES:
            representation, framing, rule = POLICIES[self.policy]
            record.update(
                representation=representation, framing=framing, action_rule=rule
            )
        if self.policy == POLICY_NAME:
            record.update(
                index_tolerance=DEFAULT_TOLERANCE,
                index_ambiguous_ranking_fraction=float(
                    np.mean([row["index_ambiguous_ranking"] for row in self.trace])
                ),
            )
        record["followup_status"] = (
            "E4-motivated preregistered confirmation; fresh held-out tasks"
        )
        return record


def task_panel():
    """Evaluator-only worlds. These records must never become question payloads."""
    panel = []
    for family in DESIGN["families"]:
        for k in DESIGN["ks"]:
            for index in range(DESIGN["n"]):
                seed, theta, outcomes = environment(
                    EXPERIMENT, family, k, index, DESIGN["horizon"]
                )
                panel.append(
                    {
                        "episode_id": f"{EXPERIMENT}:{family}:{k}:{index}",
                        "family": family,
                        "k": k,
                        "index": index,
                        "seed": seed,
                        "theta": theta.tolist(),
                        "outcomes_sha256": digest(outcomes.tolist()),
                    }
                )
    return panel


def prompt_fixtures():
    """Freeze representative payloads; the source hash freezes adaptive payloads."""
    fixtures = []
    for k in DESIGN["ks"]:
        for observed in (False, True):
            s = [i % 4 if observed else 0 for i in range(k)]
            f = [(2 * i + 1) % 5 if observed else 0 for i in range(k)]
            for remaining in (1, 50, 100):
                for policy, (representation, framing, _) in POLICIES.items():
                    fixtures.append(
                        {
                            "k": k,
                            "observed": observed,
                            "policy": policy,
                            "question": question(
                                observation(s, f, remaining, representation), framing
                            ),
                        }
                    )
    return fixtures


def manifest_content(panel, prompts):
    tasks = len(panel)
    return {
        "experiment": EXPERIMENT,
        "version": 1,
        "master_seed": SEED,
        "model": MODEL,
        "online": {EXPERIMENT: DESIGN},
        "policies": POLICIES,
        "classical": BASELINES,
        "tasks": tasks,
        "jev_episodes": tasks * len(POLICIES),
        "baseline_episodes": tasks * len(BASELINES),
        "decisions": tasks * len(POLICIES) * DESIGN["horizon"],
        "shared_state": TASK,
        "objective": OBJECTIVES["explore"],
        "action_rules": {
            "bayes_direct": "backend_choice",
            "bayes_sample": "host_categorical",
        },
        "task_panel_sha256": digest(panel),
        "prompt_fixtures_sha256": digest(prompts),
        "primary_analysis": {
            "metric": "pseudo_regret",
            "single_interaction": PRIMARY_CONTRAST,
            "hypothesis": "negative",
            "population": "matched independent Beta(1,1), K=2 versus K=10, T=100",
            "resampling": "10000 paired whole episodes within each cell",
            "pointwise_quantiles": [0.025, 0.975],
            "per_cell": "secondary pointwise 95% intervals",
            "missing": "report missing pairs; incomplete pooled design is descriptive only",
        },
        "finite_ap_index": {
            "tolerance": DEFAULT_TOLERANCE,
            "status": "finite-horizon single-arm index heuristic; not joint optimal",
            "ties": "uniform midpoint maxima within absolute 1e-12",
        },
        "collection": {
            "default_cap_usd": 18,
            "includes": "all sibling ledgers plus conservative reserves",
            "projected_incremental_usd": 1.5,
            "projection_status": "rough planning estimate, not a spending authorization",
            "gate": "after E1-E8; parent budget review; no automatic credit purchase",
        },
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in FROZEN_FILES
        },
    }


def verify_manifest(run_dir, panel=None, prompts=None):
    path = Path(run_dir)
    if not (path / "manifest.json").exists():
        raise ValueError("Run requires --prepare first")
    panel = task_panel() if panel is None else panel
    prompts = prompt_fixtures() if prompts is None else prompts
    expected = json.loads(json.dumps(manifest_content(panel, prompts)))
    existing = json.loads((path / "manifest.json").read_text())
    for key, value in expected.items():
        if existing.get(key) != value:
            raise ValueError(f"Frozen E9 {key} changed; use a new namespace/amendment")
    for filename, expected_records in [
        ("tasks.json", panel),
        ("prompt_fixtures.json", prompts),
    ]:
        if json.loads((path / filename).read_text()) != expected_records:
            raise ValueError(f"Frozen E9 {filename} changed")
    return existing


def prepare(run_dir):
    path = Path(run_dir)
    panel, prompts = task_panel(), prompt_fixtures()
    if (path / "manifest.json").exists():
        return verify_manifest(path, panel, prompts)
    if path.exists() and any(path.iterdir()):
        raise ValueError("Preparation requires an empty new run directory")
    content = manifest_content(panel, prompts)
    content["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    path.mkdir(parents=True, exist_ok=True)
    for filename, value in [
        ("tasks.json", panel),
        ("prompt_fixtures.json", prompts),
        ("manifest.json", content),
    ]:
        (path / filename).write_text(json.dumps(value, indent=2) + "\n")
    return content


async def execute(client, run_dir, *, only_baselines=False):
    """Replay deterministic histories from the durable, payload-validated cache."""
    verify_manifest(run_dir)
    mode = "baselines" if only_baselines else "jev"
    output = Path(run_dir) / f"episodes_{mode}.jsonl"
    done = {(row["episode_id"], row["policy"]) for row in read_jsonl(output)}
    policies = BASELINES if only_baselines else list(POLICIES)
    episodes = [
        SamplingEpisode(
            EXPERIMENT,
            task["family"],
            task["k"],
            task["index"],
            DESIGN["horizon"],
            policy,
        )
        for task in task_panel()
        for policy in policies
        if (task["episode_id"], policy) not in done
    ]
    if only_baselines:
        for index, ep in enumerate(episodes):
            for turn in range(1, ep.horizon + 1):
                diagnostics = {}
                if ep.policy == POLICY_NAME:
                    action, diagnostics = select_index_action(
                        ep.s,
                        ep.f,
                        ep.horizon - turn + 1,
                        ep.rng(turn),
                        DEFAULT_TOLERANCE,
                    )
                else:
                    action = select_action(
                        ep.policy, ep.s, ep.f, turn, ep.horizon, ep.rng(turn)
                    )
                ep.step(turn, action)
                ep.trace[-1].update(diagnostics)
            append_json(output, [ep.summary()])
            if (index + 1) % 10 == 0:
                print(
                    json.dumps(
                        {
                            "experiment": EXPERIMENT,
                            "baseline_episodes": index + 1,
                            "pending_at_start": len(episodes),
                        }
                    ),
                    flush=True,
                )
    elif episodes:
        for turn in range(1, DESIGN["horizon"] + 1):
            order = np.random.default_rng(
                stable_seed(SEED, EXPERIMENT, turn, "job_order")
            ).permutation(len(episodes))
            answers = await client.evaluate([episodes[i].job(turn) for i in order])
            if len(answers) != len(order):
                raise ValueError("Missing E9 answers; no partial turn will be scored")
            for index, answer in zip(order, answers, strict=True):
                ep = episodes[index]
                action = (
                    answer["action"]
                    if POLICIES[ep.policy][2] == "direct"
                    else int(ep.rng(turn).choice(ep.k, p=answer["probabilities"]))
                )
                ep.step(turn, action, answer)
                if turn == ep.horizon:
                    append_json(output, [ep.summary()])
            if turn % 5 == 0 or turn == DESIGN["horizon"]:
                print(
                    json.dumps(
                        {
                            "experiment": EXPERIMENT,
                            "turn": turn,
                            "episodes": len(episodes),
                            "accounting": client.accounting(),
                        }
                    ),
                    flush=True,
                )
    merge_episodes(run_dir)


async def run(args):
    if args.prepare:
        print(json.dumps(prepare(args.run_dir), indent=2))
        return
    verify_manifest(args.run_dir)
    if args.report:
        result = interaction_summary(
            read_jsonl(Path(args.run_dir) / "episodes_jev.jsonl")
        )
        output = Path(args.run_dir) / "interaction_summary.json"
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2))
        return
    if args.only_baselines:
        await execute(None, args.run_dir, only_baselines=True)
        return
    client = JevClient(
        args.run_dir,
        cap=args.cap,
        rate=args.rate,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
    )
    try:
        await execute(client, args.run_dir)
    finally:
        await client.close()


def interaction_summary(records):
    """Single planned interaction; whole paired tasks resampled within each K."""
    pairs = {k: {} for k in DESIGN["ks"]}
    seen = set()
    for row in records:
        if row.get("experiment") != EXPERIMENT or row.get("policy") not in POLICIES:
            continue
        k, identity, policy = row["k"], row["episode_id"], row["policy"]
        expected_ids = {f"{EXPERIMENT}:prior:{k}:{i}" for i in range(DESIGN["n"])}
        if (
            k not in pairs
            or identity not in expected_ids
            or row.get("family") != "prior"
            or row.get("horizon") != DESIGN["horizon"]
        ):
            raise ValueError("Unexpected task in E9 analysis")
        key = (identity, policy)
        if key in seen:
            raise ValueError("Duplicate episode-policy record in E9 analysis")
        seen.add(key)
        if not row.get("completed", False):
            continue
        regret = row["pseudo_regret"]
        if not np.isfinite(regret):
            raise ValueError("Nonfinite episode regret")
        pairs[k].setdefault(identity, {})[policy] = float(regret)
    effects, draws, cells = {}, {}, []
    for k in DESIGN["ks"]:
        values = np.array(
            [
                pair["bayes_sample"] - pair["bayes_direct"]
                for _, pair in sorted(pairs[k].items())
                if len(pair) == 2
            ]
        )
        effects[k] = None if not len(values) else float(values.mean())
        if len(values):
            rng = np.random.default_rng(
                stable_seed(SEED, EXPERIMENT, "interaction_bootstrap", k)
            )
            draws[k] = values[
                rng.integers(len(values), size=(10000, len(values)))
            ].mean(axis=1)
            low, high = np.quantile(draws[k], [0.025, 0.975]).tolist()
        else:
            low, high = None, None
        cells.append(
            {
                "k": k,
                "sample_minus_direct": effects[k],
                "ci_low": low,
                "ci_high": high,
                "paired_tasks": len(values),
                "planned_tasks": DESIGN["n"],
                "missing_pairs": DESIGN["n"] - len(values),
                "completed_direct": sum(
                    "bayes_direct" in pair for pair in pairs[k].values()
                ),
                "completed_sample": sum(
                    "bayes_sample" in pair for pair in pairs[k].values()
                ),
            }
        )
    if all(k in draws for k in (2, 10)):
        interaction = effects[10] - effects[2]
        low, high = np.quantile(draws[10] - draws[2], [0.025, 0.975]).tolist()
    else:
        interaction, low, high = None, None, None
    complete = all(cell["missing_pairs"] == 0 for cell in cells)
    return {
        "experiment": EXPERIMENT,
        "metric": "pseudo_regret",
        "contrast": PRIMARY_CONTRAST,
        "hypothesis": "negative",
        "estimate": interaction,
        "ci_low": low,
        "ci_high": high,
        "bootstrap_resamples": 10000,
        "interval": "two-sided 95% percentile",
        "complete_design": complete,
        "analysis_status": "planned confirmatory analysis"
        if complete
        else "incomplete; descriptive only",
        "secondary_per_k": cells,
        "interpretation": "K and resulting fixed-budget difficulty/input arity change together; no mechanistic isolation or K-threshold estimate",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument(
        "--report", action="store_true", help="Offline planned interaction analysis"
    )
    parser.add_argument(
        "--only-baselines", action="store_true", help="Offline only; requires --run"
    )
    parser.add_argument("--run-dir", default="artifacts/sampling_v1")
    parser.add_argument("--cap", type=float, default=18.0)
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.only_baselines and not args.run:
        parser.error("--only-baselines requires --run")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
