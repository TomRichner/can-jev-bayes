"""Frozen designs, common potential outcomes, and resumable scientific records."""

import hashlib
import json
import os
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.stats import beta

from .baselines import exact_q, select_action
from .prompts import observation, question, stable_seed

SEED = 20260920
CORE = {
    "counts_direct": ("counts", "explore", "direct"),
    "bayes_direct": ("bayes", "explore", "direct"),
    "counts_sample": ("counts", "explore", "sample"),
    "bayes_sample": ("bayes", "explore", "sample"),
}
POLICIES = {
    **CORE,
    "dp_direct": ("dp_values", "explore", "direct"),
    "neutral_direct": ("bayes", "neutral", "direct"),
    "regret_direct": ("bayes", "regret", "direct"),
    "thompson_prompt": ("bayes", "thompson", "direct"),
    "ucb_direct": ("ucb", "ucb", "direct"),
}
CLASSICAL = ["random", "greedy", "ts", "bayes_ucb", "ucb1", "knowledge_gradient", "ids"]
DESIGNS = {
    "pilot": {
        "ks": [2, 5, 15],
        "families": ["clear", "close", "prior"],
        "n": 3,
        "horizon": 20,
        "policies": list(CORE),
    },
    "e3_exact_online": {
        "ks": [2],
        "families": ["prior"],
        "n": 200,
        "horizon": 20,
        "policies": [*CORE, "dp_direct"],
    },
    "e4_scaling": {
        "ks": [2, 3, 5, 10, 15],
        "families": ["clear", "close", "prior"],
        "n": 40,
        "horizon": 100,
        "policies": list(CORE),
    },
    "e5_strategy": {
        "ks": [3, 10],
        "families": ["close", "prior"],
        "n": 60,
        "horizon": 100,
        "policies": [
            "bayes_direct",
            "neutral_direct",
            "regret_direct",
            "thompson_prompt",
            "ucb_direct",
        ],
    },
}


def append_json(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        for record in records:
            file.write(
                json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
            )
        file.flush()
        os.fsync(file.fileno())


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    result = []
    for i, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            # A torn final record cannot be regarded as complete.
            path.write_text("\n".join(lines[:i]) + "\n")
    return result


def environment(experiment, family, k, episode, horizon):
    seed = stable_seed(SEED, experiment, family, k, episode)
    rng = np.random.default_rng(seed)
    if family == "prior":
        theta = rng.beta(1, 1, k)
    elif family == "clear":
        theta = np.array([0.7] + [0.3] * (k - 1))
        rng.shuffle(theta)
    elif family == "close":
        theta = np.array([0.55] + [0.5] * (k - 1))
        rng.shuffle(theta)
    else:
        raise ValueError(f"Unknown family {family}")
    outcomes = (rng.random((k, horizon)) < theta[:, None]).astype(int)
    return seed, theta, outcomes


def fixtures(experiment, n=100):
    known = [
        ([0, 0], [0, 0]),
        ([10, 0], [8, 0]),
        ([18, 0], [2, 0]),
        ([9, 0], [9, 0]),
        ([1, 59], [0, 39]),
        ([0, 5], [1, 3]),
    ]
    rng = np.random.default_rng(stable_seed(SEED, experiment, "fixtures"))
    pairs = known.copy() if experiment == "e1_horizon" else []
    while len(pairs) < n:
        totals = rng.choice([0, 2, 5, 10, 20], size=2)
        successes = rng.integers(0, totals + 1)
        pairs.append((successes.tolist(), (totals - successes).tolist()))
    return pairs[:n]


async def diagnostics(client, experiment, n=100):
    output = client.run_dir / "diagnostics.jsonl"
    done = {r["decision_id"] for r in read_jsonl(output)}
    jobs, contexts = [], []
    for index, (s, f) in enumerate(fixtures(experiment, n)):
        horizons = [1, 2, 5, 10] if experiment == "e1_horizon" else [1, 2, 10]
        representations = (
            ["counts"]
            if experiment == "e1_horizon"
            else ["counts", "means", "bayes", "dp_values", "recommendation"]
        )
        framings = (
            ["neutral", "explore", "regret"]
            if experiment == "e1_horizon"
            else ["explore"]
        )
        for h in horizons:
            q = exact_q(tuple(s), tuple(f), h)
            means = (np.array(s) + 1) / (np.array(s) + np.array(f) + 2)
            for representation in representations:
                for framing in framings:
                    actual_framing = (
                        "recommendation"
                        if representation == "recommendation"
                        else framing
                    )
                    for reversal in [0, 1]:
                        order = [0, 1] if reversal == 0 else [1, 0]
                        obs = observation(
                            s, f, h, representation, q_values=q, label_order=order
                        )
                        for repeat in range(2):
                            decision_id = f"{experiment}:{index}:{h}:{representation}:{framing}:{reversal}:{repeat}"
                            if decision_id in done:
                                continue
                            jobs.append(
                                {
                                    "id": decision_id,
                                    "question": question(obs, actual_framing),
                                }
                            )
                            contexts.append(
                                {
                                    "experiment": experiment,
                                    "fixture_id": f"{experiment}:{index}",
                                    "label_order": reversal,
                                    "horizon": h,
                                    "representation": representation,
                                    "framing": framing,
                                    "repeat": repeat,
                                    "policy": f"{representation}_{framing}",
                                    "decision_id": decision_id,
                                    "successes": s,
                                    "failures": f,
                                    "q_values": q.tolist(),
                                    "means": means.tolist(),
                                    "order": order,
                                }
                            )
    # Randomize query time/order without changing fixture identity or evaluation pairing.
    permutation = np.random.default_rng(
        stable_seed(SEED, experiment, "execution")
    ).permutation(len(jobs))
    for start in range(0, len(permutation), 256):
        indices = permutation[start : start + 256]
        answers = await client.evaluate([jobs[i] for i in indices])
        records = []
        for idx, answer in zip(indices, answers):
            context = contexts[idx].copy()
            order = context.pop("order")
            q, means = np.array(context["q_values"]), np.array(context.pop("means"))
            action = order[answer["action"]]
            p = np.zeros(2)
            p[order] = answer["probabilities"]
            context.update(
                action=answer["action"],
                canonical_action=action,
                probabilities=p.tolist(),
                exact_loss=float(q.max() - q[action]),
                optimal_agreement=bool(q.max() - q[action] <= 1e-10),
                distribution_exact_loss=float(q.max() - p @ q),
                posterior_mean_greedy=bool(means.max() - means[action] <= 1e-10),
                prediction_entropy=float(-sum(x * np.log(x) for x in p if x > 0)),
                request_id=answer["request_id"],
                raw_answer=answer["raw_answer"],
                probability_mass=answer["probability_mass"],
            )
            records.append(context)
        append_json(output, records)
        print(
            json.dumps(
                {
                    "experiment": experiment,
                    "diagnostics_completed": min(start + 256, len(jobs)),
                    "remaining": max(0, len(jobs) - start - 256),
                    "cost": client.accounting()["known_cost_usd"],
                }
            ),
            flush=True,
        )


class Episode:
    def __init__(self, experiment, family, k, index, horizon, policy):
        self.experiment, self.family, self.k = experiment, family, k
        self.index, self.horizon, self.policy = index, horizon, policy
        self.episode_id = f"{experiment}:{family}:{k}:{index}"
        self.seed, self.theta, self.outcomes = environment(
            experiment, family, k, index, horizon
        )
        self.s, self.f = np.zeros(k, dtype=int), np.zeros(k, dtype=int)
        self.trace = []

    def rng(self, turn):
        return np.random.default_rng(
            stable_seed(self.seed, self.policy, turn, "actions")
        )

    def decision_id(self, turn):
        return f"{self.episode_id}:{self.policy}:{turn}"

    def job(self, turn):
        representation, framing, _ = POLICIES[self.policy]
        remaining = self.horizon - turn + 1
        q = (
            exact_q(tuple(self.s), tuple(self.f), remaining)
            if representation == "dp_values"
            else None
        )
        indices = (
            beta.ppf(1 - 1 / turn, self.s + 1, self.f + 1)
            if representation == "ucb"
            else None
        )
        obs = observation(
            self.s, self.f, remaining, representation, q_values=q, indices=indices
        )
        return {"id": self.decision_id(turn), "question": question(obs, framing)}

    def step(self, turn, action, answer=None):
        count = self.s + self.f
        means = (self.s + 1) / (count + 2)
        q = (
            exact_q(tuple(self.s), tuple(self.f), self.horizon - turn + 1)
            if self.experiment == "e3_exact_online"
            else None
        )
        reward = int(self.outcomes[action, count[action]])
        row = {
            "turn": turn,
            "action": int(action),
            "reward": reward,
            "pseudo_regret": float(self.theta.max() - self.theta[action]),
            "posterior_mean_greedy": bool(means.max() - means[action] <= 1e-10),
            "unseen": bool(count[action] == 0),
            "best_arm": bool(self.theta.max() - self.theta[action] <= 1e-12),
            "exact_loss": None if q is None else float(q.max() - q[action]),
            "optimal_agreement": None
            if q is None
            else bool(q.max() - q[action] <= 1e-10),
        }
        if answer is not None:
            row.update(
                request_id=answer["request_id"],
                probability_mass=answer["probability_mass"],
                confidence=answer["raw_answer"]["confidence"],
            )
        if self.policy == "ucb_direct":
            values = beta.ppf(1 - 1 / turn, self.s + 1, self.f + 1)
            row["advice_adherence"] = bool(values.max() - values[action] <= 1e-7)
        self.s[action] += reward
        self.f[action] += 1 - reward
        self.trace.append(row)

    def summary(self):
        assert len(self.trace) == self.horizon

        def mean(name):
            values = [x[name] for x in self.trace if x.get(name) is not None]
            return float(np.mean(values)) if values else None

        actions = [r["action"] for r in self.trace]
        representation, framing, rule = POLICIES.get(
            self.policy, ("analytic", "algorithm", "algorithm")
        )
        return {
            "experiment": self.experiment,
            "family": self.family,
            "k": self.k,
            "horizon": self.horizon,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "policy": self.policy,
            "representation": representation,
            "framing": framing,
            "action_rule": rule,
            "reward": sum(r["reward"] for r in self.trace),
            "pseudo_regret": sum(r["pseudo_regret"] for r in self.trace),
            "exact_loss": None
            if self.trace[0]["exact_loss"] is None
            else sum(r["exact_loss"] for r in self.trace),
            "optimal_agreement": mean("optimal_agreement"),
            "posterior_mean_greedy": mean("posterior_mean_greedy"),
            "unseen_fraction": mean("unseen"),
            "switches": sum(a != b for a, b in pairwise(actions)),
            "arms_visited": len(set(actions)),
            "best_arm_fraction": mean("best_arm"),
            "suffix_failure": not any(r["best_arm"] for r in self.trace[-20:]),
            "advice_adherence": mean("advice_adherence"),
            "completed": True,
            "theta": self.theta.tolist(),
            "trace": self.trace,
        }


async def online(client, run_dir, experiment, *, only="jev", n_override=None):
    design = DESIGNS[experiment]
    n = design["n"] if n_override is None else n_override
    policies = (
        design["policies"]
        if only == "jev"
        else CLASSICAL + (["exact"] if experiment == "e3_exact_online" else [])
    )
    output = Path(run_dir) / f"episodes_{only}.jsonl"
    done = {(r["episode_id"], r["policy"]) for r in read_jsonl(output)}
    episodes = [
        Episode(experiment, family, k, i, design["horizon"], policy)
        for family in design["families"]
        for k in design["ks"]
        for i in range(n)
        for policy in policies
        if (f"{experiment}:{family}:{k}:{i}", policy) not in done
    ]
    if only != "jev":
        for index, ep in enumerate(episodes):
            for turn in range(1, ep.horizon + 1):
                action = select_action(
                    ep.policy, ep.s, ep.f, turn, ep.horizon, ep.rng(turn)
                )
                ep.step(turn, action)
            append_json(output, [ep.summary()])
            if index % 100 == 0:
                print(
                    json.dumps(
                        {
                            "experiment": experiment,
                            "baseline_episodes": index + 1,
                            "total": len(episodes),
                        }
                    ),
                    flush=True,
                )
        return
    for turn in range(1, design["horizon"] + 1):
        # Interleave conditions with a deterministic turn-specific ordering to reduce time confounding.
        order = np.random.default_rng(
            stable_seed(SEED, experiment, turn, "job_order")
        ).permutation(len(episodes))
        answers = await client.evaluate([episodes[i].job(turn) for i in order])
        for index, answer in zip(order, answers):
            ep = episodes[index]
            _, _, rule = POLICIES[ep.policy]
            action = (
                answer["action"]
                if rule == "direct"
                else int(ep.rng(turn).choice(ep.k, p=answer["probabilities"]))
            )
            ep.step(turn, action, answer)
        if turn % 5 == 0 or turn == design["horizon"]:
            print(
                json.dumps(
                    {
                        "experiment": experiment,
                        "turn": turn,
                        "horizon": design["horizon"],
                        "episodes": len(episodes),
                        "known_cost": client.accounting()["known_cost_usd"],
                    }
                ),
                flush=True,
            )
    append_json(output, [ep.summary() for ep in episodes])


def merge_episodes(run_dir):
    path = Path(run_dir)
    records = read_jsonl(path / "episodes_jev.jsonl") + read_jsonl(
        path / "episodes_baselines.jsonl"
    )
    unique = {(r["episode_id"], r["policy"]): r for r in records}
    temp = path / "episodes.tmp"
    temp.write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in unique.values())
    )
    temp.replace(path / "episodes.jsonl")
    return len(unique)


async def preflight(client):
    # Identical target questions evaluated alone and in mixed batches. No other question's evidence is in shared state.
    cases = [
        ([10, 0], [8, 0], 1),
        ([10, 0], [8, 0], 2),
        ([18, 0], [2, 0], 10),
        ([1, 59], [0, 39], 10),
    ]
    records = []
    for repeat in range(10):
        jobs = [
            {
                "id": f"preflight:alone:{repeat}:{i}",
                "question": question(observation(s, f, h, "bayes")),
            }
            for i, (s, f, h) in enumerate(cases)
        ]
        for i, job in enumerate(jobs):
            answer = (await client.evaluate([job]))[0]
            records.append(
                {"condition": "alone", "repeat": repeat, "fixture": i, **answer}
            )
        batch = [
            {"id": f"preflight:mixed:{repeat}:{i}", "question": job["question"]}
            for i, job in enumerate(jobs)
        ]
        if repeat % 2:
            batch.reverse()
        answers = await client.evaluate(batch)
        for job, answer in zip(batch, answers):
            records.append(
                {
                    "condition": "mixed",
                    "repeat": repeat,
                    "fixture": int(job["id"].split(":")[-1]),
                    **answer,
                }
            )
    path = client.run_dir / "preflight.json"
    path.write_text(json.dumps(records, indent=2))
    for i in range(len(cases)):
        alone = np.array(
            [
                r["probabilities"]
                for r in records
                if r["fixture"] == i and r["condition"] == "alone"
            ]
        )
        mixed = np.array(
            [
                r["probabilities"]
                for r in records
                if r["fixture"] == i and r["condition"] == "mixed"
            ]
        )
        shift = float(np.max(np.abs(alone.mean(0) - mixed.mean(0))))
        print(
            json.dumps(
                {
                    "preflight_fixture": i,
                    "alone_mean": alone.mean(0).tolist(),
                    "batch_mean": mixed.mean(0).tolist(),
                    "max_shift": shift,
                }
            ),
            flush=True,
        )
        if shift > 0.10:
            raise RuntimeError(
                "Batching preflight probability shift exceeded .10; investigate before running"
            )


def manifest(run_dir):
    import subprocess

    from .prompts import OBJECTIVES, TASK

    content = {
        "master_seed": SEED,
        "online": DESIGNS,
        "e1_horizon": {
            "fixtures": 100,
            "horizons": [1, 2, 5, 10],
            "framings": ["neutral", "explore", "regret"],
            "repeats": 2,
            "label_orders": 2,
        },
        "e2_assistance": {
            "fixtures": 100,
            "horizons": [1, 2, 10],
            "representations": [
                "counts",
                "means",
                "bayes",
                "dp_values",
                "recommendation",
            ],
            "repeats": 2,
            "label_orders": 2,
        },
        "policies": POLICIES,
        "classical": CLASSICAL,
        "objectives": OBJECTIVES,
        "shared_state": TASK,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "uv_lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("src/jevbandits").glob("*.py")
        },
    }
    path = Path(run_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = json.loads(path.read_text())
        # Analysis changes are permitted; scientific methods require explicit new run ID.
        for field in [
            "master_seed",
            "online",
            "e1_horizon",
            "e2_assistance",
            "policies",
            "classical",
            "objectives",
            "shared_state",
        ]:
            if old[field] != json.loads(json.dumps(content[field])):
                raise ValueError(f"Manifest {field} changed; use a new run directory")
        for name in ["experiments.py", "prompts.py", "baselines.py", "client.py"]:
            if old["source_sha256"].get(name) != content["source_sha256"].get(name):
                raise ValueError(
                    f"Scientific source {name} changed; use a new run directory and document amendment"
                )
    else:
        path.write_text(json.dumps(content, indent=2))
