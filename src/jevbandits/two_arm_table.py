"""Compact exact two-arm Beta-Bernoulli DP and posthoc trajectory scoring.

The table covers all states reachable from two Beta(1,1) priors by a fixed
terminal horizon. It performs complete Bellman maximization, not an index or
rollout approximation. Earlier recursive-solver horizon guards do not apply.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from math import comb
from numbers import Integral
from pathlib import Path
from time import perf_counter

import numpy as np

MAX_HORIZON = 150
MAX_BYTES = 256 * 1024 * 1024
POLICY_NAME = "exact_h100"
AUDITED_EXPERIMENTS = ("e4_scaling", "e9_sampling_regimes")


class TwoArmTable:
    """All-state finite-horizon Bayesian values for independent uniform priors.

    layers[n][n1][s1,s2] stores optimal future reward after n observations,
    n1 on arm 1, with s1 and s2 successes. Remaining horizon is T-n, so it is
    unnecessary to store another time dimension. Total float64 entries are
    C(T+4,4), including the zero-valued terminal layer.
    """

    def __init__(self, horizon, max_bytes=MAX_BYTES):
        if not isinstance(horizon, Integral) or not 1 <= horizon <= MAX_HORIZON:
            raise ValueError(f"Require integer 1 <= horizon <= {MAX_HORIZON}")
        self.horizon = int(horizon)
        self.entries = comb(self.horizon + 4, 4)
        if self.entries * 8 > max_bytes:
            raise MemoryError(f"DP table requires {self.entries * 8} array bytes, exceeding guard")
        start = perf_counter()
        self.layers = [None] * (self.horizon + 1)
        self.layers[self.horizon] = [np.zeros((n1 + 1, self.horizon - n1 + 1))
                                    for n1 in range(self.horizon + 1)]
        for n in range(self.horizon - 1, -1, -1):
            layer = []
            following = self.layers[n + 1]
            for n1 in range(n + 1):
                n2 = n - n1
                mean1 = ((np.arange(n1 + 1) + 1) / (n1 + 2))[:, None]
                mean2 = ((np.arange(n2 + 1) + 1) / (n2 + 2))[None, :]
                arm1, arm2 = following[n1 + 1], following[n1]
                q1 = mean1 + mean1 * arm1[1:, :] + (1 - mean1) * arm1[:-1, :]
                q2 = mean2 + mean2 * arm2[:, 1:] + (1 - mean2) * arm2[:, :-1]
                layer.append(np.maximum(q1, q2))
            self.layers[n] = layer
        self.build_seconds = perf_counter() - start
        self.array_bytes = sum(matrix.nbytes for layer in self.layers for matrix in layer)

    def _state(self, successes, failures):
        if len(successes) != 2 or len(failures) != 2:
            raise ValueError("The table requires exactly two arms")
        if any(not isinstance(x, Integral) or x < 0 for x in (*successes, *failures)):
            raise ValueError("Successes and failures must be nonnegative integers")
        s1, s2 = map(int, successes)
        n1, n2 = s1 + int(failures[0]), s2 + int(failures[1])
        n = n1 + n2
        if n > self.horizon:
            raise ValueError("Observed counts exceed this table's terminal horizon")
        return n, n1, n2, s1, s2

    def value(self, successes=(0, 0), failures=(0, 0)) -> float:
        n, n1, _, s1, s2 = self._state(successes, failures)
        return float(self.layers[n][n1][s1, s2])

    def q(self, successes, failures) -> np.ndarray:
        """Two exact action values; remaining=T-sum(successes+failures)."""
        n, n1, n2, s1, s2 = self._state(successes, failures)
        if n == self.horizon:
            return np.zeros(2)
        mean1, mean2 = (s1 + 1) / (n1 + 2), (s2 + 1) / (n2 + 2)
        arm1, arm2 = self.layers[n + 1][n1 + 1], self.layers[n + 1][n1]
        return np.array([
            mean1 + mean1 * arm1[s1 + 1, s2] + (1 - mean1) * arm1[s1, s2],
            mean2 + mean2 * arm2[s1, s2 + 1] + (1 - mean2) * arm2[s1, s2],
        ])

    def select_action(self, successes, failures, rng):
        if sum(successes) + sum(failures) == self.horizon:
            raise ValueError("No decisions remain")
        values = self.q(successes, failures)
        ties = np.flatnonzero(np.isclose(values, values.max(), atol=1e-12, rtol=0))
        return int(rng.choice(ties))


def _digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_complete_records(path):
    """Read a live writer's complete records without repairing or mutating it."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open() as source:
        for line in source:
            if not line.endswith("\n"):
                break
            records.append(json.loads(line))
    return records


def _read_owned_records(path):
    """Discard an interrupted final write only in this audit's own outputs."""
    path = Path(path)
    if path.exists():
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            path.write_bytes(data[:data.rfind(b"\n") + 1])
    return _read_complete_records(path)


def score_episode(record, table):
    """Replay observed counts and score decisions; verify the paired world."""
    from .experiments import Episode

    if record["k"] != 2 or record["horizon"] != table.horizon:
        raise ValueError("Episode and two-arm table dimensions differ")
    if not record.get("completed") or len(record["trace"]) != table.horizon:
        raise ValueError("Only completed episodes may be scored")
    index = int(record["episode_id"].rsplit(":", 1)[1])
    replay = Episode(record["experiment"], record["family"], 2, index, table.horizon, record["policy"])
    if replay.seed != record["seed"] or not np.array_equal(replay.theta, record["theta"]):
        raise ValueError("Source episode does not match the frozen environment")
    scores = []
    for turn, row in enumerate(record["trace"], start=1):
        action, reward = row["action"], row["reward"]
        if row["turn"] != turn or action not in (0, 1) or reward not in (0, 1):
            raise ValueError("Invalid action/reward/turn in source trajectory")
        expected_reward = replay.outcomes[action, replay.s[action] + replay.f[action]]
        if reward != expected_reward:
            raise ValueError("Recorded reward differs from paired arm-pull outcome stream")
        q = table.q(replay.s, replay.f)
        loss = float(q.max() - q[action])
        scores.append({"turn": turn, "action": action, "reward": reward,
                       "q_values": q.tolist(), "exact_loss": loss,
                       "optimal_agreement": bool(loss <= 1e-10)})
        replay.s[action] += reward
        replay.f[action] += 1 - reward
    return {
        "experiment": record["experiment"], "family": record["family"], "k": 2,
        "horizon": table.horizon, "episode_id": record["episode_id"], "policy": record["policy"],
        "source_episode_sha256": _digest(record), "posthoc_normative_audit": True,
        "exact_loss": sum(row["exact_loss"] for row in scores),
        "optimal_agreement": float(np.mean([row["optimal_agreement"] for row in scores])),
        "reward": record["reward"], "pseudo_regret": record["pseudo_regret"],
        "trace": scores,
    }


def _run_manifest(run_dir):
    path = run_dir / "manifest.json"
    original = json.loads(path.read_text())
    for name in ["experiments.py", "prompts.py", "baselines.py"]:
        frozen = original["source_sha256"].get(name, original["source_sha256"].get(f"src/jevbandits/{name}"))
        actual = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        if frozen != actual:
            raise ValueError(f"Frozen source changed: {name}")
    audit = {
        "status": "posthoc exact normative scoring and exploratory added baseline; no API",
        "policy": POLICY_NAME, "arithmetic": "float64 full-state Bellman table",
        "original_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prior": "independent Beta(1,1)", "tie_tolerance": 1e-12,
        "agreement_tolerance": 1e-10,
    }
    output = run_dir / "two_arm_audit_manifest.json"
    if output.exists() and json.loads(output.read_text()) != audit:
        raise ValueError("Normative audit provenance changed; choose a new output directory")
    output.write_text(json.dumps(audit, indent=2) + "\n")
    return original


def add_baseline(run_dir, original, tables):
    from .experiments import Episode, append_json

    output = run_dir / "episodes_long_exact.jsonl"
    completed = {(row["episode_id"], row["policy"]) for row in _read_owned_records(output)}
    count = 0
    for experiment in AUDITED_EXPERIMENTS:
        if experiment not in original["online"]:
            continue
        design = original["online"][experiment]
        if 2 not in design["ks"]:
            continue
        horizon = design["horizon"]
        if horizon != 100:
            raise ValueError("The exact_h100 comparator is restricted to horizon 100")
        if horizon not in tables:
            tables[horizon] = TwoArmTable(horizon)
        table = tables[horizon]
        for family in design["families"]:
            for index in range(design["n"]):
                episode = Episode(experiment, family, 2, index, horizon, POLICY_NAME)
                if (episode.episode_id, POLICY_NAME) in completed:
                    continue
                for turn in range(1, horizon + 1):
                    q = table.q(episode.s, episode.f)
                    action = table.select_action(episode.s, episode.f, episode.rng(turn))
                    episode.step(turn, action)
                    episode.trace[-1].update(exact_loss=float(q.max() - q[action]), optimal_agreement=True)
                record = episode.summary()
                record["posthoc_exploratory"] = True
                record["exact_method"] = "compact two-arm full-state Bellman DP"
                append_json(output, [record])
                count += 1
    return count


def score_available(run_dir, tables):
    from .experiments import append_json

    sources = ["episodes_jev.jsonl", "episodes_baselines.jsonl", "episodes_index.jsonl", "episodes_long_exact.jsonl"]
    episodes = {}
    for source in sources:
        for row in _read_complete_records(run_dir / source):
            if row["experiment"] in AUDITED_EXPERIMENTS and row["k"] == 2:
                key = (row["episode_id"], row["policy"])
                if key in episodes and episodes[key] != row:
                    raise ValueError(f"Conflicting source copies of episode {key}")
                episodes[key] = row
    output = run_dir / "two_arm_bellman_audit.jsonl"
    existing = {(row["episode_id"], row["policy"]): row for row in _read_owned_records(output)}
    count = 0
    for key, record in episodes.items():
        if key in existing:
            if existing[key]["source_episode_sha256"] != _digest(record):
                raise ValueError(f"Scored source episode changed: {key}")
            continue
        horizon = record["horizon"]
        if horizon not in tables:
            tables[horizon] = TwoArmTable(horizon)
        score = score_episode(record, tables[horizon])
        append_json(output, [score])
        existing[key] = score
        count += 1
    groups = defaultdict(list)
    for row in existing.values():
        groups[(row["experiment"], row["family"], row["policy"])].append(row)
    summary = []
    for (experiment, family, policy), rows in sorted(groups.items()):
        summary.append({
            "experiment": experiment, "family": family, "policy": policy, "n": len(rows),
            "mean_exact_loss": float(np.mean([r["exact_loss"] for r in rows])),
            "mean_optimal_agreement": float(np.mean([r["optimal_agreement"] for r in rows])),
            "mean_pseudo_regret": float(np.mean([r["pseudo_regret"] for r in rows])),
        })
    (run_dir / "two_arm_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", default=["artifacts/overnight_v2", "artifacts/sampling_v1"])
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--horizon", type=int, default=100)
    args = parser.parse_args()
    if args.benchmark:
        table = TwoArmTable(args.horizon)
        print(json.dumps({"horizon": table.horizon, "prior_optimal_reward": table.value(),
                          "prior_bayesian_regret": 2 * table.horizon / 3 - table.value(),
                          "array_bytes": table.array_bytes, "entries": table.entries,
                          "build_seconds": table.build_seconds}))
        return
    if not args.score and not args.baseline:
        parser.error("Choose --score and/or --baseline, or --benchmark")
    tables = {}
    for path in args.run_dirs:
        run_dir = Path(path)
        original = _run_manifest(run_dir)
        new_baselines = add_baseline(run_dir, original, tables) if args.baseline else 0
        scored = score_available(run_dir, tables) if args.score else 0
        print(json.dumps({"run_dir": str(run_dir), "new_exact_baselines": new_baselines,
                          "newly_scored_episodes": scored}), flush=True)


if __name__ == "__main__":
    main()
