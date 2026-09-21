"""Post hoc public-history behavioral audit; offline and episode-clustered."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from .baselines import exact_q, knowledge_gradient
from .followup_reports import NICE
from .report import _seed, bootstrap_mean, markdown_table, read_jsonl

TOLERANCE = 1e-10
PHASES = ("early", "middle", "late")
LABELS = NICE | {
    "counts_sample": "Jev: counts, sampled",
    "bayes_sample": "Jev: Bayesian summaries, sampled",
    "dp_direct": "Jev: exact action values",
    "exact": "Bayes-optimal dynamic program",
    "neutral_direct": "Jev: neutral reward objective",
    "regret_direct": "Jev: regret objective",
    "thompson_prompt": "Jev: Thompson sampling prompt",
    "ucb_direct": "Jev: Bayes-UCB advice",
}
METRICS = (
    "posterior_mean_greedy",
    "strict_non_greedy",
    "unseen_arm",
    "lower_mean_higher_sd",
    "equal_mean_higher_sd",
    "unequal_sd_greedy_tie_available",
    "posterior_mean_deficit",
    "chosen_posterior_sd",
    "bayes_ucb_agreement",
    "knowledge_gradient_agreement",
    "bayes_ucb_top_set_size",
    "knowledge_gradient_top_set_size",
    "exact_loss",
    "strict_exploration_required",
    "useful_non_greedy",
    "inferior_non_greedy",
    "neutral_non_greedy",
    "exact_optimal_agreement",
)


def phase_for_turn(turn: int, horizon: int) -> str:
    if turn < 1 or turn > horizon:
        raise ValueError("Turn must be within the episode")
    return (
        "early"
        if 5 * turn <= horizon
        else "middle"
        if 5 * turn <= 4 * horizon
        else "late"
    )


def state_behavior(successes, failures, action: int, turn: int, horizon: int, q=None):
    """Evaluate the action on its own public history, with no latent arm means."""
    s, f = np.asarray(successes, dtype=float), np.asarray(failures, dtype=float)
    if (
        s.ndim != 1
        or len(s) < 2
        or f.shape != s.shape
        or not np.isfinite(s).all()
        or not np.isfinite(f).all()
        or np.any(s < 0)
        or np.any(f < 0)
        or np.any(s != np.floor(s))
        or np.any(f != np.floor(f))
    ):
        raise ValueError("Invalid public counts")
    if (
        isinstance(action, (bool, np.bool_))
        or not isinstance(action, (int, np.integer))
        or not 0 <= action < len(s)
    ):
        raise ValueError("Invalid action")
    remaining = horizon - turn + 1
    if remaining < 1 or turn < 1:
        raise ValueError("Invalid remaining horizon")
    alpha, b = s + 1, f + 1
    total = alpha + b
    means = alpha / total
    sd = np.sqrt(alpha * b / (total**2 * (total + 1)))
    greedy = np.isclose(means, means.max(), atol=TOLERANCE, rtol=0)
    deficit = float(means.max() - means[action])
    non_greedy = deficit > TOLERANCE
    ucb = beta.ppf(1 - 1 / turn, alpha, b)
    kg = means + (remaining - 1) * knowledge_gradient(s.astype(int), f.astype(int))
    ucb_top = np.isclose(ucb, ucb.max(), rtol=0, atol=1e-12)
    kg_top = np.isclose(kg, kg.max(), rtol=0, atol=1e-12)
    row = {
        "posterior_mean_greedy": float(not non_greedy),
        "strict_non_greedy": float(non_greedy),
        "unseen_arm": float(s[action] + f[action] == 0),
        "lower_mean_higher_sd": float(
            non_greedy and sd[action] > sd[greedy].max() + TOLERANCE
        ),
        "equal_mean_higher_sd": float(
            greedy[action] and sd[action] > sd[greedy].min() + TOLERANCE
        ),
        "unequal_sd_greedy_tie_available": float(
            sd[greedy].max() > sd[greedy].min() + TOLERANCE
        ),
        "posterior_mean_deficit": deficit,
        "chosen_posterior_sd": float(sd[action]),
        "bayes_ucb_agreement": float(ucb_top[action]),
        "knowledge_gradient_agreement": float(kg_top[action]),
        "bayes_ucb_top_set_size": int(ucb_top.sum()),
        "knowledge_gradient_top_set_size": int(kg_top.sum()),
    }
    if q is not None:
        values = np.asarray(q, dtype=float)
        if values.shape != s.shape or not np.isfinite(values).all():
            raise ValueError("Invalid exact action values")
        best_greedy = float(values[greedy].max())
        chosen_advantage = float(values[action] - best_greedy)
        row.update(
            exact_loss=float(max(0, values.max() - values[action])),
            strict_exploration_required=float(values.max() > best_greedy + TOLERANCE),
            useful_non_greedy=float(non_greedy and chosen_advantage > TOLERANCE),
            inferior_non_greedy=float(non_greedy and chosen_advantage < -TOLERANCE),
            neutral_non_greedy=float(non_greedy and abs(chosen_advantage) <= TOLERANCE),
            exact_optimal_agreement=float(values.max() - values[action] <= TOLERANCE),
        )
    return row


def audit_episode(episode: dict) -> pd.DataFrame:
    """Replay only observed actions/rewards and validate stored E3 decision loss."""
    k, horizon = episode["k"], episode["horizon"]
    if not isinstance(k, int) or k < 2 or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Invalid episode dimensions")
    trace = episode["trace"]
    if len(trace) != horizon:
        raise ValueError("Incomplete or overlong episode trace")
    s, f = np.zeros(k, dtype=int), np.zeros(k, dtype=int)
    scored = []
    for turn, point in enumerate(trace, 1):
        if point.get("turn") != turn or isinstance(point.get("turn"), bool):
            raise ValueError("Nonconsecutive trace turns")
        action, reward = point["action"], point["reward"]
        if (
            isinstance(action, bool)
            or not isinstance(action, int)
            or not 0 <= action < k
        ):
            raise ValueError("Invalid trace action")
        if (
            isinstance(reward, bool)
            or not isinstance(reward, int)
            or reward not in (0, 1)
        ):
            raise ValueError("Invalid Bernoulli reward")
        if int((s + f).sum()) != turn - 1:
            raise ValueError("Count reconstruction failed")
        q = (
            exact_q(tuple(s), tuple(f), horizon - turn + 1)
            if episode["experiment"] == "e3_exact_online"
            else None
        )
        row = state_behavior(s, f, action, turn, horizon, q)
        if "posterior_mean_greedy" in point and bool(
            point["posterior_mean_greedy"]
        ) != bool(row["posterior_mean_greedy"]):
            raise ValueError(
                "Recorded posterior-mean-greedy flag disagrees with replay"
            )
        if "unseen" in point and bool(point["unseen"]) != bool(row["unseen_arm"]):
            raise ValueError("Recorded unseen-arm flag disagrees with replay")
        if q is not None:
            recorded_loss = point.get("exact_loss")
            if (
                recorded_loss is None
                or not np.isfinite(recorded_loss)
                or not np.isclose(recorded_loss, row["exact_loss"], atol=1e-9, rtol=0)
            ):
                raise ValueError(
                    "Recorded E3 exact loss disagrees with public-history recomputation"
                )
            if "optimal_agreement" in point and bool(
                point["optimal_agreement"]
            ) != bool(row["exact_optimal_agreement"]):
                raise ValueError("Recorded E3 optimality flag disagrees with replay")
        row.update(phase=phase_for_turn(turn, horizon), turn=turn)
        scored.append(row)
        s[action] += reward
        f[action] += 1 - reward
    if "reward" in episode and episode["reward"] != int(s.sum()):
        raise ValueError("Episode reward total disagrees with reconstructed counts")
    frame = pd.DataFrame(scored)
    if (
        "exact_loss" in frame
        and episode.get("exact_loss") is not None
        and not np.isclose(
            episode["exact_loss"], frame.exact_loss.sum(), rtol=0, atol=1e-8
        )
    ):
        raise ValueError("Episode exact-loss total disagrees with replay")
    metadata = {
        key: episode[key]
        for key in ["experiment", "family", "k", "horizon", "policy", "episode_id"]
    }
    phase_rows = []
    for phase in PHASES:
        block = frame.loc[frame.phase == phase]
        if block.empty:
            continue
        values = {
            metric: float(block[metric].mean()) for metric in METRICS if metric in block
        }
        phase_rows.append({**metadata, "phase": phase, "n_turns": len(block), **values})
    return pd.DataFrame(phase_rows)


def summarize_phases(phases: pd.DataFrame, *, samples=10_000) -> pd.DataFrame:
    if phases.empty:
        return pd.DataFrame()
    keys = ["experiment", "family", "k", "horizon", "policy", "phase"]
    if phases.duplicated([*keys, "episode_id"]).any():
        raise ValueError("Duplicate episode-phase rows")
    rows = []
    for key, block in phases.groupby(keys, sort=True):
        for metric in METRICS:
            if metric not in block or block[metric].isna().all():
                continue
            values = block[metric].dropna()
            mean, low, high = bootstrap_mean(
                values, samples=samples, seed=_seed(f"behavior:{key}:{metric}")
            )
            rows.append(
                dict(zip(keys, key))
                | {
                    "metric": metric,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n_episodes": len(values),
                    "n_recorded_turns": int(block.n_turns.sum()),
                    "analysis_role": "post_hoc_behavioral",
                }
            )
    return pd.DataFrame(rows)


def _plots(summary, output):
    if summary.empty:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    for (experiment, family, k), block in summary.groupby(
        ["experiment", "family", "k"], sort=True
    ):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True)
        policies = sorted(block.policy.unique())
        for ax, metric, title in zip(
            axes,
            ("posterior_mean_greedy", "lower_mean_higher_sd", "equal_mean_higher_sd"),
            (
                "Posterior-mean maximizing",
                "Lower mean, higher uncertainty",
                "Equal best mean, higher uncertainty",
            ),
            strict=True,
        ):
            for idx, policy in enumerate(policies):
                rows = (
                    block.loc[(block.policy == policy) & (block.metric == metric)]
                    .set_index("phase")
                    .reindex(PHASES)
                )
                style = (
                    "-."
                    if policy.endswith("sample")
                    else "-"
                    if policy.endswith("direct")
                    else "--"
                )
                ax.plot(
                    range(3),
                    rows["mean"],
                    marker="o",
                    markersize=3,
                    color=plt.get_cmap("tab20")(idx % 20),
                    linestyle=style,
                    label=LABELS.get(policy, policy.replace("_", " ")),
                )
            maximum = float(block.loc[block.metric == metric, "mean"].max())
            upper = (
                1.02
                if metric == "posterior_mean_greedy"
                else min(1.02, max(0.05, maximum * 1.2 + 0.005))
            )
            ax.set(
                title=title, ylim=(-upper * 0.02, upper), ylabel="Episode-mean fraction"
            )
            ax.set_xticks(range(3), ["Early", "Middle", "Late"])
            ax.grid(alpha=0.2)
        axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
        fig.suptitle(f"{experiment} · {family} · {k} arms · descriptive phase means")
        fig.tight_layout()
        filename = f"{experiment}_{family}_k{k}_behavior.png"
        fig.savefig(output / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
        files.append(filename)
    return files


def generate_behavior_report(
    run_dir, output_dir=None, *, experiments=None, samples=10_000
):
    source = Path(run_dir)
    output = Path(output_dir) if output_dir else source / "behavior_report"
    output.mkdir(parents=True, exist_ok=True)
    inputs = [source / "episodes.jsonl", source / "episodes_index.jsonl"]
    frames = [read_jsonl(path) for path in inputs]
    data = (
        pd.concat([x for x in frames if not x.empty], ignore_index=True)
        if any(not x.empty for x in frames)
        else pd.DataFrame()
    )
    if not data.empty and data.duplicated(["episode_id", "policy"]).any():
        raise ValueError("Duplicate input episode/policy records")
    if experiments is not None and not data.empty:
        data = data.loc[data.experiment.isin(experiments)]
    incomplete = 0
    if not data.empty:
        incomplete = int((~data.completed.eq(True)).sum())
        data = data.loc[data.completed.eq(True)]
    episode_phases = []
    for row in data.to_dict("records"):
        episode_phases.append(audit_episode(row))
    phases = (
        pd.concat(episode_phases, ignore_index=True)
        if episode_phases
        else pd.DataFrame()
    )
    summary = summarize_phases(phases, samples=samples)
    phases.to_csv(output / "episode_phase_metrics.csv", index=False)
    summary.to_csv(output / "phase_summary.csv", index=False)
    figures = _plots(summary, output)
    exact = (
        summary.loc[
            summary.metric.isin(
                [
                    "strict_exploration_required",
                    "useful_non_greedy",
                    "inferior_non_greedy",
                    "neutral_non_greedy",
                    "exact_loss",
                ]
            )
        ]
        if not summary.empty
        else summary
    )
    sections = [
        "# Post hoc exploration-strategy audit",
        f"Completed episode-policy records: {len(data):,}; incomplete records excluded: {incomplete}. Public Beta(1,1) counts were reconstructed from each policy's own observed actions and rewards. No hidden success probabilities or other policies' histories enter behavioral scoring.",
        "Early is t/T <= 0.2, middle is 0.2 < t/T <= 0.8, and late is t/T > 0.8. Each metric is first averaged within episode and phase. The reported 95% intervals bootstrap whole episode-phase averages; individual turns are not replicates. Plots show descriptive means; intervals are in `phase_summary.csv`.",
        "Posterior-mean maximizers form a tie-aware set (absolute tolerance 1e-10). Strict non-greedy choices have a larger mean deficit. Lower-mean/higher-uncertainty requires the chosen posterior standard deviation to exceed the maximum among posterior-mean maximizers. Equal-best-mean/higher-uncertainty requires a greedy-set choice with greater standard deviation than at least one other greedy-set member; it is reported separately and can be exploratory without sacrificing immediate expected reward.",
        "Same-history agreement compares the chosen action with all maximizers of Bayes-UCB (q_t=1-1/t) or the finite-horizon knowledge-gradient score, recomputed on this policy's own public state. Index ties use tolerance 1e-12. Mean top-set sizes are supplied because tied recommendations can inflate agreement. These agreements do not identify Jev's internal algorithm.",
        "## Exact two-arm decision-value checks",
        "E3 exact Q values and recorded decision losses were checked at every visited state. These Q values assume Bayes-optimal continuation after the current action. Useful non-greedy actions strictly improve on the best posterior-mean-maximizing action's Q value; inferior non-greedy actions are strictly worse; neutral choices are tied within tolerance. This distinguishes value from merely leaving the greedy set. Elsewhere, non-greedy or uncertainty-seeking behavior is descriptive and is not labeled useful or wasteful without a normative action-value reference.",
        markdown_table(exact),
        "This is a post hoc behavioral description with exploratory pointwise intervals and no multiple-comparison correction. Phase trajectories are descriptive associations: policies visit different states, and remaining horizon, evidence and time change together. Equal mean choices can gather valuable information; high agreement with a simple policy does not establish a common computation or mechanism.",
    ]
    sections.extend(f"![Phase behavior]({name})" for name in figures)
    path = output / "report.md"
    path.write_text("\n\n".join(sections) + "\n")
    manifest = {
        "experiments": experiments,
        "bootstrap_samples": samples,
        "completed_episode_rows": len(data),
        "source_hashes": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in [
                "behavior_audit.py",
                "baselines.py",
                "report.py",
                "followup_reports.py",
            ]
        },
        "inputs": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in inputs
            if p.exists()
        },
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--experiment", action="append")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    print(
        generate_behavior_report(
            args.run_dir,
            args.output_dir,
            experiments=args.experiment,
            samples=args.bootstrap_samples,
        )
    )


if __name__ == "__main__":
    main()
