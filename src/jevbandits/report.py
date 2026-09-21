"""Offline, paired analyses for the Jev bandit study.

The resampling unit is an episode (or an underlying fixed-state fixture), never
an individual pull or a repeat query. Intervals are exploratory percentile
bootstrap intervals, not simultaneous inference or evidence of equivalence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Effect:
    """Mean left-minus-right difference; lower regret is better."""

    left: str
    right: str
    metric: str
    mean_difference: float
    ci_low: float
    ci_high: float
    n_pairs: int
    n_left: int
    n_right: int
    unit: str = "episode"


def _seed(value: str, seed: int = 20260920) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{value}".encode()).digest()[:8])


def bootstrap_mean(
    values: Sequence[float], *, seed: int = 20260920, samples: int = 10_000
) -> tuple[float, float, float]:
    """Bootstrap independent units in bounded-memory batches.

    A single unit has an estimate but no estimable uncertainty interval. Missing
    or infinite inputs are errors rather than silently dropped observations.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Bootstrap values must be a finite one-dimensional array")
    if samples < 2:
        raise ValueError("At least two bootstrap resamples are required")
    if not len(x):
        return np.nan, np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    batch_size = max(1, min(1000, 1_000_000 // len(x)))
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        indices = rng.integers(0, len(x), size=(count, len(x)))
        means[start : start + count] = x[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(x.mean()), float(low), float(high)


def paired_differences(
    frame: pd.DataFrame,
    left: str,
    right: str,
    metric: str,
    pair_cols: Sequence[str],
    *,
    policy_col: str = "policy",
) -> tuple[pd.Series, int, int]:
    """Join only complete pairs; reject accidental pseudo-replication."""
    if not pair_cols:
        raise ValueError("An independent pairing identifier is required")
    subset = frame.loc[frame[policy_col].isin([left, right])]
    if subset.duplicated([*pair_cols, policy_col]).any():
        raise ValueError(
            "Duplicate policy/pair rows: aggregate to independent units first"
        )
    if subset[list(pair_cols)].isna().any().any():
        raise ValueError("Pairing identifiers cannot be missing")
    a = subset.loc[subset[policy_col] == left].set_index(list(pair_cols))[metric]
    b = subset.loc[subset[policy_col] == right].set_index(list(pair_cols))[metric]
    for series in (a, b):
        if not np.isfinite(series.to_numpy(dtype=float)).all():
            raise ValueError(
                "Metric values must be finite; report incomplete data separately"
            )
    joined = pd.concat([a.rename("left"), b.rename("right")], axis=1).dropna()
    return joined["left"] - joined["right"], len(a), len(b)


def paired_effect(
    frame: pd.DataFrame,
    left: str,
    right: str,
    metric: str,
    pair_cols: Sequence[str],
    *,
    policy_col: str = "policy",
    seed: int = 20260920,
    samples: int = 10_000,
    unit: str = "episode",
) -> Effect:
    difference, n_left, n_right = paired_differences(
        frame, left, right, metric, pair_cols, policy_col=policy_col
    )
    mean, low, high = bootstrap_mean(difference, seed=seed, samples=samples)
    return Effect(
        left, right, metric, mean, low, high, len(difference), n_left, n_right, unit
    )


def stratified_paired_effect(
    frame: pd.DataFrame,
    left: str,
    right: str,
    metric: str,
    pair_cols: Sequence[str],
    cell_cols: Sequence[str],
    *,
    seed: int = 20260920,
    samples: int = 10_000,
) -> dict:
    """Equal-weight cells, paired resampling within each cell.

    A cell missing an entire policy makes the pooled effect undefined. This
    avoids silently changing the target estimand after failures.
    """
    if not cell_cols:
        raise ValueError("Cell identifiers are required for a stratified estimate")
    diffs = []
    n_left = n_right = 0
    for _, cell in frame.groupby(list(cell_cols), dropna=False, sort=True):
        difference, nl, nr = paired_differences(cell, left, right, metric, pair_cols)
        n_left += nl
        n_right += nr
        if not len(difference):
            return {
                **asdict(
                    Effect(
                        left, right, metric, np.nan, np.nan, np.nan, 0, n_left, n_right
                    )
                ),
                "n_cells": frame.groupby(list(cell_cols), dropna=False).ngroups,
                "status": "missing_paired_cell",
            }
        diffs.append(difference.to_numpy(dtype=float))
    if not diffs:
        raise ValueError("No cells to pool")
    rng = np.random.default_rng(seed)
    boot = np.zeros(samples)
    for values in diffs:
        batch_size = max(1, min(1000, 1_000_000 // len(values)))
        for start in range(0, samples, batch_size):
            count = min(batch_size, samples - start)
            idx = rng.integers(0, len(values), size=(count, len(values)))
            boot[start : start + count] += values[idx].mean(axis=1) / len(diffs)
    mean = float(np.mean([v.mean() for v in diffs]))
    low, high = (
        (np.nan, np.nan)
        if any(len(v) < 2 for v in diffs)
        else np.quantile(boot, [0.025, 0.975])
    )
    return {
        **asdict(
            Effect(
                left,
                right,
                metric,
                mean,
                float(low),
                float(high),
                sum(map(len, diffs)),
                n_left,
                n_right,
            )
        ),
        "n_cells": len(diffs),
        "status": "ok",
    }


def fixture_effect(
    frame: pd.DataFrame,
    left: str,
    right: str,
    metric: str,
    *,
    fixture_col: str = "fixture_id",
    condition_col: str = "condition",
    seed: int = 20260920,
    samples: int = 10_000,
) -> Effect:
    """Average query repetitions first and bootstrap whole underlying fixtures.

    The caller must select a balanced intervention contrast before this call:
    e.g. one horizon and framing, with the same label assignments per condition.
    Six hand-selected fixtures support descriptive contrasts, not a population
    generalization. The interval reflects fixture sensitivity only.
    """
    means = frame.groupby([fixture_col, condition_col], as_index=False)[metric].mean()
    return paired_effect(
        means,
        left,
        right,
        metric,
        [fixture_col],
        policy_col=condition_col,
        seed=seed,
        samples=samples,
        unit="underlying fixture",
    )


def markdown_table(frame: pd.DataFrame, *, digits: int = 4) -> str:
    """Small dependency-free table writer (pandas.to_markdown needs tabulate)."""
    if frame.empty:
        return "No complete observations."

    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = ["| " + " | ".join(map(str, frame.columns)) + " |"]
    rows.append("| " + " | ".join(["---"] * len(frame.columns)) + " |")
    rows.extend(
        "| " + " | ".join(fmt(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return pd.DataFrame(rows)


def episode_summaries(episodes: pd.DataFrame, *, samples: int = 10_000) -> pd.DataFrame:
    """One mean/interval per policy, task cell and metric."""
    keys = ["experiment", "family", "k", "horizon", "policy"]
    metrics = [
        "pseudo_regret",
        "reward",
        "exact_loss",
        "optimal_agreement",
        "posterior_mean_greedy",
        "unseen_fraction",
        "switches",
        "arms_visited",
        "best_arm_fraction",
        "suffix_failure",
        "advice_adherence",
    ]
    rows = []
    for key, cell in episodes.groupby(keys, sort=True, dropna=False):
        if cell.episode_id.duplicated().any():
            raise ValueError(f"Duplicate episode rows for {key}")
        for metric in metrics:
            if metric not in cell or not cell[metric].notna().any():
                continue
            values = cell[metric].dropna().to_numpy(dtype=float)
            mean, low, high = bootstrap_mean(
                values, samples=samples, seed=_seed(f"{key}:{metric}")
            )
            rows.append(
                dict(zip(keys, key))
                | {
                    "metric": metric,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n": len(values),
                    "n_metric_missing": len(cell) - len(values),
                }
            )
    return pd.DataFrame(rows)


def episode_effects(
    episodes: pd.DataFrame, *, samples: int = 10_000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, pooled = [], []
    cell_cols = ["family", "k", "horizon"]
    for experiment, exp in episodes.groupby("experiment", sort=True):
        policies = set(exp.policy)
        learned = sorted(
            p
            for p in policies
            if p.endswith(("direct", "sample")) or p == "thompson_prompt"
        )
        contrasts = {(p, "ts") for p in learned if "ts" in policies}
        if {"bayes_direct", "counts_direct"} <= policies:
            contrasts.add(("bayes_direct", "counts_direct"))
        for prefix in ("counts", "bayes"):
            if {f"{prefix}_direct", f"{prefix}_sample"} <= policies:
                contrasts.add((f"{prefix}_sample", f"{prefix}_direct"))
        if experiment == "e5_strategy" and "bayes_direct" in policies:
            contrasts.update(
                (p, "bayes_direct") for p in learned if p != "bayes_direct"
            )
        if "exact" in policies:
            contrasts.update((p, "exact") for p in learned)
        for left, right in sorted(contrasts):
            for metric in ("pseudo_regret", "reward", "exact_loss"):
                if (
                    metric not in exp
                    or exp.loc[exp.policy.isin([left, right]), metric].isna().any()
                ):
                    continue
                for key, cell in exp.groupby(cell_cols, sort=True):
                    effect = paired_effect(
                        cell,
                        left,
                        right,
                        metric,
                        ["episode_id"],
                        samples=samples,
                        seed=_seed(f"{experiment}:{key}:{left}:{right}:{metric}"),
                    )
                    rows.append(
                        {
                            "experiment": experiment,
                            **dict(zip(cell_cols, key)),
                            **asdict(effect),
                        }
                    )
                # Pool only across cells within one horizon, never across experiments.
                for horizon, block in exp.groupby("horizon", sort=True):
                    effect = stratified_paired_effect(
                        block,
                        left,
                        right,
                        metric,
                        ["episode_id"],
                        ["family", "k"],
                        samples=samples,
                        seed=_seed(
                            f"{experiment}:{horizon}:{left}:{right}:{metric}:pooled"
                        ),
                    )
                    pooled.append(
                        {"experiment": experiment, "horizon": horizon, **effect}
                    )
    return pd.DataFrame(rows), pd.DataFrame(pooled)


def diagnostic_summaries(
    diagnostics: pd.DataFrame, *, samples: int = 10_000
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Balanced-state descriptions, contrasts, and label sensitivity."""
    summary, effects, label_rows = [], [], []
    keys = ["experiment", "horizon", "policy"]
    metrics = [
        "exact_loss",
        "distribution_exact_loss",
        "optimal_agreement",
        "posterior_mean_greedy",
        "prediction_entropy",
    ]
    for key, group in diagnostics.groupby(keys, sort=True):
        for metric in metrics:
            if metric not in group or not group[metric].notna().any():
                continue
            # Each display order receives equal weight, irrespective of retries.
            by_order = group.groupby(["fixture_id", "label_order"])[metric].mean()
            by_fixture = by_order.groupby("fixture_id").mean().dropna()
            mean, low, high = bootstrap_mean(
                by_fixture, samples=samples, seed=_seed(f"{key}:{metric}")
            )
            summary.append(
                dict(zip(keys, key))
                | {
                    "metric": metric,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n_fixtures": len(by_fixture),
                    "n_queries": int(group[metric].notna().sum()),
                }
            )
        if "canonical_action" in group:
            fractions = group.assign(
                chose_canonical_a=(group.canonical_action == 0).astype(float)
            )
            by_order = (
                fractions.groupby(["fixture_id", "label_order"])
                .chose_canonical_a.mean()
                .unstack()
            )
            if 0 in by_order and 1 in by_order:
                difference = (by_order[0] - by_order[1]).dropna()
                for metric, values in (
                    ("absolute_choice_order_sensitivity", difference.abs()),
                    ("signed_choice_order_sensitivity", difference),
                ):
                    mean, low, high = bootstrap_mean(
                        values, samples=samples, seed=_seed(f"{key}:{metric}")
                    )
                    label_rows.append(
                        dict(zip(keys, key))
                        | {
                            "metric": metric,
                            "mean": mean,
                            "ci_low": low,
                            "ci_high": high,
                            "n_fixtures": len(values),
                        }
                    )
    # Full pairwise policy contrasts are secondary; do not select comparisons by result.
    for (experiment, horizon), exp in diagnostics.groupby(
        ["experiment", "horizon"], sort=True
    ):
        policies = sorted(exp.policy.unique())
        for index, left in enumerate(policies):
            for right in policies[index + 1 :]:
                for metric in metrics:
                    if metric not in exp:
                        continue
                    subset = exp.loc[exp.policy.isin([left, right])]
                    if subset[metric].isna().any():
                        continue
                    # Match order before fixture averaging; reject incomplete display pairs.
                    order_means = subset.groupby(
                        ["fixture_id", "label_order", "policy"], as_index=False
                    )[metric].mean()
                    difference, _, _ = paired_differences(
                        order_means, left, right, metric, ["fixture_id", "label_order"]
                    )
                    cluster_differences = difference.groupby("fixture_id").mean()
                    mean, low, high = bootstrap_mean(
                        cluster_differences,
                        samples=samples,
                        seed=_seed(f"{experiment}:{horizon}:{left}:{right}:{metric}"),
                    )
                    effects.append(
                        {
                            "experiment": experiment,
                            "horizon": horizon,
                            "left": left,
                            "right": right,
                            "metric": metric,
                            "mean_difference": mean,
                            "ci_low": low,
                            "ci_high": high,
                            "n_fixtures": len(cluster_differences),
                            "unit": "underlying fixture",
                        }
                    )
    return pd.DataFrame(summary), pd.DataFrame(effects), pd.DataFrame(label_rows)


def trajectory_summaries(episodes: pd.DataFrame) -> pd.DataFrame:
    """Descriptive episode-average curves without treating pulls as replicates."""
    keys = ["experiment", "family", "k", "horizon", "policy"]
    totals = {}
    for _, episode in episodes.iterrows():
        trace = episode.get("trace")
        if not isinstance(trace, list):
            continue
        regret = reward = 0.0
        for point in trace:
            regret += point["pseudo_regret"]
            reward += point["reward"]
            key = tuple(episode[k] for k in keys) + (point["turn"],)
            sums = totals.setdefault(key, [0, 0.0, 0.0, 0.0])
            sums[0] += 1
            sums[1] += regret
            sums[2] += reward
            sums[3] += float(not point["posterior_mean_greedy"])
    return pd.DataFrame(
        [
            dict(zip([*keys, "turn"], key))
            | {
                "n_episodes": value[0],
                "mean_cumulative_pseudo_regret": value[1] / value[0],
                "mean_cumulative_reward": value[2] / value[0],
                "non_greedy_fraction": value[3] / value[0],
            }
            for key, value in sorted(totals.items())
        ]
    )


def _trajectory_figures(trajectories: pd.DataFrame, output: Path) -> list[str]:
    if trajectories.empty:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for (experiment, family), exp in trajectories.groupby(
        ["experiment", "family"], sort=True
    ):
        arm_counts = sorted(exp.k.unique())
        fig, axes = plt.subplots(
            1, len(arm_counts), figsize=(4 * len(arm_counts), 4), squeeze=False
        )
        for ax, k in zip(axes.flat, arm_counts):
            for policy, curve in exp.loc[exp.k == k].groupby("policy", sort=True):
                curve = curve.sort_values("turn")
                ax.plot(
                    curve.turn,
                    curve.mean_cumulative_pseudo_regret,
                    label=policy,
                    linewidth=1.1,
                )
            ax.set(
                title=f"{k} arms", xlabel="Pull", ylabel="Mean cumulative pseudo-regret"
            )
            ax.grid(alpha=0.2)
        axes[0, -1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
        fig.suptitle(f"{experiment}, {family}: descriptive episode means")
        fig.tight_layout()
        name = f"{experiment}_{family}_trajectories.png"
        fig.savefig(output / name, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(name)
    return paths


def _figures(
    summary: pd.DataFrame, diagnostic: pd.DataFrame, output: Path
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    if not summary.empty:
        regrets = summary.loc[summary.metric == "pseudo_regret"]
        for experiment, exp in regrets.groupby("experiment", sort=True):
            families = sorted(exp.family.unique())
            fig, axes = plt.subplots(
                1, len(families), figsize=(5 * len(families), 4.5), squeeze=False
            )
            for ax, family in zip(axes.flat, families):
                for policy, values in exp.loc[exp.family == family].groupby(
                    "policy", sort=True
                ):
                    values = values.sort_values("k")
                    ax.plot(
                        values.k, values["mean"], marker="o", markersize=3, label=policy
                    )
                    ax.fill_between(values.k, values.ci_low, values.ci_high, alpha=0.08)
                ax.set(
                    title=family,
                    xlabel="Number of arms",
                    ylabel="Mean cumulative pseudo-regret",
                )
                ax.set_xticks(sorted(exp.k.unique()))
                ax.grid(alpha=0.2)
            axes[0, -1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
            fig.suptitle(f"{experiment}: episode bootstrap 95% intervals")
            fig.tight_layout()
            name = f"{experiment}_regret.png"
            fig.savefig(output / name, dpi=180, bbox_inches="tight")
            plt.close(fig)
            paths.append(name)
    if not diagnostic.empty:
        for experiment, exp in diagnostic.groupby("experiment", sort=True):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            for ax, metric in zip(axes, ("optimal_agreement", "exact_loss")):
                for policy, values in exp.loc[exp.metric == metric].groupby(
                    "policy", sort=True
                ):
                    values = values.sort_values("horizon")
                    ax.plot(
                        values.horizon,
                        values["mean"],
                        marker="o",
                        markersize=3,
                        label=policy,
                    )
                    ax.fill_between(
                        values.horizon, values.ci_low, values.ci_high, alpha=0.10
                    )
                ax.set(xlabel="Remaining horizon", ylabel=metric.replace("_", " "))
                ax.grid(alpha=0.2)
            axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
            fig.suptitle(f"{experiment}: underlying-fixture bootstrap 95% intervals")
            fig.tight_layout()
            name = f"{experiment}_diagnostics.png"
            fig.savefig(output / name, dpi=180, bbox_inches="tight")
            plt.close(fig)
            paths.append(name)
    return paths


def generate_report(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    bootstrap_samples: int = 10_000,
) -> Path:
    """Create CSVs, scientific plots and a Markdown report without API access."""
    source = Path(run_dir)
    output = Path(output_dir) if output_dir is not None else source / "report"
    output.mkdir(parents=True, exist_ok=True)
    episodes = read_jsonl(source / "episodes.jsonl")
    diagnostics = read_jsonl(source / "diagnostics.jsonl")
    if not diagnostics.empty and diagnostics["decision_id"].duplicated().any():
        raise ValueError(
            "Duplicate diagnostic decision IDs; recovery records must be deduplicated"
        )
    incomplete = pd.DataFrame()
    if not episodes.empty:
        completed = episodes["completed"].eq(True)
        incomplete = episodes.loc[~completed].copy()
        episodes = episodes.loc[completed].copy()
    if episodes.empty and diagnostics.empty:
        raise ValueError("No completed episode or diagnostic results are available")
    summary = (
        episode_summaries(episodes, samples=bootstrap_samples)
        if not episodes.empty
        else pd.DataFrame()
    )
    effects, pooled = (
        episode_effects(episodes, samples=bootstrap_samples)
        if not episodes.empty
        else (pd.DataFrame(), pd.DataFrame())
    )
    d_summary, d_effects, labels = (
        diagnostic_summaries(diagnostics, samples=bootstrap_samples)
        if not diagnostics.empty
        else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    )
    frames = {
        "episode_summary": summary,
        "paired_effects": effects,
        "pooled_effects": pooled,
        "diagnostic_summary": d_summary,
        "diagnostic_effects": d_effects,
        "label_sensitivity": labels,
        "incomplete_episodes": incomplete,
        "trajectory_summary": trajectory_summaries(episodes),
    }
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    figures = _figures(summary, d_summary, output)
    figures.extend(_trajectory_figures(frames["trajectory_summary"], output))
    sections = [
        "# Jev bandit experimental results",
        (
            "This report is generated from recorded observations; it does not query Jev. "
            "Positive paired pseudo-regret differences favor the comparator (right policy); "
            "negative differences favor the left policy. Positive reward differences favor the left policy."
        ),
        "## Analysis and provenance",
        (
            f"Input directory: `{source}`. Completed episode rows: {len(episodes):,}; "
            f"recorded incomplete rows excluded: {len(incomplete):,}; fixed-state queries: {len(diagnostics):,}. "
            f"Intervals use {bootstrap_samples:,} percentile bootstrap resamples with fixed analysis seeds."
        ),
        (
            "Episode comparisons pair identical environment IDs and bootstrap whole episodes. "
            "Fixed-state comparisons average repeated calls and label orders within each underlying fixture, "
            "then bootstrap fixtures. Repeated calls are not independent environments. "
            "Pooled online effects weight task cells equally within each experiment and horizon. "
            "All intervals are exploratory, pointwise, and unadjusted for multiple comparisons. "
            "An interval overlapping zero does not establish equivalence. "
            "A single independent unit has no reported interval."
        ),
    ]
    accounting_path = source / "accounting.json"
    if accounting_path.exists():
        accounting = json.loads(accounting_path.read_text())
        sections += [
            "## API usage and cost",
            markdown_table(pd.DataFrame([accounting])),
            (
                "Known cost is computed from reported usage. Uncertain reserves cover possibly billed "
                "failed attempts and are not verified account charges. Multiple isolated decision "
                "questions can share one HTTP request; decisions and request counts differ."
            ),
        ]
    if not pooled.empty:
        sections += [
            "## Online experiments",
            "Equal-cell-weight cumulative pseudo-regret differences:",
        ]
        for experiment, exp in pooled.loc[pooled.metric == "pseudo_regret"].groupby(
            "experiment", sort=True
        ):
            sections += [
                f"### {experiment}",
                markdown_table(
                    exp[
                        [
                            "left",
                            "right",
                            "mean_difference",
                            "ci_low",
                            "ci_high",
                            "n_pairs",
                            "n_cells",
                        ]
                    ]
                ),
            ]
            main = exp.loc[
                (exp.left == "bayes_direct") & (exp.right == "counts_direct")
            ]
            if len(main):
                row = main.iloc[0]
                direction = "lower" if row.mean_difference < 0 else "higher"
                sections.append(
                    f"Supplying Bayesian summaries was associated with {abs(row.mean_difference):.3f} "
                    f"{direction} cumulative pseudo-regret per episode on the equal-cell-weight scale "
                    f"(assisted minus counts: {row.mean_difference:.3f}, 95% interval "
                    f"[{row.ci_low:.3f}, {row.ci_high:.3f}]). This intervention changes information "
                    "presentation and arithmetic accessibility, not the underlying observed evidence."
                )
    if not d_summary.empty:
        sections += ["## Fixed-state experiments"]
        for experiment, exp in d_summary.groupby("experiment", sort=True):
            sections += [
                f"### {experiment}",
                markdown_table(
                    exp.loc[
                        exp.metric.isin(["optimal_agreement", "exact_loss"]),
                        [
                            "horizon",
                            "policy",
                            "metric",
                            "mean",
                            "ci_low",
                            "ci_high",
                            "n_fixtures",
                        ],
                    ]
                ),
            ]
        sections.append(
            "Exact action-value loss measures expected remaining reward sacrificed by the first action "
            "assuming optimal continuation. Optimal-action agreement is tie-aware and can treat tiny "
            "and consequential mistakes identically, so it must be read alongside action-value loss. "
            "Long-horizon states without exact values are behavioral diagnostics only. "
            "Label sensitivity is in `label_sensitivity.csv`; its absolute version measures observed "
            "choice changes and includes repeated-call variability, so it is not pure systematic label bias."
        )
        sections.append(
            "Fixtures are designed state panels: E1 has six named anchors plus seeded random states; "
            "E2 uses a disjoint seed namespace and entirely random states. Random states have arm counts "
            "chosen from 0, 2, 5, 10 and 20 and successes uniform from zero to that count. "
            "This is not the distribution of states visited by a policy and is not a draw from the "
            "posterior-predictive process. Fixture bootstrap intervals describe sensitivity across "
            "this panel. They do not by themselves generalize to online performance. The numeric "
            "action-value and recommendation conditions supply outputs of an exact planning algorithm; "
            "their comparison measures the use of supplied computation, not independent planning ability."
        )
    sections += ["## Figures"]
    sections.extend(
        f"![{name.replace('_', ' ').removesuffix('.png')}]({name})" for name in figures
    )
    sections += [
        "## Interpretation boundaries and next tests",
        (
            "Thompson sampling and Bayes-UCB are useful Bayesian algorithms, not exact finite-horizon "
            "optima. The exact two-arm panel supplies a separate optimality reference. Posterior-greedy "
            "choices do not prove absence of exploration reasoning, and non-greedy choices do not prove "
            "valuable exploration. Reward and exact action-value loss determine whether the behavior helps."
        ),
        (
            "The service's probabilities describe answer choices, not reward probabilities or the Bayesian "
            "probability that an arm is best. Direct and sampled policies differ in their action-selection "
            "rule; interpreting sampling gains requires comparison with simple randomized baselines. "
            "Counts and Bayesian summaries contain the same information under the stated prior. "
            "Better assisted performance supports a representation or computation benefit, not extra data."
        ),
        (
            "The experiments use stationary Bernoulli rewards, supplied sufficient statistics, anonymous "
            "arms, and one pinned model. They do not establish general Bayesian competence or incompetence, "
            "an internal algorithm, transfer to nonstationary environments, or benefits for reasoning-model "
            "harnesses. Stress-test families and matched-prior tasks should be interpreted separately. "
            "Any outcome-informed prompt changes require fresh evaluation seeds and a new experiment ID."
        ),
        (
            "Next investigations should follow the observed contrasts: check whether numerical assistance "
            "reduces meaningful action-value loss, whether horizon framing changes valuable exploration, "
            "and whether any sampling benefit persists against TS and uncertainty-index controls. "
            "Validate promising effects on fresh tasks before making publication-level superiority claims."
        ),
        "## Machine-readable outputs",
        "\n".join(f"- [{name}.csv]({name}.csv)" for name in frames),
    ]
    path = output / "report.md"
    path.write_text("\n\n".join(sections) + "\n")
    manifest = {
        "bootstrap_samples": bootstrap_samples,
        "analysis_seed": 20260920,
        "inputs": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source / "episodes.jsonl", source / "diagnostics.jsonl")
            if p.exists()
        },
        "completed_episode_rows": len(episodes),
        "incomplete_episode_rows": len(incomplete),
        "diagnostic_rows": len(diagnostics),
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return path
