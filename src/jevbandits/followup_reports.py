"""Offline, protocol-specific reports for the frozen E6 and E8 follow-ups."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .report import (
    _seed,
    bootstrap_mean,
    episode_summaries,
    markdown_table,
    paired_differences,
    paired_effect,
    read_jsonl,
    stratified_paired_effect,
)

FORMATS = (
    "nested",
    "path_explicit",
    "inline",
    "per_option",
    "advice_only",
    "dp_values",
)
COHORTS = ("random", "targeted")
LABEL_SCHEMES = ("original", "letters")
HORIZONS = (1, 2, 10)
E8_CELLS = (("close", 3), ("close", 10), ("prior", 3), ("prior", 10))
E8_CONTRASTS = (("means_direct", "counts_direct"), ("bayes_direct", "means_direct"))
NICE = {
    "nested": "Nested advice",
    "path_explicit": "Explicit path",
    "inline": "Inline instruction",
    "per_option": "Advice in each option",
    "advice_only": "Advice without evidence",
    "dp_values": "Exact action values",
    "counts_direct": "Jev: counts",
    "means_direct": "Jev: counts + means",
    "bayes_direct": "Jev: Bayesian summaries",
    "ts": "Thompson sampling",
    "greedy": "Posterior-mean greedy",
    "bayes_ucb": "Bayes-UCB",
    "knowledge_gradient": "Knowledge gradient",
    "ids": "Information-directed sampling",
    "finite_ap_index": "Finite-horizon AP index",
    "ucb1": "UCB1",
    "random": "Uniform random",
}
ADVICE_METRICS = (
    "advice_adherence",
    "distribution_advice_adherence",
    "optimal_agreement",
    "exact_loss",
    "distribution_exact_loss",
    "posterior_mean_greedy",
    "prediction_entropy",
    "choice_mismatch",
)


def advice_tables(data: pd.DataFrame, *, samples: int = 10_000):
    """Keep cohort/labels/horizon separate and average four calls per fixture."""
    group_cols = ["cohort", "label_scheme", "horizon", "format"]
    observations = data.copy()
    if not observations.empty:
        if (
            observations.decision_id.duplicated().any()
            or observations.duplicated(
                [*group_cols, "fixture_id", "label_order", "repeat"]
            ).any()
        ):
            raise ValueError("Duplicate advice decision or scientific cell")
        if (
            not set(observations.cohort) <= set(COHORTS)
            or not set(observations.label_scheme) <= set(LABEL_SCHEMES)
            or not set(observations.format) <= set(FORMATS)
            or not set(observations.horizon) <= set(HORIZONS)
        ):
            raise ValueError("Unexpected E6 design cell")
        if not observations.advice_present.eq(observations.format != "dp_values").all():
            raise ValueError(
                "Advice-present flag does not match frozen E6 intervention"
            )
    completeness, fixture_rows = [], []
    for cohort, labels, horizon, format_name in itertools.product(
        COHORTS, LABEL_SCHEMES, HORIZONS, FORMATS
    ):
        metadata = dict(zip(group_cols, (cohort, labels, horizon, format_name)))
        cell = observations
        if not observations.empty:
            for key, value in metadata.items():
                cell = cell.loc[cell[key] == value]
        complete = 0
        seen_fixtures = 0 if cell.empty else cell.fixture_id.nunique()
        if seen_fixtures > 100:
            raise ValueError("More fixtures than the frozen E6 design")
        if not cell.empty:
            for fixture_id, fixture in cell.groupby("fixture_id"):
                coordinates = set(zip(fixture.label_order, fixture.repeat))
                if len(fixture) != 4 or coordinates != {(0, 0), (0, 1), (1, 0), (1, 1)}:
                    continue
                complete += 1
                row = {
                    **metadata,
                    "fixture_id": fixture_id,
                    "advice_present": format_name != "dp_values",
                }
                for metric in ADVICE_METRICS:
                    if metric not in fixture:
                        continue
                    if format_name == "dp_values" and metric not in {
                        "optimal_agreement",
                        "exact_loss",
                        "distribution_exact_loss",
                    }:
                        continue
                    values = fixture[metric].to_numpy(dtype=float)
                    if not np.isfinite(values).all():
                        raise ValueError(f"Nonfinite E6 metric {metric}")
                    row[metric] = float(values.mean())
                fixture_rows.append(row)
        completeness.append(
            {
                **metadata,
                "expected_fixtures": 100,
                "observed_fixtures": seen_fixtures,
                "complete_fixtures": complete,
                "incomplete_observed_fixtures": seen_fixtures - complete,
                "missing_fixtures": 100 - seen_fixtures,
                "expected_queries": 400,
                "observed_queries": len(cell),
                "missing_queries": 400 - len(cell),
            }
        )
    fixtures = pd.DataFrame(fixture_rows)
    summary, contrasts, interactions = [], [], []
    if not fixtures.empty:
        for key, cell in fixtures.groupby(group_cols, sort=True):
            for metric in ADVICE_METRICS:
                if metric not in cell or cell[metric].isna().all():
                    continue
                values = cell[metric].dropna()
                mean, low, high = bootstrap_mean(
                    values, samples=samples, seed=_seed(f"E6:{key}:{metric}")
                )
                summary.append(
                    dict(zip(group_cols, key))
                    | {
                        "metric": metric,
                        "mean": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "n_fixtures": len(values),
                    }
                )
        for key, cell in fixtures.groupby(
            ["cohort", "label_scheme", "horizon"], sort=True
        ):
            cohort, labels, horizon = key
            for left in FORMATS[1:]:
                for metric in (
                    "advice_adherence",
                    "optimal_agreement",
                    "exact_loss",
                    "distribution_exact_loss",
                ):
                    if metric not in cell or (
                        left == "dp_values" and metric == "advice_adherence"
                    ):
                        continue
                    selected = cell.loc[cell.format.isin([left, "nested"])].dropna(
                        subset=[metric]
                    )
                    effect = paired_effect(
                        selected,
                        left,
                        "nested",
                        metric,
                        ["fixture_id"],
                        policy_col="format",
                        samples=samples,
                        seed=_seed(f"E6contrast:{key}:{left}:{metric}"),
                        unit="fixture",
                    )
                    primary = (
                        horizon == 10
                        and left in {"path_explicit", "inline", "per_option"}
                        and metric == "advice_adherence"
                    )
                    contrasts.append(
                        {
                            "cohort": cohort,
                            "label_scheme": labels,
                            "horizon": horizon,
                            **effect.__dict__,
                            "comparison_role": "primary_descriptive"
                            if primary
                            else "secondary_exploratory",
                            "expected_pairs": 100,
                            "missing_pairs": 100 - effect.n_pairs,
                        }
                    )
        # Within-fixture label and horizon contrasts remain secondary, never pooled across cohorts.
        for cohort in COHORTS:
            for fmt in FORMATS:
                block = fixtures.loc[
                    (fixtures.cohort == cohort) & (fixtures.format == fmt)
                ]
                for metric in ("advice_adherence", "optimal_agreement", "exact_loss"):
                    if metric not in block or (
                        fmt == "dp_values" and metric == "advice_adherence"
                    ):
                        continue
                    for horizon in HORIZONS:
                        selected = block.loc[block.horizon == horizon].dropna(
                            subset=[metric]
                        )
                        effect = paired_effect(
                            selected,
                            "letters",
                            "original",
                            metric,
                            ["fixture_id"],
                            policy_col="label_scheme",
                            samples=samples,
                            seed=_seed(f"E6labels:{cohort}:{fmt}:{horizon}:{metric}"),
                            unit="fixture",
                        )
                        interactions.append(
                            {
                                "cohort": cohort,
                                "format": fmt,
                                "contrast_type": "label_scheme",
                                "fixed_factor": f"horizon={horizon}",
                                **effect.__dict__,
                                "analysis_role": "secondary_exploratory",
                            }
                        )
                    for labels in LABEL_SCHEMES:
                        selected = (
                            block.loc[block.label_scheme == labels]
                            .dropna(subset=[metric])
                            .assign(horizon_name=lambda x: x.horizon.astype(str))
                        )
                        for high in (2, 10):
                            effect = paired_effect(
                                selected,
                                str(high),
                                "1",
                                metric,
                                ["fixture_id"],
                                policy_col="horizon_name",
                                samples=samples,
                                seed=_seed(
                                    f"E6horizon:{cohort}:{fmt}:{labels}:{high}:{metric}"
                                ),
                                unit="fixture",
                            )
                            interactions.append(
                                {
                                    "cohort": cohort,
                                    "format": fmt,
                                    "contrast_type": "horizon",
                                    "fixed_factor": f"labels={labels}",
                                    **effect.__dict__,
                                    "analysis_role": "secondary_exploratory",
                                }
                            )
    return (
        pd.DataFrame(completeness),
        pd.DataFrame(summary),
        pd.DataFrame(contrasts),
        pd.DataFrame(interactions),
    )


def evidence_primary(data: pd.DataFrame, *, samples: int = 10_000):
    """Exactly two equal-four-cell contrasts with both declared interval levels."""
    if samples < 2:
        raise ValueError("At least two resamples required")
    if not data.empty and data.duplicated(["episode_id", "policy"]).any():
        raise ValueError("Duplicate evidence episode/policy")
    if not data.empty and not set(zip(data.family, data.k)) <= set(E8_CELLS):
        raise ValueError("Unexpected E8 environment cell")
    rows, completeness, per_cell = [], [], []
    for left, right in E8_CONTRASTS:
        differences = []
        for family, k in E8_CELLS:
            cell = (
                data.loc[(data.family == family) & (data.k == k)]
                if not data.empty
                else pd.DataFrame(columns=["episode_id", "policy", "pseudo_regret"])
            )
            delta, nl, nr = paired_differences(
                cell, left, right, "pseudo_regret", ["episode_id"]
            )
            if nl > 80 or nr > 80:
                raise ValueError("More episodes than the frozen E8 design")
            differences.append(delta.to_numpy(dtype=float))
            completeness.append(
                {
                    "left": left,
                    "right": right,
                    "family": family,
                    "k": k,
                    "expected_pairs": 80,
                    "n_left": nl,
                    "n_right": nr,
                    "n_pairs": len(delta),
                    "missing_left": 80 - nl,
                    "missing_right": 80 - nr,
                    "missing_pairs": 80 - len(delta),
                }
            )
            mean, low, high = bootstrap_mean(
                delta,
                samples=samples,
                seed=_seed(f"e8_evidence_online:{left}:{right}:{family}:{k}"),
            )
            per_cell.append(
                {
                    "left": left,
                    "right": right,
                    "family": family,
                    "k": k,
                    "mean_difference": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_pairs": len(delta),
                    "analysis_role": "exploratory_cell",
                }
            )
        missing_cells = [
            f"{family}/K{k}"
            for (family, k), values in zip(E8_CELLS, differences, strict=True)
            if not len(values)
        ]
        result = {
            "left": left,
            "right": right,
            "n_pairs": sum(map(len, differences)),
            "expected_pairs": 320,
            "missing_pairs": 320 - sum(map(len, differences)),
            "missing_cells": ", ".join(missing_cells),
            "analysis_role": "prespecified_primary",
            "status": "missing_cell_no_pooled_estimate"
            if missing_cells
            else "complete"
            if all(len(x) == 80 for x in differences)
            else "incomplete_descriptive_only",
        }
        if missing_cells:
            result.update(
                {
                    key: np.nan
                    for key in [
                        "mean_difference",
                        "ci95_low",
                        "ci95_high",
                        "ci975_low",
                        "ci975_high",
                    ]
                }
            )
        else:
            rng = np.random.default_rng(
                _seed(f"e8_evidence_online:{left}:{right}:primary")
            )
            bootstrap = np.zeros(samples)
            for values in differences:
                for start in range(0, samples, 1000):
                    count = min(1000, samples - start)
                    bootstrap[start : start + count] += (
                        values[rng.integers(0, len(values), (count, len(values)))].mean(
                            axis=1
                        )
                        / 4
                    )
            intervals = (
                np.quantile(bootstrap, [0.025, 0.975, 0.0125, 0.9875])
                if all(len(x) > 1 for x in differences)
                else [np.nan] * 4
            )
            result.update(
                dict(
                    zip(
                        ["ci95_low", "ci95_high", "ci975_low", "ci975_high"],
                        map(float, intervals),
                    )
                )
            )
            result["mean_difference"] = float(np.mean([x.mean() for x in differences]))
        rows.append(result)
    return pd.DataFrame(rows), pd.DataFrame(completeness), pd.DataFrame(per_cell)


def _plot_advice(summary, output):
    if summary.empty:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = []
    for horizon in HORIZONS:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for ax, (cohort, labels) in zip(
            axes.flat, itertools.product(COHORTS, LABEL_SCHEMES), strict=True
        ):
            cell = summary.loc[
                (summary.cohort == cohort)
                & (summary.label_scheme == labels)
                & (summary.horizon == horizon)
                & (summary.metric == "advice_adherence")
            ].set_index("format")
            formats = [
                fmt for fmt in FORMATS if fmt in cell.index and fmt != "dp_values"
            ]
            for position, fmt in enumerate(formats):
                row = cell.loc[fmt]
                ax.hlines(
                    position, row.ci_low, row.ci_high, color="#0072B2", linewidth=2
                )
                ax.plot(row["mean"], position, "o", color="#0072B2", markersize=5)
            ax.set_yticks(range(len(formats)), [NICE[x] for x in formats])
            ax.invert_yaxis()
            ax.set(
                xlim=(-0.02, 1.02),
                title=f"{cohort.title()} cohort · {'arm_00 / arm_01' if labels == 'original' else 'A / B'}",
                xlabel="Advice adherence",
            )
            ax.grid(axis="x", alpha=0.2)
        fig.suptitle(f"E6: horizon {horizon} · fixture bootstrap 95% intervals")
        fig.tight_layout()
        filename = f"advice_h{horizon}.png"
        fig.savefig(output / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
        names.append(filename)
    return names


def _plot_evidence(primary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = primary.dropna(subset=["mean_difference"])
    if valid.empty:
        return []
    fig, ax = plt.subplots(figsize=(10, 3.8))
    labels = []
    for y, (_, row) in enumerate(valid.iterrows()):
        ax.hlines(
            y,
            row.ci975_low,
            row.ci975_high,
            color="#999999",
            linewidth=5,
            label="97.5% Bonferroni interval" if y == 0 else None,
        )
        ax.hlines(
            y,
            row.ci95_low,
            row.ci95_high,
            color="#0072B2",
            linewidth=2,
            label="95% pointwise interval" if y == 0 else None,
        )
        ax.plot(row.mean_difference, y, "o", color="#0072B2", markersize=6)
        labels.append(f"{NICE[row.left]}\nminus {NICE[row.right]}")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#444444", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Difference in cumulative pseudo-regret\nNegative favors the first policy",
        title="E8: two prespecified contrasts, four cells equally weighted",
    )
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    name = "evidence_primary.png"
    fig.savefig(output / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [name]


def _manifest(source, output, mode, samples):
    files = [
        "manifest.json",
        "fixtures.json",
        "tasks.json",
        "diagnostics.jsonl",
        "episodes.jsonl",
    ]
    manifest = {
        "mode": mode,
        "bootstrap_samples": samples,
        "analysis_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "shared_analysis_sha256": hashlib.sha256(
            Path(__file__).with_name("report.py").read_bytes()
        ).hexdigest(),
        "inputs": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in files
            if (source / name).exists()
        },
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def report_advice(run_dir, output_dir=None, *, samples=10_000):
    source = Path(run_dir)
    output = Path(output_dir) if output_dir else source / "report"
    output.mkdir(parents=True, exist_ok=True)
    data = read_jsonl(source / "diagnostics.jsonl")
    completeness, summary, contrasts, interactions = advice_tables(
        data, samples=samples
    )
    for name, frame in [
        ("completeness", completeness),
        ("summary", summary),
        ("contrasts", contrasts),
        ("interactions", interactions),
    ]:
        frame.to_csv(output / f"advice_{name}.csv", index=False)
    figures = _plot_advice(summary, output)
    primary = (
        contrasts.loc[contrasts.comparison_role == "primary_descriptive"]
        if not contrasts.empty
        else contrasts
    )
    sections = [
        "# E6: advice binding follow-up",
        f"Recorded decisions: {len(data):,} of 28,800. Complete four-query fixture cells: {int(completeness.complete_fixtures.sum()):,} of 7,200. Missingness is listed for every cohort, label scheme, horizon and format in `advice_completeness.csv`.",
        "Primary descriptive endpoint: chosen-ID adherence, only when an advice pointer was actually supplied. Exact-action-value inputs have no advice pointer and are evaluated by tie-aware optimal-action agreement and Q loss, never advice adherence. A different optimal arm can have zero loss but fail pointer adherence in advice-present conditions.",
        "All estimates average two assignments and two repeats within each complete fixture. Incomplete four-query fixture cells are disclosed and excluded. Intervals resample fixtures within cohort and retain label schemes and horizons as separate factors. Random and targeted cohorts are never pooled. Targeted states were selected using exact exploration value and do not represent a natural population.",
        "## Primary horizon-10 descriptive contrasts",
        markdown_table(primary),
        "Differences are intervention minus nested advice. Positive adherence differences favor the intervention. These 95% percentile intervals are exploratory, not simultaneous tests or equivalence evidence. Zero-width boundary bootstrap intervals reflect empirical resampling and do not guarantee perfect population performance.",
        "## Secondary outcomes",
        markdown_table(
            summary.loc[summary.metric.isin(["optimal_agreement", "exact_loss"])]
            if not summary.empty
            else summary
        ),
        "Advice-only contrasts, action-value comparisons, and other horizons are secondary exploratory analyses. Paired label-scheme and horizon contrasts appear in `advice_interactions.csv`. Numeric action values supply planning results; their successful use is distinct from independently computing those values. Label and advice-format interventions change communication, not necessarily a single cognitive mechanism.",
    ]
    sections += [
        f"![Advice adherence at horizon {h}]({name})"
        for h, name in zip(HORIZONS, figures)
    ]
    path = output / "report.md"
    path.write_text("\n\n".join(sections) + "\n")
    _manifest(source, output, "advice", samples)
    return path


def report_evidence(run_dir, output_dir=None, *, samples=10_000):
    source = Path(run_dir)
    output = Path(output_dir) if output_dir else source / "report"
    output.mkdir(parents=True, exist_ok=True)
    data = read_jsonl(source / "episodes.jsonl")
    incomplete = pd.DataFrame()
    if not data.empty:
        if (
            not data.experiment.eq("e8_evidence_online").all()
            or not data.horizon.eq(100).all()
        ):
            raise ValueError("Unexpected E8 experiment or horizon")
        incomplete = data.loc[~data.completed.eq(True)]
        data = data.loc[data.completed.eq(True)].copy()
    primary, completeness, per_cell = evidence_primary(data, samples=samples)
    summary = (
        episode_summaries(data, samples=samples) if not data.empty else pd.DataFrame()
    )
    baseline_rows = []
    if not data.empty:
        for policy in ("counts_direct", "means_direct", "bayes_direct"):
            for baseline in sorted(
                set(data.policy) - {"counts_direct", "means_direct", "bayes_direct"}
            ):
                # Require all four cells; classical comparisons are secondary, even versus TS.
                if all(((data.family == f) & (data.k == k)).any() for f, k in E8_CELLS):
                    result = stratified_paired_effect(
                        data,
                        policy,
                        baseline,
                        "pseudo_regret",
                        ["episode_id"],
                        ["family", "k"],
                        samples=samples,
                        seed=_seed(f"E8secondary:{policy}:{baseline}"),
                    )
                    baseline_rows.append(
                        {**result, "analysis_role": "secondary_exploratory"}
                    )
    frames = {
        "primary": primary,
        "completeness": completeness,
        "per_cell": per_cell,
        "summary": summary,
        "baseline_comparisons": pd.DataFrame(baseline_rows),
        "incomplete_episodes": incomplete,
    }
    for name, frame in frames.items():
        frame.to_csv(output / f"evidence_{name}.csv", index=False)
    figures = _plot_evidence(primary, output)
    sections = [
        "# E8: fresh online evidence ladder",
        "The endpoint is cumulative pseudo-regret. Negative left-minus-right differences favor the first policy. Exactly two overall contrasts were specified before E8 collection: means versus counts, and Bayesian summaries versus means.",
        "## Prespecified overall contrasts",
        markdown_table(primary),
        "Each estimate equally weights close/prior environments crossed with 3/10 arms. Bootstrap resampling pairs whole episodes within each cell. The 95% intervals are pointwise; 97.5% intervals use quantiles 0.0125 and 0.9875 and target Bonferroni 95% family coverage for these two contrasts only, subject to bootstrap approximation. They do not cover subgroup or classical comparisons. Crossing zero does not establish equivalence.",
        "## Completeness",
        markdown_table(completeness),
        f"Recorded incomplete episode rows excluded: {len(incomplete)}. The frozen target is 80 paired tasks per cell, 320 overall. A missing cell produces no pooled estimate; an available but incomplete four-cell comparison is labeled descriptive only. No cell is silently omitted and no sample size is redefined.",
        "## Per-cell exploratory contrasts",
        markdown_table(per_cell),
        "## Classical comparisons (secondary)",
        markdown_table(frames["baseline_comparisons"]),
        "Counts, means, and Bayesian summaries contain the same underlying evidence under the stated prior. Effects concern the usefulness of supplied computation and its representation package on adaptive trajectories. These fresh tasks evaluate E2-motivated hypotheses; they do not reveal internal reasoning or establish broad superiority. Baseline comparisons, reward, coverage, switching and abandonment summaries are exploratory.",
    ]
    sections += [f"![Two prespecified evidence contrasts]({name})" for name in figures]
    path = output / "report.md"
    path.write_text("\n\n".join(sections) + "\n")
    _manifest(source, output, "evidence", samples)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["advice", "evidence"])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    reporter = report_advice if args.mode == "advice" else report_evidence
    print(reporter(args.run_dir, args.output_dir, samples=args.bootstrap_samples))


if __name__ == "__main__":
    main()
