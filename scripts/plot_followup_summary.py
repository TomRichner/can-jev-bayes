"""Rebuild interpretation figures from exported E8-E10 aggregate results; no API."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def interval_pair(ax, row, y, low95, high95, low975, high975, *, labels=False):
    ax.hlines(
        y,
        row[low975],
        row[high975],
        color="#aaaaaa",
        linewidth=7,
        label="97.5% Bonferroni interval" if labels else None,
    )
    ax.hlines(
        y,
        row[low95],
        row[high95],
        color="#0072B2",
        linewidth=2.6,
        label="95% pointwise interval" if labels else None,
    )
    ax.plot(row["mean_difference"], y, "o", color="#0072B2", markersize=6)


def online_figure(output):
    e8 = pd.read_csv(ROOT / "reports/e8_evidence/evidence_primary.csv")
    e9 = json.loads((ROOT / "results/sampling_v1/interaction_summary.json").read_text())
    if len(e8) != 2 or not e8.n_pairs.eq(320).all() or not e9["complete_design"]:
        raise ValueError(
            "The online interpretation figure requires the complete frozen evaluations"
        )
    fig, axes = plt.subplots(
        1, 2, figsize=(13.8, 4.7), gridspec_kw={"width_ratios": [1, 1.12]}
    )
    a = axes[0]
    for y, (_, row) in enumerate(e8.iterrows()):
        interval_pair(
            a, row, y, "ci95_low", "ci95_high", "ci975_low", "ci975_high", labels=y == 0
        )
    a.set_yticks(
        [0, 1],
        [
            "Posterior means\nminus counts",
            "Full Bayesian summaries\nminus posterior means",
        ],
    )
    a.set_ylim(1.7, -0.6)
    a.set_xlabel("Difference in cumulative pseudo-regret")
    a.set_title("E8: two primary evidence contrasts", fontweight="bold")
    a.axvline(0, color="#555555", ls="--", lw=1)
    a.grid(axis="x", alpha=0.2)
    a.legend(loc="lower left", fontsize=8, frameon=False)
    b = axes[1]
    points = [
        (x["sample_minus_direct"], x["ci_low"], x["ci_high"])
        for x in e9["secondary_per_k"]
    ]
    points.append((e9["estimate"], e9["ci_low"], e9["ci_high"]))
    for y, (mean, low, high) in enumerate(points):
        color = "#D55E00" if y == 2 else "#0072B2"
        b.hlines(y, low, high, color=color, lw=2.5)
        b.plot(mean, y, "o", color=color, markersize=6)
    b.set_yticks(
        [0, 1, 2],
        [
            "2 arms: sample minus direct\n(secondary)",
            "10 arms: sample minus direct\n(secondary)",
            "10-arm effect minus 2-arm effect\n(primary interaction)",
        ],
    )
    b.get_yticklabels()[-1].set_fontweight("bold")
    b.set_ylim(2.6, -0.6)
    b.set_xlabel("Difference in cumulative pseudo-regret")
    b.set_title("E9: sampling contrast (95% intervals)", fontweight="bold")
    b.axvline(0, color="#555555", ls="--", lw=1)
    b.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Fresh online follow-ups: representation and action selection", fontsize=15
    )
    fig.text(
        0.5,
        0.075,
        "Each episode has 100 pulls. E8: 80 paired tasks per cell, four cells equally weighted. E9: 100 paired tasks per arm-count stratum.",
        ha="center",
        fontsize=9,
    )
    fig.text(
        0.5,
        0.027,
        "Negative differences favor the first policy; a negative E9 interaction means sampling is more favorable at 10 arms than at 2 arms.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.13, 1, 0.92], w_pad=3)
    path = output / "followup_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def primitive_figure(output):
    primary = pd.read_csv(ROOT / "reports/primitive_v1/primary_effects.csv")
    risks = pd.read_csv(ROOT / "reports/primitive_v1/risk_summary.csv")
    if len(primary) != 2 or not primary.complete.eq(True).all():
        raise ValueError(
            "The elicitation figure requires both complete primary comparisons"
        )
    fig, axes = plt.subplots(
        1, 2, figsize=(13.8, 4.9), gridspec_kw={"width_ratios": [1, 1.12]}
    )
    a = axes[0]
    for y, (_, row) in enumerate(primary.iterrows()):
        interval_pair(
            a,
            row,
            y,
            "ci95_low",
            "ci95_high",
            "ci97_5_low",
            "ci97_5_high",
            labels=y == 0,
        )
    a.set_yticks(
        [0, 1],
        [
            "Next reward:\nbinary Choice minus Noul",
            "Latent best:\nNoul minus joint Choice",
        ],
    )
    a.set_ylim(1.7, -0.6)
    a.set_title("A. Prespecified elicitation contrasts", fontweight="bold")
    a.set_xlabel("Mean binary excess Brier risk difference")
    a.axvline(0, color="#555555", ls="--", lw=1)
    a.grid(axis="x", alpha=0.2)
    a.legend(loc="lower left", fontsize=8, frameon=False)
    b = axes[1]
    for primitive, color in [("choice", "#D55E00"), ("noul", "#0072B2")]:
        for representation, style, wording in [
            ("bayes", "-", "summaries"),
            ("oracle_probs_full", "--", "supplied probabilities + summaries"),
        ]:
            values = risks.loc[
                risks.target.eq("best_arm")
                & risks.primitive.eq(primitive)
                & risks.representation.eq(representation)
            ].sort_values("k")
            b.plot(
                values.k,
                values.mean_binary_excess_brier,
                marker="o",
                color=color,
                ls=style,
                label=f"{primitive.title()}: {wording}",
            )
    b.set_title("B. Latent-best controls: descriptive means", fontweight="bold")
    b.set(
        xlabel="Number of arms", ylabel="Mean binary excess Brier risk", ylim=(0, 0.15)
    )
    b.set_xticks([2, 5, 15])
    b.grid(alpha=0.2)
    b.legend(loc="upper right", fontsize=7.5, frameon=False)
    fig.suptitle(
        "E10: forecast fidelity depends on the elicitation interface", fontsize=15
    )
    fig.text(
        0.5,
        0.075,
        "90 fresh fixtures, 30 per arm-count stratum. Panel A uses full summaries and equal weight across strata. Panel B is secondary.",
        ha="center",
        fontsize=9,
    )
    fig.text(
        0.5,
        0.027,
        "Noul is the documented binary interface. Best-arm Nouls are separate, unnormalized marginal forecasts, not a coherent joint vector by construction.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.13, 1, 0.92], w_pad=3)
    path = output / "elicitation_controls.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(online_figure(args.output_dir))
    print(primitive_figure(args.output_dir))


if __name__ == "__main__":
    main()
