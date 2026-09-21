"""Posthoc repeat-variability decomposition for E7; no API calls."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .report import bootstrap_mean, markdown_table


def decompose(first, second, reference):
    a, b, q = (np.asarray(x, dtype=float) for x in (first, second, reference))
    if a.shape != b.shape or a.shape != q.shape:
        raise ValueError("Repeat/reference shapes differ")
    raw = float(((a - q) @ (a - q) + (b - q) @ (b - q)) / 2)
    variability = float((a - b) @ (a - b) / 2)
    cross = float((a - q) @ (b - q))
    if abs(raw - variability - cross) > 1e-12:
        raise ValueError("Repeat decomposition identity failed")
    return {
        "raw_excess_brier": raw,
        "repeat_variability": variability,
        "cross_repeat_error_product": cross,
    }


def generate(run_dir, output_dir):
    source, output = Path(run_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_json(source / "forecasts.jsonl", lines=True)
    keys = ["target", "k", "fixture_id", "representation", "arm"]
    paired = []
    for key, g in data.groupby(keys, dropna=False):
        if len(g) != 2 or set(g.repeat) != {0, 1}:
            raise ValueError("Exactly two complete API repeats required")
        g = g.sort_values("repeat")
        if not np.allclose(
            g.iloc[0].reference, g.iloc[1].reference, atol=1e-12, rtol=0
        ):
            raise ValueError("Reference changed between repeats")
        paired.append(
            dict(zip(keys, key))
            | decompose(g.iloc[0].prediction, g.iloc[1].prediction, g.iloc[0].reference)
        )
    metrics = ["raw_excess_brier", "repeat_variability", "cross_repeat_error_product"]
    units = pd.DataFrame(paired).groupby(keys[:-1], as_index=False)[metrics].mean()
    rows = []
    for (target, k, representation), g in units.groupby(
        ["target", "k", "representation"]
    ):
        for metric in metrics:
            mean, low, high = bootstrap_mean(g[metric])
            rows.append(
                {
                    "target": target,
                    "k": int(k),
                    "representation": representation,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n_fixtures": len(g),
                }
            )
    summary = pd.DataFrame(rows)
    units.to_csv(output / "fixture_repeat_decomposition.csv", index=False)
    summary.to_csv(output / "repeat_decomposition.csv", index=False)
    # Descriptive aggregate weights each arm count equally, as in the main design.
    overview = summary.pivot(
        index=["target", "k", "representation"], columns="metric", values="mean"
    ).reset_index()
    overview = overview.groupby(["target", "representation"], as_index=False)[
        metrics
    ].mean()
    overview["variability_fraction"] = (
        overview.repeat_variability / overview.raw_excess_brier
    )
    overview.to_csv(output / "overview.csv", index=False)
    text = [
        "# E7 repeat variability versus reference error",
        "This posthoc numerical audit uses the two independently requested repeats of each identical forecast question. It does not query the API or infer an internal sampling mechanism.",
        markdown_table(overview, digits=6),
        "For vectors $p_1,p_2$ and analytic reference $q$, the exact arithmetic identity is",
        "$$\frac{\\|p_1-q\\|^2+\\|p_2-q\\|^2}{2}=\frac{\\|p_1-p_2\\|^2}{2}+(p_1-q)^\top(p_2-q).$$",
        "Under independent, stationary repeat errors conditional on a fixed question, the first right-hand term estimates summed response variance and the second estimates squared deviation of the mean response from the reference. The cross-product estimate may be negative in a finite sample and is not clipped. If repeat errors are correlated, its expectation also includes that covariance. Repeat variability may include service or batch effects; it is not automatically Monte Carlo sampling inside Jev.",
        "Arms are averaged within each fixture before 10,000-resample bootstrap intervals. The overview equally weights arm-count cells and is descriptive. The decomposition does not turn repeated API calls into independent task samples. Near-zero repeat variability cannot rule out shared service drift or systematic batch effects.",
        "The observed Noul probabilities lie on a .01 grid. Nearest rounding of an otherwise exact scalar reference to that grid would change it by at most .005; this alone does not explain larger errors. No particular internal quantization or sampling procedure is assumed.",
    ]
    (output / "report.md").write_text("\n\n".join(text) + "\n")
    return output / "report.md"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="artifacts/forecast_v1")
    parser.add_argument("--output-dir", default="reports/forecast_v1/repeats")
    args = parser.parse_args()
    print(generate(args.run_dir, args.output_dir))


if __name__ == "__main__":
    main()
