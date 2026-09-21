"""Precollection reporting checks; synthetic records only, no API access."""

import json

import numpy as np
import pandas as pd
import pytest

from jevbandits.followup_reports import (
    E8_CELLS,
    E8_CONTRASTS,
    FORMATS,
    advice_tables,
    evidence_primary,
    report_advice,
    report_evidence,
)
from jevbandits.report import _seed


def advice_records(n=3):
    rows = []
    for cohort in ("random", "targeted"):
        for fixture in range(n):
            for labels in ("original", "letters"):
                for horizon in (1, 2, 10):
                    for fmt in FORMATS:
                        for order in (0, 1):
                            for repeat in (0, 1):
                                adherence = float(fmt != "nested")
                                rows.append(
                                    {
                                        "experiment": "e6_advice_binding",
                                        "cohort": cohort,
                                        "fixture_id": f"{cohort}:{fixture}",
                                        "format": fmt,
                                        "label_scheme": labels,
                                        "horizon": horizon,
                                        "label_order": order,
                                        "repeat": repeat,
                                        "decision_id": f"{cohort}:{fixture}:{fmt}:{labels}:{horizon}:{order}:{repeat}",
                                        "advice_present": fmt != "dp_values",
                                        "advice_adherence": adherence,
                                        "distribution_advice_adherence": adherence,
                                        "optimal_agreement": 1.0,
                                        "exact_loss": 0.0,
                                        "distribution_exact_loss": 0.0,
                                        "posterior_mean_greedy": 1.0,
                                        "prediction_entropy": 0.3,
                                        "choice_mismatch": 0.0,
                                    }
                                )
    return pd.DataFrame(rows)


def evidence_records(n=80, variable=False):
    rows = []
    for cell, (family, k) in enumerate(E8_CELLS):
        for episode in range(n):
            delta = episode / max(1, n - 1) if variable else cell + 1
            for policy, shift in [
                ("counts_direct", 0),
                ("means_direct", delta),
                ("bayes_direct", delta + 0.5),
            ]:
                rows.append(
                    {
                        "experiment": "e8_evidence_online",
                        "family": family,
                        "k": k,
                        "horizon": 100,
                        "episode_id": f"{family}:{k}:{episode}",
                        "policy": policy,
                        "pseudo_regret": episode * 10 + shift,
                        "reward": 50,
                        "completed": True,
                    }
                )
    return pd.DataFrame(rows)


def test_e6_adherence_excludes_dp_without_dropping_optimality_control():
    complete, summary, contrasts, interactions = advice_tables(
        advice_records(), samples=100
    )
    dp = summary.loc[summary.format == "dp_values"]
    assert set(dp.metric) == {
        "optimal_agreement",
        "exact_loss",
        "distribution_exact_loss",
    }
    assert not (
        (contrasts.left == "dp_values") & (contrasts.metric == "advice_adherence")
    ).any()
    assert not (
        (interactions.format == "dp_values")
        & (interactions.metric == "advice_adherence")
    ).any()
    assert set(complete.complete_fixtures) == {3}
    assert set(complete.missing_fixtures) == {97}
    assert set(summary.n_fixtures) == {3}
    assert set(summary.cohort) == {"random", "targeted"}


def test_e6_exactly_twelve_primary_descriptive_factor_specific_contrasts():
    _, _, contrasts, _ = advice_tables(advice_records(), samples=100)
    primary = contrasts.loc[contrasts.comparison_role == "primary_descriptive"]
    assert len(primary) == 3 * 2 * 2
    assert set(primary.left) == {"path_explicit", "inline", "per_option"}
    assert set(primary.right) == {"nested"}
    assert set(primary.horizon) == {10}
    assert set(primary.metric) == {"advice_adherence"}
    assert set(primary.mean_difference) == {1.0}
    assert set(primary.n_pairs) == {3}


def test_e6_incomplete_four_query_fixture_is_disclosed_and_excluded():
    data = advice_records()
    removed = data.iloc[0]
    complete, summary, _, _ = advice_tables(data.iloc[1:], samples=100)
    row = complete.loc[
        (complete.cohort == removed.cohort)
        & (complete.label_scheme == removed.label_scheme)
        & (complete.horizon == removed.horizon)
        & (complete.format == removed.format)
    ].iloc[0]
    assert row.observed_queries == 11
    assert row.complete_fixtures == 2
    assert row.incomplete_observed_fixtures == 1
    assert row.missing_queries == 389
    result = summary.loc[
        (summary.cohort == removed.cohort)
        & (summary.label_scheme == removed.label_scheme)
        & (summary.horizon == removed.horizon)
        & (summary.format == removed.format)
    ]
    assert set(result.n_fixtures) == {2}


def test_e6_duplicate_and_invalid_advice_flag_rejected():
    data = advice_records(1)
    with pytest.raises(ValueError, match="Duplicate"):
        advice_tables(pd.concat([data, data.iloc[:1]]), samples=100)
    data.loc[data.format == "dp_values", "advice_present"] = True
    with pytest.raises(ValueError, match="Advice-present"):
        advice_tables(data, samples=100)


def test_e8_two_primary_contrasts_equal_cell_weight_and_paired_noise():
    primary, complete, per_cell = evidence_primary(evidence_records(), samples=300)
    assert list(zip(primary.left, primary.right)) == list(E8_CONTRASTS)
    assert set(primary.status) == {"complete"}
    assert set(primary.n_pairs) == {320}
    assert primary.mean_difference.tolist() == [2.5, 0.5]
    assert primary.ci95_low.tolist() == primary.ci975_low.tolist() == [2.5, 0.5]
    assert set(complete.n_pairs) == {80}
    assert len(per_cell) == 8


def test_e8_bonferroni_quantiles_match_declared_percentiles():
    data = evidence_records(5, variable=True)
    primary, _, _ = evidence_primary(data, samples=500)
    rng = np.random.default_rng(
        _seed("e8_evidence_online:means_direct:counts_direct:primary")
    )
    values = np.linspace(0, 1, 5)
    bootstrap = sum(
        values[rng.integers(0, 5, (500, 5))].mean(axis=1) / 4 for _ in range(4)
    )
    expected = np.quantile(bootstrap, [0.025, 0.975, 0.0125, 0.9875])
    result = primary.iloc[0]
    np.testing.assert_allclose(
        [result.ci95_low, result.ci95_high, result.ci975_low, result.ci975_high],
        expected,
    )
    assert result.ci975_low <= result.ci95_low
    assert result.ci975_high >= result.ci95_high
    assert result.status == "incomplete_descriptive_only"


def test_e8_missing_entire_cell_produces_no_pooled_estimate():
    data = evidence_records()
    data = data.loc[~((data.family == "prior") & (data.k == 10))]
    primary, complete, _ = evidence_primary(data, samples=100)
    assert set(primary.status) == {"missing_cell_no_pooled_estimate"}
    assert primary.mean_difference.isna().all()
    assert set(primary.missing_cells) == {"prior/K10"}
    assert set(
        complete.loc[complete.family.eq("prior") & complete.k.eq(10), "missing_pairs"]
    ) == {80}


def test_e8_incomplete_cells_keep_equal_weight_instead_of_episode_weight():
    data = evidence_records()
    keep = ~((data.family == "close") & (data.k == 3)) | data.episode_id.eq("close:3:0")
    primary, _, _ = evidence_primary(data.loc[keep], samples=100)
    assert primary.iloc[0].mean_difference == 2.5
    assert primary.iloc[0].status == "incomplete_descriptive_only"
    assert np.isnan(primary.iloc[0].ci95_low)


def test_e8_one_missing_policy_episode_reports_left_right_and_pair_counts():
    data = evidence_records()
    missing = (
        (data.family == "close")
        & (data.k == 3)
        & (data.policy == "means_direct")
        & (data.episode_id == "close:3:0")
    )
    primary, complete, _ = evidence_primary(data.loc[~missing], samples=100)
    assert set(primary.status) == {"incomplete_descriptive_only"}
    assert set(primary.n_pairs) == {319}
    cell = complete.loc[(complete.family == "close") & (complete.k == 3)]
    assert cell.missing_left.tolist() == [1, 0]
    assert cell.missing_right.tolist() == [0, 1]
    assert cell.missing_pairs.tolist() == [1, 1]


def test_e8_duplicate_and_unexpected_cells_rejected():
    data = evidence_records(2)
    with pytest.raises(ValueError, match="Duplicate"):
        evidence_primary(pd.concat([data, data.iloc[:1]]), samples=100)
    data.loc[0, "k"] = 15
    with pytest.raises(ValueError, match="Unexpected"):
        evidence_primary(data, samples=100)


def test_precollection_reports_show_missingness_without_network(tmp_path, monkeypatch):
    import httpx

    def forbidden(*args, **kwargs):
        raise AssertionError("No API access is allowed during reporting")

    monkeypatch.setattr(httpx.AsyncClient, "post", forbidden)
    a = report_advice(tmp_path, tmp_path / "advice", samples=100)
    e = report_evidence(tmp_path, tmp_path / "evidence", samples=100)
    assert "Recorded decisions: 0 of 28,800" in a.read_text()
    assert "missing_cell_no_pooled_estimate" in e.read_text()
    assert len(pd.read_csv(e.parent / "evidence_primary.csv")) == 2


def test_mock_data_reports_write_figures_tables_and_input_hashes(tmp_path):
    a_dir, e_dir = tmp_path / "a", tmp_path / "e"
    a_dir.mkdir()
    e_dir.mkdir()
    a_dir.joinpath("diagnostics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in advice_records(2).to_dict("records"))
    )
    e_dir.joinpath("episodes.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in evidence_records(4, variable=True).to_dict("records")
        )
    )
    a = report_advice(a_dir, samples=100)
    e = report_evidence(e_dir, samples=100)
    assert (a.parent / "advice_h10.png").exists()
    assert (e.parent / "evidence_primary.png").exists()
    metadata = json.loads((e.parent / "analysis_manifest.json").read_text())
    assert metadata["inputs"]["episodes.jsonl"]
    assert "97.5%" in e.read_text()
    summary = pd.read_csv(a.parent / "advice_summary.csv")
    assert not (
        (summary.format == "dp_values") & (summary.metric == "advice_adherence")
    ).any()
