"""Inference unit, pairing, missingness and reproducibility checks."""

import json

import numpy as np
import pandas as pd
import pytest

from jevbandits.report import (
    bootstrap_mean,
    diagnostic_summaries,
    fixture_effect,
    generate_report,
    paired_effect,
    stratified_paired_effect,
)


def test_paired_bootstrap_preserves_shared_environment_noise():
    # Between-environment variation cancels exactly in a paired comparison.
    frame = pd.DataFrame(
        [
            {"episode_id": episode, "policy": policy, "regret": noise + shift}
            for episode, noise in enumerate([0, 1000, 100000])
            for policy, shift in [("a", 3), ("b", 1)]
        ]
    )
    effect = paired_effect(frame, "a", "b", "regret", ["episode_id"], samples=200)
    assert effect.mean_difference == effect.ci_low == effect.ci_high == 2
    assert effect.n_pairs == 3


def test_missing_policy_is_excluded_pairwise_and_reported():
    frame = pd.DataFrame(
        [
            {"episode_id": 0, "policy": "a", "regret": 3},
            {"episode_id": 0, "policy": "b", "regret": 1},
            {"episode_id": 1, "policy": "a", "regret": 100},
        ]
    )
    effect = paired_effect(frame, "a", "b", "regret", ["episode_id"])
    assert (effect.n_left, effect.n_right, effect.n_pairs) == (2, 1, 1)
    assert effect.mean_difference == 2
    assert np.isnan(effect.ci_low)


def test_duplicate_pull_rows_rejected_as_pseudoreplication():
    frame = pd.DataFrame(
        [
            {"episode_id": 0, "policy": "a", "regret": 1},
            {"episode_id": 0, "policy": "a", "regret": 2},
        ]
    )
    with pytest.raises(ValueError, match="Duplicate"):
        paired_effect(frame, "a", "b", "regret", ["episode_id"])


def test_pool_equally_weights_cells_not_episodes():
    frame = pd.DataFrame(
        [
            {"family": family, "episode_id": i, "policy": policy, "regret": shift}
            for family, count, difference in [("small", 2, 10), ("large", 20, 0)]
            for i in range(count)
            for policy, shift in [("a", difference), ("b", 0)]
        ]
    )
    effect = stratified_paired_effect(
        frame, "a", "b", "regret", ["episode_id"], ["family"], samples=200
    )
    assert effect["mean_difference"] == effect["ci_low"] == effect["ci_high"] == 5
    assert effect["n_pairs"] == 22


def test_missing_entire_cell_does_not_change_estimand_silently():
    frame = pd.DataFrame(
        [
            {"family": "x", "episode_id": 0, "policy": "a", "regret": 2},
            {"family": "x", "episode_id": 0, "policy": "b", "regret": 0},
            {"family": "y", "episode_id": 0, "policy": "a", "regret": 9},
        ]
    )
    effect = stratified_paired_effect(
        frame, "a", "b", "regret", ["episode_id"], ["family"], samples=100
    )
    assert effect["status"] == "missing_paired_cell"
    assert np.isnan(effect["mean_difference"])


def test_query_repetition_does_not_inflate_fixture_sample_size():
    frame = pd.DataFrame(
        [
            {"fixture_id": fixture, "condition": condition, "loss": shift + fixture}
            for fixture in range(3)
            for condition, shift in [("a", 1), ("b", 0)]
            for repeat in range(100)
        ]
    )
    effect = fixture_effect(frame, "a", "b", "loss", samples=100)
    assert effect.n_pairs == 3
    assert effect.mean_difference == effect.ci_low == effect.ci_high == 1
    assert effect.unit == "underlying fixture"


def test_bootstrap_reproducibility_and_bad_values():
    assert bootstrap_mean([1, 2, 8, 10], seed=6, samples=300) == bootstrap_mean(
        [1, 2, 8, 10], seed=6, samples=300
    )
    with pytest.raises(ValueError, match="finite"):
        bootstrap_mean([1, np.nan])


def test_diagnostic_pairing_uses_canonical_actions_and_fixture_clusters():
    rows = [
        {
            "experiment": "e1_horizon",
            "fixture_id": str(fixture),
            "horizon": 2,
            "policy": policy,
            "label_order": order,
            "repeat": repeat,
            "canonical_action": order,
            "exact_loss": float(policy == "a"),
            "optimal_agreement": float(policy == "b"),
        }
        for fixture in range(3)
        for policy in ["a", "b"]
        for order in [0, 1]
        for repeat in range(5)
    ]
    summary, effects, labels = diagnostic_summaries(pd.DataFrame(rows), samples=100)
    assert set(summary.n_fixtures) == {3}
    assert set(effects.n_fixtures) == {3}
    assert set(labels["mean"]) == {1.0}
    loss = effects.loc[effects.metric == "exact_loss"].iloc[0]
    assert loss.mean_difference == 1


def test_report_writes_offline_artifacts_and_excludes_incomplete(tmp_path):
    rows = []
    for episode in range(3):
        for policy, regret in [("counts_direct", 4), ("bayes_direct", 2), ("ts", 1)]:
            rows.append(
                {
                    "experiment": "e4_scaling",
                    "family": "prior",
                    "k": 2,
                    "horizon": 10,
                    "episode_id": str(episode),
                    "policy": policy,
                    "pseudo_regret": regret + episode,
                    "reward": 10 - regret,
                    "exact_loss": None,
                    "completed": True,
                }
            )
    rows.append({**rows[0], "episode_id": "incomplete", "completed": False})
    (tmp_path / "episodes.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    report = generate_report(tmp_path, bootstrap_samples=100)
    assert report.exists()
    assert "Completed episode rows: 9" in report.read_text()
    assert "2.000 lower cumulative pseudo-regret" in report.read_text()
    assert (report.parent / "e4_scaling_regret.png").exists()
    metadata = json.loads((report.parent / "analysis_manifest.json").read_text())
    assert metadata["incomplete_episode_rows"] == 1
    assert metadata["inputs"]["episodes.jsonl"]
