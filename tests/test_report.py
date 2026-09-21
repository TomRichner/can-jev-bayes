"""Inference unit, pairing, missingness and reproducibility checks."""

import json

import numpy as np
import pandas as pd
import pytest

from jevbandits.report import (
    bootstrap_mean,
    classify_diagnostic_states,
    diagnostic_summaries,
    episode_effects,
    fixture_effect,
    generate_report,
    generate_two_arm_audit_report,
    horizon_contrasts,
    paired_effect,
    random_fixture_error_intervals,
    state_regime_summaries,
    stratified_paired_effect,
    trajectory_summaries,
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


def test_secondary_baselines_remain_distinct_from_primary_ts_comparison():
    frame = pd.DataFrame(
        [
            {
                "experiment": "e4_scaling",
                "family": "prior",
                "k": 2,
                "horizon": 10,
                "episode_id": str(i),
                "policy": policy,
                "pseudo_regret": regret,
                "reward": 10 - regret,
            }
            for i in range(3)
            for policy, regret in [
                ("counts_direct", 2),
                ("ts", 3),
                ("greedy", 1),
                ("bayes_ucb", 2),
                ("knowledge_gradient", 1),
                ("ids", 1),
                ("ucb1", 4),
            ]
        ]
    )
    effects, pooled = episode_effects(frame, samples=100)
    assert set(effects.loc[effects.right == "ts", "comparison_role"]) == {
        "primary_vs_ts"
    }
    assert set(effects.loc[effects.right != "ts", "comparison_role"]) == {
        "secondary_baseline"
    }
    greedy = pooled.loc[
        (pooled.right == "greedy") & (pooled.metric == "pseudo_regret")
    ].iloc[0]
    assert greedy.mean_difference == 1


def state_panel():
    return pd.DataFrame(
        [
            {
                "experiment": "e1_horizon",
                "fixture_id": fixture,
                "policy": "counts_explore",
                "label_order": order,
                "repeat": repeat,
                "horizon": horizon,
                "successes": s,
                "failures": f,
                "q_values": q,
                "canonical_action": action,
                "probabilities": p,
                "exact_loss": max(q) - q[action],
                "optimal_agreement": float(q[action] == max(q)),
            }
            for fixture, s, f in [
                ("crossover", [10, 0], [8, 0]),
                ("tied_means", [9, 0], [9, 0]),
            ]
            for horizon in [1, 10]
            for q, action, p in [
                (
                    ([0.55, 0.5] if fixture == "crossover" else [0.5, 0.5])
                    if horizon == 1
                    else [5.5, 5.6],
                    0 if horizon == 1 else 1,
                    [0.8, 0.2] if horizon == 1 else [0.2, 0.8],
                )
            ]
            for order in [0, 1]
            for repeat in range(5)
        ]
    )


def test_exploration_requirement_respects_posterior_mean_tie_sets():
    classified = classify_diagnostic_states(state_panel())
    assert set(classified.loc[classified.horizon == 1, "state_regime"]) == {"terminal"}
    high = classified.loc[classified.horizon == 10]
    assert set(high.loc[high.fixture_id == "crossover", "state_regime"]) == {
        "exploration_required"
    }
    assert set(high.loc[high.fixture_id == "tied_means", "state_regime"]) == {
        "greedy_can_be_optimal"
    }
    assert set(high.loc[high.fixture_id == "tied_means", "non_greedy_choice"]) == {0.0}
    summary = state_regime_summaries(state_panel(), samples=100)
    assert set(
        summary.loc[summary.state_regime == "exploration_required", "n_fixtures"]
    ) == {1}


def test_horizon_contrasts_pair_fixtures_without_counting_repeated_calls():
    result = horizon_contrasts(state_panel(), samples=100)
    target = result.loc[
        (result.stratum == "high_horizon_requires_exploration")
        & (result.metric == "non_greedy_choice")
    ].iloc[0]
    assert target.n_fixtures == 1
    assert target.low_mean == 0
    assert target.high_mean == target.mean_difference == 1
    assert np.isnan(target.ci_low)
    probability = result.loc[
        (result.stratum == "high_horizon_requires_exploration")
        & (result.metric == "non_greedy_probability")
    ].iloc[0]
    assert probability.mean_difference == pytest.approx(0.6)
    assert set(result.analysis_role) == {"post_hoc_secondary"}


def test_diagnostic_only_report_includes_conditional_and_horizon_outputs(tmp_path):
    frame = state_panel()
    frame["decision_id"] = [f"decision-{i}" for i in range(len(frame))]
    (tmp_path / "diagnostics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in frame.to_dict("records"))
    )
    path = generate_report(tmp_path, bootstrap_samples=100)
    text = path.read_text()
    assert "Secondary analysis: when exploration is valuable" in text
    assert "Secondary analysis: paired horizon response" in text
    assert "post hoc secondary" in text
    assert (path.parent / "state_regime_summary.csv").exists()
    assert (path.parent / "horizon_contrasts.csv").exists()


def test_zero_fixture_errors_have_nondegenerate_exact_binomial_bounds():
    frame = pd.DataFrame(
        [
            {
                "experiment": "e2_assistance",
                "horizon": 10,
                "policy": "dp_values_explore",
                "fixture_id": str(i),
                "label_order": order,
                "repeat": repeat,
                "optimal_agreement": True,
            }
            for i in range(100)
            for order in [0, 1]
            for repeat in [0, 1]
        ]
    )
    row = random_fixture_error_intervals(frame).iloc[0]
    assert row.n_random_fixtures == 100
    assert row.n_error_fixtures == 0
    assert row.exact_95_ci_low == 0
    assert row.exact_95_ci_high == pytest.approx(1 - 0.025**0.01)
    assert row.one_sided_95_upper == pytest.approx(1 - 0.05**0.01)
    # One failure anywhere in a fixture yields one event, not one event per call.
    frame.loc[0, "optimal_agreement"] = False
    row = random_fixture_error_intervals(frame).iloc[0]
    assert row.n_error_fixtures == 1
    assert row.any_error_rate == 0.01
    assert random_fixture_error_intervals(frame.assign(experiment="e1_horizon")).empty


def test_incomplete_fixture_protocol_is_excluded_from_error_bounds():
    frame = pd.DataFrame(
        [
            {
                "experiment": "e2_assistance",
                "horizon": 1,
                "policy": "counts_explore",
                "fixture_id": str(i),
                "label_order": order,
                "repeat": repeat,
                "optimal_agreement": True,
            }
            for i in range(2)
            for order in [0, 1]
            for repeat in [0, 1]
        ]
    )
    row = random_fixture_error_intervals(frame.iloc[1:]).iloc[0]
    assert row.n_random_fixtures == 1
    assert row.n_incomplete_fixtures_excluded == 1


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


def test_trajectories_accumulate_within_episode_before_averaging():
    frame = pd.DataFrame(
        [
            {
                "experiment": "e4_scaling",
                "family": "prior",
                "k": 2,
                "horizon": 2,
                "policy": "ts",
                "trace": [
                    {
                        "turn": 1,
                        "reward": reward,
                        "pseudo_regret": regret,
                        "posterior_mean_greedy": True,
                    },
                    {
                        "turn": 2,
                        "reward": reward,
                        "pseudo_regret": regret,
                        "posterior_mean_greedy": False,
                    },
                ],
            }
            for reward, regret in [(1, 0.1), (0, 0.3)]
        ]
    )
    result = trajectory_summaries(frame)
    assert result.n_episodes.tolist() == [2, 2]
    assert result.mean_cumulative_pseudo_regret.tolist() == pytest.approx([0.2, 0.4])
    assert result.mean_cumulative_reward.tolist() == [0.5, 1.0]
    assert result.non_greedy_fraction.tolist() == [0.0, 1.0]


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
                    "choice_mismatch_fraction": 0.1 if policy != "ts" else None,
                    "completed": True,
                }
            )
    rows.append({**rows[0], "episode_id": "incomplete", "completed": False})
    (tmp_path / "episodes.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    report = generate_report(tmp_path, bootstrap_samples=100)
    assert report.exists()
    assert "Completed episode rows: 9" in report.read_text()
    assert "2.000 lower cumulative pseudo-regret" in report.read_text()
    assert "Backend choice versus probability argmax" in report.read_text()
    summary = pd.read_csv(report.parent / "episode_summary.csv")
    mismatch = summary.loc[summary.metric == "choice_mismatch_fraction"]
    assert set(mismatch.policy) == {"counts_direct", "bayes_direct"}
    assert mismatch["mean"].tolist() == pytest.approx([0.1, 0.1])
    assert (report.parent / "e4_scaling_regret.png").exists()
    metadata = json.loads((report.parent / "analysis_manifest.json").read_text())
    assert metadata["incomplete_episode_rows"] == 1
    assert metadata["inputs"]["episodes.jsonl"]


def test_additional_index_episodes_loaded_hashed_and_compared(tmp_path):
    shared = {
        "experiment": "e3_exact_online",
        "family": "prior",
        "k": 2,
        "horizon": 20,
        "episode_id": "fixture",
        "completed": True,
        "reward": 12,
        "pseudo_regret": 1.0,
    }
    (tmp_path / "episodes.jsonl").write_text(
        json.dumps({**shared, "policy": "counts_direct"}) + "\n"
    )
    (tmp_path / "episodes_index.jsonl").write_text(
        json.dumps({**shared, "policy": "finite_ap_index"}) + "\n"
    )
    report = generate_report(
        tmp_path,
        bootstrap_samples=100,
        experiment_filter=["e3_exact_online"],
        interim=True,
    )
    assert "interim report" in report.read_text()
    manifest = json.loads((report.parent / "analysis_manifest.json").read_text())
    assert "episodes_index.jsonl" in manifest["inputs"]
    assert manifest["experiment_filter"] == ["e3_exact_online"]
    effects = pd.read_csv(report.parent / "paired_effects.csv")
    assert set(effects.right) == {"finite_ap_index"}
    assert set(effects.comparison_role) == {"secondary_baseline"}


def test_duplicate_additional_episode_policy_is_rejected(tmp_path):
    row = {"episode_id": "fixture", "policy": "finite_ap_index"}
    for name in ["episodes.jsonl", "episodes_index.jsonl"]:
        (tmp_path / name).write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Duplicate episode/policy"):
        generate_report(tmp_path, bootstrap_samples=100)


def test_long_horizon_exact_audit_is_separate_and_counts_tasks_once(tmp_path):
    rows = [
        {
            "experiment": "e4_scaling",
            "family": "prior",
            "k": 2,
            "horizon": 100,
            "episode_id": str(i),
            "policy": policy,
            "reward": 60,
            "pseudo_regret": i + loss,
            "exact_loss": loss,
            "optimal_agreement": 1 - loss / 100,
        }
        for i in range(3)
        for policy, loss in [("counts_direct", 1.0), ("exact_h100", 0.0)]
    ]
    (tmp_path / "two_arm_bellman_audit.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows)
    )
    path = generate_two_arm_audit_report(tmp_path, bootstrap_samples=100)
    metadata = json.loads((path.parent / "analysis_manifest.json").read_text())
    assert metadata["n_episode_policy_rows"] == 6
    assert metadata["n_unique_tasks"] == 3
    assert "not pooled" in metadata["policy"]
    assert (path.parent / "exact_loss_by_family.png").exists()
    assert not (tmp_path / "episodes.jsonl").exists()
    rows[0]["k"] = 3
    (tmp_path / "two_arm_bellman_audit.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows)
    )
    with pytest.raises(ValueError, match="two arms"):
        generate_two_arm_audit_report(tmp_path, bootstrap_samples=100)
