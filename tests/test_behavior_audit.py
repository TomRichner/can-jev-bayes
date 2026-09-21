"""Toy-history and independent-unit checks for the offline behavior audit."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from jevbandits.baselines import exact_q
from jevbandits.behavior_audit import (
    audit_episode,
    generate_behavior_report,
    phase_for_turn,
    state_behavior,
    summarize_phases,
)


def episode(actions=(0, 0, 1, 1, 0), rewards=(1, 1, 0, 1, 0), *, experiment="toy"):
    return {
        "experiment": experiment,
        "family": "prior",
        "k": 2,
        "horizon": len(actions),
        "policy": "counts_direct",
        "episode_id": "example",
        "completed": True,
        "reward": sum(rewards),
        "trace": [
            {"turn": i, "action": a, "reward": r}
            for i, (a, r) in enumerate(zip(actions, rewards, strict=True), 1)
        ],
    }


def test_public_count_reconstruction_precedes_reward_observation():
    result = audit_episode(episode()).set_index("phase")
    assert result.loc["early", "posterior_mean_greedy"] == 1
    # After two successes on arm 0, pulling the unseen arm 1 is non-greedy.
    assert result.loc["middle", "strict_non_greedy"] == pytest.approx(2 / 3)
    assert result.loc["middle", "unseen_arm"] == pytest.approx(1 / 3)
    assert result.loc["middle", "lower_mean_higher_sd"] == pytest.approx(2 / 3)
    assert result.loc["late", "posterior_mean_greedy"] == 1
    assert result.n_turns.tolist() == [1, 3, 1]


def test_phase_boundaries_are_exact_and_terminal_singleton_is_late():
    assert [phase_for_turn(t, 10) for t in [1, 2, 3, 8, 9, 10]] == [
        "early",
        "early",
        "middle",
        "middle",
        "late",
        "late",
    ]
    assert phase_for_turn(1, 1) == "late"
    with pytest.raises(ValueError):
        phase_for_turn(0, 5)


def test_equal_mean_uncertainty_selection_is_separate_from_non_greedy():
    result = state_behavior([9, 0], [9, 0], 1, 19, 20)
    assert result["posterior_mean_greedy"] == 1
    assert result["strict_non_greedy"] == 0
    assert result["equal_mean_higher_sd"] == 1
    assert result["unequal_sd_greedy_tie_available"] == 1
    assert result["lower_mean_higher_sd"] == 0


def test_normative_crossover_is_useful_at_two_pulls_inferior_at_terminal():
    q2 = exact_q((10, 0), (8, 0), 2)
    two = state_behavior([10, 0], [8, 0], 1, 19, 20, q2)
    assert two["strict_exploration_required"] == 1
    assert two["useful_non_greedy"] == 1
    assert two["inferior_non_greedy"] == 0
    assert two["exact_loss"] == 0
    one = state_behavior([10, 0], [8, 0], 1, 20, 20, [0.55, 0.5])
    assert one["strict_exploration_required"] == 0
    assert one["inferior_non_greedy"] == 1
    assert one["useful_non_greedy"] == 0
    assert one["knowledge_gradient_agreement"] == 0
    assert one["exact_loss"] == pytest.approx(0.05)


def test_initial_equal_evidence_index_ties_are_recorded():
    result = state_behavior([0, 0, 0], [0, 0, 0], 2, 1, 10)
    assert result["bayes_ucb_agreement"] == 1
    assert result["knowledge_gradient_agreement"] == 1
    assert result["bayes_ucb_top_set_size"] == 3
    assert result["knowledge_gradient_top_set_size"] == 3


def test_e3_recomputed_loss_matches_recorded_trace_and_detects_corruption():
    record = episode(
        actions=tuple([0] * 18 + [1, 1]),
        rewards=tuple([1] * 10 + [0] * 8 + [1, 1]),
        experiment="e3_exact_online",
    )
    s, f = np.zeros(2, dtype=int), np.zeros(2, dtype=int)
    for point in record["trace"]:
        q = exact_q(tuple(s), tuple(f), 21 - point["turn"])
        point["exact_loss"] = float(q.max() - q[point["action"]])
        s[point["action"]] += point["reward"]
        f[point["action"]] += 1 - point["reward"]
    rows = audit_episode(record).set_index("phase")
    assert rows.loc["late", "useful_non_greedy"] == 0.25
    corrupted = copy.deepcopy(record)
    corrupted["trace"][-1]["exact_loss"] += 0.1
    with pytest.raises(ValueError, match="exact loss"):
        audit_episode(corrupted)
    record["exact_loss"] = sum(p["exact_loss"] for p in record["trace"]) + 0.1
    with pytest.raises(ValueError, match="exact-loss total"):
        audit_episode(record)


def test_long_horizon_two_arm_scores_null_original_fields_and_tags_working_prior():
    record = episode(actions=(0,) * 100, rewards=(1,) * 100, experiment="e4_scaling")
    record["family"] = "clear"
    record["exact_loss"] = None
    for point in record["trace"]:
        point.update(exact_loss=None, optimal_agreement=None)
    result = audit_episode(record)
    assert set(result.normative_reference) == {"posthoc_exact_two_arm_h100"}
    assert set(result.normative_interpretation) == {"working_prior_stress_test"}
    assert result.exact_loss.notna().all()
    record["family"] = "prior"
    assert set(audit_episode(record).normative_interpretation) == {"matched_prior"}
    # pandas promotes mixed numeric/null episode totals to NaN while preserving nested nulls.
    record["exact_loss"] = np.nan
    assert audit_episode(record).exact_loss.notna().all()


def test_long_horizon_larger_arm_counts_have_no_exact_reference():
    record = episode(actions=(0,) * 100, rewards=(1,) * 100, experiment="e4_scaling")
    record["k"] = 3
    result = audit_episode(record)
    assert set(result.normative_reference) == {"unavailable"}
    assert "exact_loss" not in result


@pytest.mark.parametrize(
    "field,value",
    [
        ("action", 2),
        ("action", -1),
        ("action", True),
        ("reward", 2),
        ("reward", 0.5),
        ("turn", 3),
    ],
)
def test_invalid_trace_values_rejected(field, value):
    record = episode()
    record["trace"][0][field] = value
    with pytest.raises(ValueError):
        audit_episode(record)


def test_bad_reward_total_and_recorded_greedy_flag_rejected():
    record = episode()
    record["reward"] = 100
    with pytest.raises(ValueError, match="reward total"):
        audit_episode(record)
    record = episode()
    record["trace"][0]["posterior_mean_greedy"] = False
    with pytest.raises(ValueError, match="greedy flag"):
        audit_episode(record)


def test_phase_summary_resamples_episodes_not_turns():
    first = audit_episode(episode())
    second = first.assign(episode_id="second", posterior_mean_greedy=0.0, n_turns=1000)
    summary = summarize_phases(
        pd.concat([first, second], ignore_index=True), samples=100
    )
    early = summary.loc[
        (summary.phase == "early") & (summary.metric == "posterior_mean_greedy")
    ].iloc[0]
    assert early.n_episodes == 2
    assert early["mean"] == 0.5
    assert early.n_recorded_turns == 1001
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_phases(pd.concat([first, first]), samples=100)


def test_hidden_means_do_not_affect_replayed_behavior():
    one = episode()
    two = copy.deepcopy(one)
    one["theta"] = [1.0, 0.0]
    two["theta"] = [0.0, 1.0]
    pd.testing.assert_frame_equal(audit_episode(one), audit_episode(two))


def test_offline_report_has_episode_units_filter_and_figures(tmp_path):
    first, second = episode(), episode()
    second.update(experiment="unrequested", episode_id="second")
    (tmp_path / "episodes.jsonl").write_text(
        "\n".join(json.dumps(x) for x in [first, second])
    )
    path = generate_behavior_report(tmp_path, experiments=["toy"], samples=100)
    assert "Completed episode-policy records: 1" in path.read_text()
    assert "do not identify Jev's internal algorithm" in path.read_text()
    assert (path.parent / "toy_prior_k2_behavior.png").exists()
    summary = pd.read_csv(path.parent / "phase_summary.csv")
    assert set(summary.n_episodes) == {1}
