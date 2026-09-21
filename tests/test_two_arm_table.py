"""Validate compact DP against independent recursive and analytic values."""

from math import comb

import numpy as np
import pytest

from jevbandits.baselines import exact_q
from jevbandits.experiments import Episode
from jevbandits.two_arm_table import (
    TwoArmTable,
    _read_complete_records,
    _read_owned_records,
    score_episode,
)


@pytest.mark.parametrize("horizon", [1, 2, 4, 6])
def test_every_reachable_small_state_matches_recursive_solver(horizon):
    table = TwoArmTable(horizon)
    for n in range(horizon + 1):
        for n1 in range(n + 1):
            n2 = n - n1
            for s1 in range(n1 + 1):
                for s2 in range(n2 + 1):
                    s, f = (s1, s2), (n1 - s1, n2 - s2)
                    expected = exact_q(s, f, horizon - n)
                    np.testing.assert_allclose(table.q(s, f), expected, atol=2e-14, rtol=0)
                    assert table.value(s, f) == pytest.approx(max(expected), abs=2e-14)
    assert table.array_bytes == comb(horizon + 4, 4) * 8


def test_known_prior_value_and_two_pull_rational_exploration_crossover():
    table = TwoArmTable(20)
    assert table.value() == pytest.approx(12.431264905858743, abs=2e-13)
    np.testing.assert_allclose(table.q((10, 0), (8, 0)), [1.1, 133 / 120], atol=1e-14)
    assert table.select_action((10, 0), (8, 0), np.random.default_rng(1)) == 1


def test_symmetry_terminal_values_and_seeded_ties():
    table = TwoArmTable(10)
    np.testing.assert_allclose(table.q((1, 3), (2, 1))[::-1], table.q((3, 1), (1, 2)), atol=1e-14)
    np.testing.assert_array_equal(table.q((3, 4), (2, 1)), [0, 0])
    rng = np.random.default_rng(444)
    choices = [table.select_action((0, 0), (0, 0), rng) for _ in range(1000)]
    assert abs(np.mean(choices) - 0.5) < 0.05
    with pytest.raises(ValueError, match="No decisions"):
        table.select_action((3, 4), (2, 1), rng)


def test_memory_and_input_guards():
    with pytest.raises(MemoryError):
        TwoArmTable(100, max_bytes=1000)
    for horizon in [0, 151, 3.5]:
        with pytest.raises(ValueError):
            TwoArmTable(horizon)
    table = TwoArmTable(2)
    for s, f in [((0,), (0,)), ((0, -1), (0, 0)), ((0.5, 0), (0, 0)), ((2, 2), (0, 0))]:
        with pytest.raises(ValueError):
            table.q(s, f)


def test_scoring_replays_world_and_counts_and_exact_policy_has_zero_loss():
    table = TwoArmTable(8)
    episode = Episode("test_compact", "prior", 2, 3, 8, "test_exact")
    for turn in range(1, 9):
        action = table.select_action(episode.s, episode.f, episode.rng(turn))
        episode.step(turn, action)
    record = episode.summary()
    scored = score_episode(record, table)
    assert scored["exact_loss"] == pytest.approx(0, abs=1e-14)
    assert scored["optimal_agreement"] == 1
    record["trace"][0]["reward"] = 1 - record["trace"][0]["reward"]
    with pytest.raises(ValueError, match="Recorded reward"):
        score_episode(record, table)


def test_live_sources_are_not_modified_and_owned_torn_writes_can_resume(tmp_path):
    source = tmp_path / "source.jsonl"
    data = b'{"a":1}\n{"a":'
    source.write_bytes(data)
    assert _read_complete_records(source) == [{"a": 1}]
    assert source.read_bytes() == data
    assert _read_owned_records(source) == [{"a": 1}]
    assert source.read_bytes() == b'{"a":1}\n'
