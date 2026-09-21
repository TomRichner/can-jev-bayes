"""No live API: pre-registration, advice mapping, backend choice, and replay."""

import asyncio
import copy
import json

import httpx
import numpy as np
import pytest

from jevbandits import advice_followup as e6
from jevbandits.baselines import exact_q
from jevbandits.client import JevClient
from jevbandits.experiments import fixtures, read_jsonl
from jevbandits.prompts import MODEL, TASK, observation, question


@pytest.fixture(scope="module")
def panel():
    return e6.fixture_panel()


def test_panel_fixed_fresh_and_targeted_without_model_selection(panel):
    assert panel == e6.fixture_panel()
    assert len(panel) == 200
    random = panel[:100]
    assert [(x["successes"], x["failures"]) for x in random] == fixtures(
        "e6_advice_random", 100
    )
    target = panel[100:]
    assert len({tuple(x["successes"] + x["failures"]) for x in target}) == 100
    for item in target:
        s, f = item["successes"], item["failures"]
        assert s[1] == f[1] == 0
        assert 10 <= s[0] + f[0] <= 80
        assert 0.51 <= (s[0] + 1) / (s[0] + f[0] + 2) <= 0.68
        q = exact_q(tuple(s), tuple(f), 10)
        assert q[1] - q[0] > 1e-4


@pytest.mark.parametrize("format_name", e6.FORMATS)
@pytest.mark.parametrize("labels", e6.LABELS)
@pytest.mark.parametrize("reversal", [0, 1])
@pytest.mark.parametrize("horizon", [1, 10])
def test_advice_all_payload_references_and_statistics_map_consistently(
    format_name, labels, reversal, horizon
):
    s, f = [10, 0], [8, 0]
    q = exact_q(tuple(s), tuple(f), horizon)
    result, gold, recommended, order = e6.make_question(
        s, f, horizon, format_name, labels, reversal, q
    )
    assert gold == (0 if horizon == 1 else 1)
    assert recommended == e6.LABELS[labels][order.index(gold)]
    obs = result["instructions"]["observation"]
    assert list(result["criteria"]) == list(e6.LABELS[labels])
    for display, arm in enumerate(obs["arms"]):
        assert arm["id"] == e6.LABELS[labels][display]
        if format_name == "advice_only":
            assert set(arm) == {"id"}
        else:
            assert arm["successes"] == s[order[display]]
            assert arm["failures"] == f[order[display]]
            assert arm["posterior_mean"] == pytest.approx(
                (s[order[display]] + 1) / (s[order[display]] + f[order[display]] + 2)
            )
    if format_name != "dp_values":
        assert obs["recommended_arm"] == recommended
    else:
        assert "recommended_arm" not in obs
        for display, arm in enumerate(obs["arms"]):
            assert arm["expected_total_reward_if_chosen_then_optimal"] == pytest.approx(
                q[order[display]]
            )
    if format_name == "inline":
        assert result["instructions"]["question"].startswith(
            f"Choose EXACTLY {recommended}."
        )
    if format_name == "per_option":
        assert "This is the recommended arm" in result["criteria"][recommended]
    if labels == "letters":
        assert "arm_00" not in json.dumps(result)
        assert "arm_01" not in json.dumps(result)


def test_nested_and_dp_original_exactly_match_existing_payload():
    s, f = [10, 0], [8, 0]
    q = exact_q(tuple(s), tuple(f), 10)
    for format_name, representation, framing in (
        ("nested", "recommendation", "recommendation"),
        ("dp_values", "dp_values", "explore"),
    ):
        for reverse in (0, 1):
            order = [0, 1] if reverse == 0 else [1, 0]
            expected = question(
                observation(s, f, 10, representation, q_values=q, label_order=order),
                framing,
            )
            actual = e6.make_question(s, f, 10, format_name, "original", reverse, q)[0]
            assert actual == expected


def test_full_factorial_has_unique_ids_and_repeats_identical_questions(panel):
    jobs, contexts = e6.build_design(panel)
    assert len(jobs) == len(contexts) == 28800
    assert len({job["id"] for job in jobs}) == 28800
    by_id = {job["id"]: job for job in jobs}
    for job, context in zip(jobs, contexts):
        assert job["id"] == context["decision_id"]
        if context["repeat"] == 0:
            assert job["question"] == by_id[job["id"][:-1] + "1"]["question"]
    assert e6.build_design(panel)[0] == jobs


def test_ties_separate_adherence_from_optimal_agreement_and_remap_probabilities():
    _, contexts = e6.build_design(
        [{"cohort": "random", "index": 0, "successes": [0, 0], "failures": [0, 0]}]
    )
    for context in contexts:
        other_display = context["order"].index(1)
        answer = {
            "action": other_display,
            "probabilities": [0.6, 0.4],
            "request_id": "mock",
            "raw_answer": {},
            "probability_mass": 1,
            "choice_probability_gap": 0.2 if other_display == 1 else 0,
        }
        row = e6.score(context, answer)
        assert row["advice_adherence"] is False
        assert row["optimal_agreement"] is True
        assert row["exact_loss"] == 0
        assert row["canonical_action"] == 1
        assert row["probabilities"] == (
            [0.6, 0.4] if context["label_order"] == 0 else [0.4, 0.6]
        )
        assert row["choice_mismatch"] == (other_display == 1)


def test_manifest_freezes_sources_panel_and_jobs_without_api(tmp_path, monkeypatch):
    small = [{"cohort": "random", "index": 0, "successes": [1, 0], "failures": [0, 0]}]
    monkeypatch.setattr(e6, "fixture_panel", lambda: small)
    monkeypatch.setattr(
        e6, "JevClient", lambda *a, **k: pytest.fail("prepare opened client")
    )
    e6.prepare(tmp_path)
    jobs, _ = e6.build_design(small)
    e6.verify_manifest(tmp_path, small, jobs)
    e6.prepare(tmp_path)
    changed = copy.deepcopy(jobs)
    changed[0]["question"]["instructions"]["question"] += " changed"
    with pytest.raises(ValueError, match="ordered_jobs_sha256"):
        e6.verify_manifest(tmp_path, small, changed)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["source_sha256"]["src/jevbandits/prompts.py"] = "changed"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source_sha256"):
        e6.verify_manifest(tmp_path, small, jobs)


def test_run_refuses_unprepared_before_api(tmp_path, monkeypatch):
    monkeypatch.setattr(
        e6, "JevClient", lambda *a, **k: pytest.fail("opened client before validation")
    )
    with pytest.raises(ValueError, match="previously prepared"):
        e6.verify_manifest(tmp_path, [], [])


def test_mock_sqlite_replay_maps_labels_and_preserves_backend_choice(tmp_path):
    jobs, contexts = e6.build_design(
        [{"cohort": "targeted", "index": 0, "successes": [10, 0], "failures": [8, 0]}]
    )
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        assert payload["state"] == TASK
        calls.append(payload)
        answers = {}
        for key, q in payload["questions"].items():
            ids = list(q["criteria"])
            answers[key] = {
                "type": "choice",
                "choice": ids[0],
                "probabilities": {ids[0]: 0.49, ids[1]: 0.51},
                "confidence": 0.5,
            }
        return httpx.Response(
            200,
            json={"model": MODEL, "usage": {"input_tokens": 100}, "answers": answers},
        )

    async def run():
        client = JevClient(
            tmp_path, key="test", transport=httpx.MockTransport(handler), rate=1e9
        )
        try:
            await e6.execute(client, jobs, contexts, max_decisions=80)
            assert len(calls) == 5
            assert len(read_jsonl(tmp_path / "diagnostics.jsonl")) == 80
            # Simulate loss of derived output; successful scientific IDs replay from SQLite.
            (tmp_path / "diagnostics.jsonl").unlink()
            await e6.execute(client, jobs, contexts, max_decisions=80)
            assert len(calls) == 5
            await e6.execute(client, jobs, contexts)
            count = len(calls)
            await e6.execute(client, jobs, contexts)
            assert len(calls) == count
        finally:
            await client.close()

    asyncio.run(run())
    records = read_jsonl(tmp_path / "diagnostics.jsonl")
    assert len(records) == 144
    for row in records:
        assert row["action"] == 0
        assert row["canonical_action"] == row["label_order"]
        assert row["choice_probability_gap"] == pytest.approx(0.02)
        assert row["choice_mismatch"] is True
        assert row["advice_adherence"] == (
            row["canonical_action"] == row["recommended_canonical_action"]
        )
        assert row["distribution_exact_loss"] == pytest.approx(
            max(row["q_values"]) - np.dot(row["probabilities"], row["q_values"])
        )
