"""Offline tests for frozen batching, real transport shapes, replay, and pairing."""

import asyncio
import copy
import json
from collections import Counter

import httpx
import pytest

from jevbandits import advice_followup as e6
from jevbandits import batch_control as bc
from jevbandits.client import JevClient
from jevbandits.experiments import read_jsonl
from jevbandits.prompts import MODEL, TASK


@pytest.fixture(scope="module")
def design():
    panel = [f for f in e6.fixture_panel() if f["index"] < 10]
    jobs, contexts = bc.build_design(panel)
    return panel, jobs, contexts


def test_exact_source_payloads_full_size_balanced_order(design):
    panel, jobs, contexts = design
    assert Counter(f["cohort"] for f in panel) == {"random": 10, "targeted": 10}
    assert len(jobs) == len(contexts) == len({j["id"] for j in jobs}) == 1920
    source_jobs, _ = e6.build_design(panel)
    source = {j["id"]: j["question"] for j in source_jobs}
    identities = Counter()
    for job, context in zip(jobs, contexts):
        assert job["question"] == source[context["source_decision_id"]]
        assert job["id"] == context["decision_id"]
        assert context["horizon"] == 10
        assert context["source_experiment"] == e6.EXPERIMENT
        identities[context["source_decision_id"]] += 1
        assert len(json.dumps(job["question"]).encode()) < 54000 / 16
    assert len(identities) == 960
    assert set(identities.values()) == {2}
    first = []
    for start in range(0, 1920, 32):
        first.append(contexts[start]["batch_mode"])
        assert {c["batch_mode"] for c in contexts[start : start + 16]} == {first[-1]}
        assert {c["batch_mode"] for c in contexts[start + 16 : start + 32]} == {
            "mixed" if first[-1] == "alone" else "alone"
        }
        assert [j["question"] for j in jobs[start : start + 16]] == [
            j["question"] for j in jobs[start + 16 : start + 32]
        ]
    assert Counter(first) == {"alone": 30, "mixed": 30}
    assert bc.build_design(panel) == (jobs, contexts)


def test_prepare_freezes_without_client_and_rejects_changes(
    tmp_path, monkeypatch, design
):
    panel, jobs, contexts = design
    monkeypatch.setattr(bc, "source_panel", lambda: panel)
    monkeypatch.setattr(
        bc, "JevClient", lambda *a, **k: pytest.fail("Opened API client")
    )
    manifest = bc.prepare(tmp_path)
    assert manifest["decisions"] == 1920
    assert len(manifest["question_sha256"]) == 1920
    assert bc.prepare(tmp_path) == manifest
    changed = copy.deepcopy(jobs)
    changed[0]["question"]["instructions"]["question"] += " changed"
    with pytest.raises(ValueError, match="ordered_jobs_sha256"):
        bc.verify_manifest(tmp_path, panel, changed, contexts)
    frozen = json.loads((tmp_path / "jobs.json").read_text())
    frozen.reverse()
    (tmp_path / "jobs.json").write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="jobs file"):
        bc.verify_manifest(tmp_path, panel, jobs, contexts)


def mock_handler(calls):
    def handler(request):
        payload = json.loads(request.content)
        assert payload["state"] == TASK
        assert len(payload["questions"]) in (1, 16)
        calls.append(payload)
        answers = {}
        for key, question in payload["questions"].items():
            ids = list(question["criteria"])
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

    return handler


def test_transport_exact_shapes_pause_resume_and_lost_diagnostics(tmp_path, design):
    _, jobs, contexts = design
    jobs, contexts = jobs[:64], contexts[:64]
    calls = []

    async def exercise():
        client = JevClient(
            tmp_path,
            key="mock",
            transport=httpx.MockTransport(mock_handler(calls)),
            rate=1e9,
        )
        try:
            await bc.execute(client, jobs, contexts, max_decisions=17)
            assert len(read_jsonl(tmp_path / "diagnostics.jsonl")) == 16
            await bc.execute(client, jobs, contexts)
            assert Counter(len(c["questions"]) for c in calls) == {1: 32, 16: 2}
            (tmp_path / "diagnostics.jsonl").unlink()
            await bc.execute(client, jobs, contexts)
            await bc.execute(client, jobs, contexts)
            assert len(calls) == 34
        finally:
            await client.close()

    asyncio.run(exercise())
    rows = read_jsonl(tmp_path / "diagnostics.jsonl")
    assert len(rows) == 64
    assert all(
        r["experiment"] == "batch_control" and r["source_experiment"] == e6.EXPERIMENT
        for r in rows
    )
    assert all(r["choice_mismatch"] for r in rows)
    assert all(r["canonical_action"] == r["label_order"] for r in rows)
    result = bc.summarize(rows, design[2])
    assert result["matched_questions"] == 32
    assert result["complete"] is False
    assert all(r["complete_fixtures"] == 0 for r in result["estimates"])


def test_partial_mixed_unpack_replays_original_complete_response(tmp_path, design):
    _, jobs, contexts = design
    start = next(
        i for i in range(0, len(jobs), 16) if contexts[i]["batch_mode"] == "mixed"
    )
    jobs, contexts = jobs[start : start + 16], contexts[start : start + 16]
    calls = []

    async def exercise():
        client = JevClient(
            tmp_path,
            key="mock",
            transport=httpx.MockTransport(mock_handler(calls)),
            rate=1e9,
        )
        try:
            client.batch_size = 16
            await client.evaluate(jobs)
            client.db.execute("DELETE FROM decisions WHERE id=?", (jobs[0]["id"],))
            client.db.commit()
            await bc.execute(client, jobs, contexts)
            assert len(calls) == 1
            assert len(read_jsonl(tmp_path / "diagnostics.jsonl")) == 16
        finally:
            await client.close()

    asyncio.run(exercise())


def test_report_pairing_adherence_exclusion_and_repeat_baseline(design, tmp_path):
    _, _, contexts = design
    rows = []
    for context in contexts:
        row = {k: v for k, v in context.items() if k != "order"}
        p = [0.75, 0.25] if context["batch_mode"] == "alone" else [0.25, 0.75]
        row.update(
            probabilities=p,
            optimal_agreement=context["batch_mode"] == "mixed",
            advice_adherence=context["batch_mode"] == "mixed",
            distribution_exact_loss=0.0,
            distribution_advice_adherence=p[0],
        )
        rows.append(row)
    (tmp_path / "diagnostics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    result = bc.report(tmp_path, contexts)
    assert result["complete"] is True
    assert result["matched_questions"] == 960
    for record in result["estimates"]:
        assert record["complete_fixtures"] == 10
        metric = record["metric"]
        if record["format"] == "dp_values":
            assert "adherence" not in metric
        expected = {
            "probability_tv": 0.5,
            "repeat_mean_probability_tv": 0.5,
            "alone_repeat_probability_tv": 0.0,
            "mixed_repeat_probability_tv": 0.0,
            "optimal_agreement_delta": 1.0,
            "advice_adherence_delta": 1.0,
            "distribution_exact_loss_delta": 0.0,
            "distribution_advice_adherence_delta": -0.5,
        }[metric]
        assert record["mean"] == pytest.approx(expected)
        assert record["ci95"] == pytest.approx([expected, expected])
    assert all(r["mean"] == 0 for r in result["format_minus_nested_contrasts"])
    assert (tmp_path / "report.md").exists()
    with pytest.raises(ValueError, match="Duplicate"):
        bc.paired_observations(rows + [rows[0]])
