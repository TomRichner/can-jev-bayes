"""E10 probability semantics, proper scoring, frozen payloads, and mock replay."""

import asyncio
import copy
import json

import httpx
import numpy as np
import pandas as pd
import pytest

from jevbandits import primitive_followup as e10
from jevbandits.experiments import read_jsonl
from jevbandits.forecast_followup import ForecastClient, best_probabilities
from jevbandits.prompts import MODEL, TASK, stable_seed


@pytest.fixture
def fixture():
    s, f = np.array([10, 0]), np.array([8, 0])
    return {
        "fixture_id": f"{e10.EXPERIMENT}:2:0",
        "k": 2,
        "index": 0,
        "successes": s.tolist(),
        "failures": f.tolist(),
        "next_reward": ((s + 1) / (s + f + 2)).tolist(),
        "best_arm": best_probabilities(s, f).tolist(),
    }


@pytest.mark.parametrize("representation", e10.REPRESENTATIONS)
def test_reward_changes_only_primitive_and_exact_event_fields(fixture, representation):
    noul = e10.make_question(fixture, representation, "next_reward", "noul", 0)
    choice = e10.make_question(fixture, representation, "next_reward", "choice", 0)
    assert noul.pop("type") == "noul"
    assert choice.pop("type") == "choice"
    assert noul == choice
    assert list(choice["criteria"]) == ["true", "false"]
    arms = choice["instructions"]["observation"]["arms"]
    if representation.startswith("oracle"):
        assert [a[e10.EVENT_FIELDS["next_reward"]] for a in arms] == [0.55, 0.5]
        assert all(e10.EVENT_FIELDS["best_arm"] not in a for a in arms)
    if representation == "oracle_probs_only":
        assert all(set(a) == {"id", e10.EVENT_FIELDS["next_reward"]} for a in arms)
    if representation == "counts":
        assert all(set(a) == {"id", "successes", "failures"} for a in arms)


def test_best_marginal_and_joint_share_evidence_and_reference(fixture):
    assert fixture["best_arm"] != fixture["next_reward"]
    assert sum(fixture["best_arm"]) == pytest.approx(1)
    for representation in e10.REPRESENTATIONS:
        joint = e10.make_question(fixture, representation, "best_arm", "choice")
        marginal = e10.make_question(fixture, representation, "best_arm", "noul", 1)
        obs = marginal["instructions"]["observation"].copy()
        assert obs.pop("queried_arm") == "arm_01"
        assert obs == joint["instructions"]["observation"]
        assert "largest" in marginal["instructions"]["question"]
        if representation.startswith("oracle"):
            assert [a[e10.EVENT_FIELDS["best_arm"]] for a in obs["arms"]] == fixture[
                "best_arm"
            ]
            assert all(e10.EVENT_FIELDS["next_reward"] not in a for a in obs["arms"])


def test_fresh_seed_and_reference_generation(monkeypatch):
    monkeypatch.setattr(e10, "N", 2)
    panel = e10.fixture_panel()
    assert panel == e10.fixture_panel()
    assert len(panel) == 6
    for k in e10.KS:
        rng = np.random.default_rng(
            stable_seed(e10.SEED, e10.EXPERIMENT, "fixtures", k)
        )
        n = rng.choice([0, 2, 5, 10, 20], k)
        s = rng.integers(0, n + 1)
        item = next(f for f in panel if f["k"] == k)
        assert item["successes"] == s.tolist()
        np.testing.assert_allclose(item["next_reward"], (s + 1) / (n + 2))
        assert sum(item["best_arm"]) == pytest.approx(1)
        assert stable_seed(e10.SEED, e10.EXPERIMENT, "fixtures", k) != stable_seed(
            e10.SEED, "e7_probability_forecasts", k
        )


def test_full_factorial_count_unique_ids_and_repeats():
    panel = []
    for k in e10.KS:
        for i in range(30):
            panel.append(
                {
                    "fixture_id": f"{e10.EXPERIMENT}:{k}:{i}",
                    "k": k,
                    "index": i,
                    "successes": [0] * k,
                    "failures": [0] * k,
                    "next_reward": [0.5] * k,
                    "best_arm": [1 / k] * k,
                }
            )
    jobs, contexts = e10.build_design(panel)
    assert len(jobs) == len(contexts) == 16560
    assert len({j["id"] for j in jobs}) == 16560
    repeated = {}
    for job, context in zip(jobs, contexts, strict=True):
        assert job["id"] == context["decision_id"]
        key = tuple(
            context[n]
            for n in ["fixture_id", "representation", "target", "primitive", "arm"]
        )
        if key in repeated:
            assert repeated[key] == job["question"]
        repeated[key] = job["question"]


def test_common_binary_scale_and_unnormalized_noul_primary():
    q, p = np.array([0.7, 0.2, 0.1]), np.array([0.4, 0.3, 0.3])
    joint = e10.event_scores(q, p)
    individual = [e10.event_scores([qi], [pi]) for qi, pi in zip(q, p, strict=True)]
    assert joint["mean_binary_excess_brier"] == pytest.approx(2 / 3 * sum((p - q) ** 2))
    assert joint["mean_binary_excess_brier"] == pytest.approx(
        np.mean([r["mean_binary_excess_brier"] for r in individual])
    )
    assert e10.event_scores(q, q)["mean_binary_excess_brier"] == 0
    raw = [0.8, 0.8, 0.8]
    risk = e10.event_scores(q, raw)["mean_binary_excess_brier"]
    assert risk != pytest.approx(
        e10.event_scores(q, np.array(raw) / sum(raw))["mean_binary_excess_brier"]
    )
    assert np.isfinite(e10.event_scores([0.5], [0])["mean_binary_excess_log_loss"])
    with pytest.raises(ValueError):
        e10.event_scores([0.5], [1.1])


def score_all(jobs, contexts):
    records = []
    for job, context in zip(jobs, contexts, strict=True):
        probabilities = (
            context["reference_event_probabilities"]
            if context["arm"] is None
            else [
                context["reference_event_probabilities"][0],
                1 - context["reference_event_probabilities"][0],
            ]
        )
        records.append(
            e10.score(
                context,
                {
                    "probabilities": probabilities,
                    "raw_answer": {},
                    "request_id": job["id"],
                    "probability_mass": 1,
                },
            )
        )
    return records


def test_fixture_units_require_complete_packages_and_zero_sum_is_undefined(fixture):
    jobs, contexts = e10.build_design([fixture])
    records = score_all(jobs, contexts)
    units, incomplete = e10.fixture_units(records)
    assert len(units) == 16 and incomplete.empty
    assert (units.mean_binary_excess_brier == 0).all()
    units, incomplete = e10.fixture_units(records[1:])
    assert len(units) == 15 and len(incomplete) == 1
    for record in records:
        if record["target"] == "best_arm" and record["primitive"] == "noul":
            record["prediction_event_probabilities"] = [0.0]
    coherence = e10.coherence_records(records)
    assert len(coherence) == 8
    assert all(
        r["sum_probabilities"] == 0
        and r["normalized_mean_binary_excess_brier"] is None
        and not r["normalization_defined"]
        for r in coherence
    )


def test_equal_k_bootstrap_primary_pairing_and_intervals():
    records = []
    for k, n, effect in [(2, 2, 1.0), (5, 3, 2.0), (15, 4, 3.0)]:
        for i in range(n):
            for primitive, risk in [("choice", effect), ("noul", 0.0)]:
                records.append(
                    {
                        "fixture_id": f"{k}:{i}",
                        "k": k,
                        "target": "next_reward",
                        "representation": "bayes",
                        "primitive": primitive,
                        "mean_binary_excess_brier": risk,
                    }
                )
    result = e10.bootstrap_contrast(
        pd.DataFrame(records), *e10.PRIMARY[0], samples=1000
    )
    assert result["mean_difference"] == 2  # equal K, not sample-size weighting
    assert result["ci95_low"] == result["ci97_5_low"] == 2
    assert not result["complete"]
    without_k15 = pd.DataFrame([r for r in records if r["k"] != 15])
    missing = e10.bootstrap_contrast(without_k15, *e10.PRIMARY[0], samples=100)
    assert missing["mean_difference"] is None and missing["ci95_low"] is None


def test_manifest_freezes_without_client_and_detects_payload_change(
    tmp_path, monkeypatch, fixture
):
    monkeypatch.setattr(e10, "fixture_panel", lambda: [fixture])
    monkeypatch.setattr(
        e10, "ForecastClient", lambda *a, **k: pytest.fail("prepare opened API client")
    )
    manifest = e10.prepare(tmp_path)
    assert manifest["questions"] == 56
    assert len(manifest["primary"]["contrasts_target_left_right"]) == 2
    assert manifest["primary"]["bonferroni_quantiles"] == [0.0125, 0.9875]
    assert set(manifest["source_sha256"]) == set(e10.FROZEN_FILES)
    assert not (tmp_path / "ledger.sqlite").exists()
    e10.prepare(tmp_path)
    jobs, contexts = e10.build_design([fixture])
    changed = copy.deepcopy(jobs)
    changed[0]["question"]["instructions"]["question"] += " changed"
    with pytest.raises(ValueError, match="ordered_jobs_sha256"):
        e10.verify_manifest(tmp_path, [fixture], changed, contexts)


def test_durable_mock_replay_and_report(tmp_path, fixture):
    jobs, contexts = e10.build_design([fixture])
    # Record-level replay is insensitive to changed batching after a partial response.
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        payload = json.loads(request.content)
        assert payload["state"] == TASK
        answers = {}
        for key, q in payload["questions"].items():
            if q["type"] == "noul":
                answers[key] = {"type": "noul", "noul": 0.4}
            else:
                ids = list(q["criteria"])
                answers[key] = {
                    "type": "choice",
                    "choice": ids[0],
                    "probabilities": {name: 1 / len(ids) for name in ids},
                    "confidence": 0.5,
                }
        return httpx.Response(
            200,
            json={"model": MODEL, "usage": {"input_tokens": 100}, "answers": answers},
        )

    def client():
        return ForecastClient(
            tmp_path / "run",
            key="mock-only",
            transport=httpx.MockTransport(handler),
            rate=1e9,
        )

    async def run():
        c = client()
        try:
            # Simulate process loss after durable responses, before any scored row.
            await c.evaluate(jobs[:7])
            assert c.accounting()["inference_decisions"] == 7
        finally:
            await c.close()
        c = client()
        try:
            await e10.execute(c, jobs, contexts)
            assert c.accounting()["inference_decisions"] == len(jobs)
            done = calls
            await e10.execute(c, jobs, contexts)
            assert calls == done
            rows = read_jsonl(c.run_dir / "forecasts.jsonl")
            assert len(rows) == len(jobs)
            path = e10.report(c.run_dir, tmp_path / "report")
            assert path.exists()
            assert "not realized reward" in path.read_text()
            assert len(pd.read_csv(tmp_path / "report" / "fixture_risks.csv")) == 16
        finally:
            await c.close()

    asyncio.run(run())
