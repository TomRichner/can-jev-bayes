"""E8 tests use local worlds and mock HTTP only; never access live Jev."""

import asyncio
import copy
import json

import httpx
import numpy as np
import pytest

from jevbandits import evidence_followup as e8
from jevbandits import experiments
from jevbandits.client import JevClient
from jevbandits.experiments import read_jsonl
from jevbandits.prompts import MODEL, OBJECTIVES, TASK


def small_design(monkeypatch):
    monkeypatch.setattr(
        e8,
        "DESIGN",
        {
            "ks": [3],
            "families": ["prior"],
            "n": 2,
            "horizon": 4,
            "policies": list(e8.POLICIES),
        },
    )


def test_public_means_payload_and_no_global_registry_mutation():
    original_policies = copy.deepcopy(experiments.POLICIES)
    original_designs = copy.deepcopy(experiments.DESIGNS)
    payloads = {}
    for policy in e8.POLICIES:
        ep = e8.EvidenceEpisode(e8.EXPERIMENT, "prior", 3, 0, 100, policy)
        ep.s[:] = [0, 10, 1]
        ep.f[:] = [0, 8, 3]
        original = ep.job(23)
        ep.theta[:] = [0.9, 0.8, 0.7]
        ep.outcomes[:] = 1
        assert ep.job(23) == original
        payloads[policy] = original["question"]
    means = payloads["means_direct"]["instructions"]["observation"]
    assert means["remaining_pulls_including_this_one"] == 78
    assert [arm["posterior_mean"] for arm in means["arms"]] == [0.5, 0.55, 0.333333]
    for arm in means["arms"]:
        assert set(arm) == {"id", "successes", "failures", "posterior_mean"}
    for payload in payloads.values():
        assert payload["instructions"]["question"] == OBJECTIVES["explore"]
        assert not {"theta", "outcomes", "family", "policy"}.intersection(
            payload["instructions"]["observation"]
        )
        for actual, expected in zip(
            payload["instructions"]["observation"]["arms"], means["arms"], strict=True
        ):
            assert (actual["successes"], actual["failures"]) == (
                expected["successes"],
                expected["failures"],
            )
    assert (
        "posterior_sd"
        in payloads["bayes_direct"]["instructions"]["observation"]["arms"][0]
    )
    assert experiments.POLICIES == original_policies
    assert experiments.DESIGNS == original_designs


def test_fresh_seed_pairing_and_nth_pull_streams():
    episodes = [
        e8.EvidenceEpisode(e8.EXPERIMENT, "prior", 3, 7, 100, p)
        for p in [*e8.POLICIES, *e8.BASELINES]
    ]
    first = episodes[0]
    for ep in episodes[1:]:
        assert ep.seed == first.seed
        np.testing.assert_array_equal(ep.theta, first.theta)
        np.testing.assert_array_equal(ep.outcomes, first.outcomes)
    old = experiments.Episode("e4_scaling", "prior", 3, 7, 100, "counts_direct")
    assert first.seed != old.seed
    second = episodes[1]
    first.step(1, 0)
    first.step(2, 0)
    second.step(1, 1)
    second.step(2, 0)
    second.step(3, 0)
    assert [r["reward"] for r in first.trace] == [r["reward"] for r in second.trace[1:]]


def test_manifest_full_design_and_prepare_has_no_api(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Preparation must not initialize an API client")

    monkeypatch.setattr(e8, "JevClient", forbidden)
    manifest = e8.prepare(tmp_path)
    assert manifest["tasks"] == 320
    assert manifest["jev_episodes"] == 960
    assert manifest["baseline_episodes"] == 2560
    assert manifest["decisions"] == 96000
    assert list(manifest["policies"]) == [
        "counts_direct",
        "means_direct",
        "bayes_direct",
    ]
    assert manifest["classical"] == [
        "random",
        "greedy",
        "ts",
        "bayes_ucb",
        "ucb1",
        "knowledge_gradient",
        "ids",
        "finite_ap_index",
    ]
    assert manifest["primary_analysis"]["bonferroni_two_contrast_quantiles"] == [
        0.0125,
        0.9875,
    ]
    assert len(manifest["primary_analysis"]["contrasts_left_minus_right"]) == 2
    assert set(manifest["source_sha256"]) == set(e8.FROZEN_FILES)
    assert not (tmp_path / "ledger.sqlite").exists()
    assert e8.prepare(tmp_path) == json.loads(json.dumps(manifest))
    e8.verify_manifest(tmp_path)
    prompts = json.loads((tmp_path / "prompt_fixtures.json").read_text())
    assert len(prompts) == 36
    prompts[0]["question"]["instructions"]["observation"]["arms"][0]["successes"] = 9
    (tmp_path / "prompt_fixtures.json").write_text(json.dumps(prompts))
    with pytest.raises(ValueError, match="prompt_fixtures.json changed"):
        e8.verify_manifest(tmp_path)


def test_manifest_refuses_design_or_source_change(tmp_path, monkeypatch):
    small_design(monkeypatch)
    e8.prepare(tmp_path)
    monkeypatch.setitem(e8.DESIGN, "n", 3)
    with pytest.raises(ValueError, match="changed"):
        e8.verify_manifest(tmp_path)
    monkeypatch.setitem(e8.DESIGN, "n", 2)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["source_sha256"]["src/jevbandits/evidence_followup.py"] = "tampered"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source_sha256 changed"):
        e8.verify_manifest(tmp_path)


def mock_response(request):
    payload = json.loads(request.content)
    assert payload["state"] == TASK
    answers = {}
    for key, q in payload["questions"].items():
        arms = q["instructions"]["observation"]["arms"]
        # A deterministic state-dependent action exercises replay beyond fixed-arm traces.
        selected = sum(a["successes"] + a["failures"] for a in arms) % len(arms)
        ids = list(q["criteria"])
        answers[key] = {
            "type": "choice",
            "choice": ids[selected],
            "probabilities": {arm: float(i == selected) for i, arm in enumerate(ids)},
            "confidence": 1.0,
        }
    return httpx.Response(
        200, json={"model": MODEL, "usage": {"input_tokens": 100}, "answers": answers}
    )


def test_durable_checkpoint_replay_and_completed_skip(tmp_path, monkeypatch):
    small_design(monkeypatch)
    resumed, uninterrupted = tmp_path / "resumed", tmp_path / "uninterrupted"
    e8.prepare(resumed)
    e8.prepare(uninterrupted)
    requests = 0

    def interrupted_handler(request):
        nonlocal requests
        requests += 1
        if requests == 3:
            raise RuntimeError("intentional interruption")
        return mock_response(request)

    def make_client(path, handler):
        return JevClient(
            path,
            key="mock-only",
            transport=httpx.MockTransport(handler),
            rate=1e9,
            batch_size=16,
        )

    async def run():
        client = make_client(resumed, interrupted_handler)
        try:
            with pytest.raises(RuntimeError, match="intentional"):
                await e8.execute(client, resumed)
            assert client.accounting()["inference_decisions"] == 12
            assert not read_jsonl(resumed / "episodes_jev.jsonl")
        finally:
            await client.close()
        # New client/connection proves durable disk replay rather than object memory.
        client = make_client(resumed, mock_response)
        try:
            await e8.execute(client, resumed)
            assert client.accounting()["inference_decisions"] == 24
            assert (
                client.accounting()["http_requests"] == 5
            )  # four successful, one interrupted
            before = client.accounting()["http_requests"]
            await e8.execute(client, resumed)
            assert client.accounting()["http_requests"] == before
        finally:
            await client.close()
        client = make_client(uninterrupted, mock_response)
        try:
            await e8.execute(client, uninterrupted)
        finally:
            await client.close()
        a = read_jsonl(resumed / "episodes.jsonl")
        b = read_jsonl(uninterrupted / "episodes.jsonl")
        assert a == b
        assert len(a) == 6
        assert {r["representation"] for r in a} == {"counts", "means", "bayes"}
        assert all(r["exact_loss"] is None and r["completed"] for r in a)

    asyncio.run(run())


def test_all_baselines_offline_and_merge(tmp_path, monkeypatch):
    small_design(monkeypatch)
    monkeypatch.setitem(e8.DESIGN, "n", 1)
    monkeypatch.setitem(e8.DESIGN, "horizon", 2)
    e8.prepare(tmp_path)
    asyncio.run(e8.execute(None, tmp_path, only_baselines=True))
    rows = read_jsonl(tmp_path / "episodes.jsonl")
    assert len(rows) == len(e8.BASELINES)
    assert {r["policy"] for r in rows} == set(e8.BASELINES)
    assert len({r["seed"] for r in rows}) == 1
    index = next(r for r in rows if r["policy"] == "finite_ap_index")
    assert index["index_tolerance"] == 1e-6
    assert all("index_ambiguous_ranking" in t for t in index["trace"])
    assert not (tmp_path / "ledger.sqlite").exists()
    asyncio.run(e8.execute(None, tmp_path, only_baselines=True))
    assert read_jsonl(tmp_path / "episodes.jsonl") == rows
