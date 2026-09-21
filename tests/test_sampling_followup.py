"""E9 tests use local worlds and mock HTTP only; never access live Jev."""

import asyncio
import copy
import json

import httpx
import numpy as np
import pytest

from jevbandits import experiments
from jevbandits import sampling_followup as e9
from jevbandits.client import JevClient
from jevbandits.experiments import read_jsonl
from jevbandits.prompts import MODEL, OBJECTIVES, TASK


def small_design(monkeypatch):
    monkeypatch.setattr(
        e9,
        "DESIGN",
        {
            "ks": [2],
            "families": ["prior"],
            "n": 2,
            "horizon": 4,
            "policies": list(e9.POLICIES),
        },
    )


def test_payloads_identical_to_original_policies_without_global_mutation():
    original_policies = copy.deepcopy(experiments.POLICIES)
    original_designs = copy.deepcopy(experiments.DESIGNS)
    for k in (2, 10):
        payloads = []
        for policy in e9.POLICIES:
            ep = e9.SamplingEpisode(e9.EXPERIMENT, "prior", k, 0, 100, policy)
            ep.s[:] = np.arange(k)
            ep.f[:] = np.arange(k)[::-1]
            original = ep.job(23)
            reference = experiments.Episode("e4_scaling", "prior", k, 0, 100, policy)
            reference.s[:], reference.f[:] = ep.s, ep.f
            assert original["question"] == reference.job(23)["question"]
            ep.theta[:] = 0.99
            ep.outcomes[:] = 1
            assert ep.job(23) == original
            payloads.append(original["question"])
        assert payloads[0] == payloads[1]
        assert payloads[0]["instructions"]["question"] == OBJECTIVES["explore"]
    assert experiments.POLICIES == original_policies
    assert experiments.DESIGNS == original_designs


def test_fresh_seed_pairing_and_nth_pull_streams():
    episodes = [
        e9.SamplingEpisode(e9.EXPERIMENT, "prior", 2, 7, 100, p)
        for p in [*e9.POLICIES, *e9.BASELINES]
    ]
    first = episodes[0]
    for ep in episodes[1:]:
        assert ep.seed == first.seed
        np.testing.assert_array_equal(ep.theta, first.theta)
        np.testing.assert_array_equal(ep.outcomes, first.outcomes)
    old = experiments.Episode("e4_scaling", "prior", 2, 7, 100, "counts_direct")
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

    monkeypatch.setattr(e9, "JevClient", forbidden)
    manifest = e9.prepare(tmp_path)
    assert manifest["tasks"] == 200
    assert manifest["jev_episodes"] == 400
    assert manifest["baseline_episodes"] == 1600
    assert manifest["decisions"] == 40000
    assert list(manifest["policies"]) == [
        "bayes_direct",
        "bayes_sample",
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
    assert manifest["primary_analysis"]["single_interaction"] == e9.PRIMARY_CONTRAST
    assert manifest["primary_analysis"]["pointwise_quantiles"] == [0.025, 0.975]
    assert set(manifest["source_sha256"]) == set(e9.FROZEN_FILES)
    assert not (tmp_path / "ledger.sqlite").exists()
    assert e9.prepare(tmp_path) == json.loads(json.dumps(manifest))
    e9.verify_manifest(tmp_path)
    prompts = json.loads((tmp_path / "prompt_fixtures.json").read_text())
    assert len(prompts) == 24
    prompts[0]["question"]["instructions"]["observation"]["arms"][0]["successes"] = 9
    (tmp_path / "prompt_fixtures.json").write_text(json.dumps(prompts))
    with pytest.raises(ValueError, match="prompt_fixtures.json changed"):
        e9.verify_manifest(tmp_path)


def test_manifest_refuses_design_or_source_change(tmp_path, monkeypatch):
    small_design(monkeypatch)
    e9.prepare(tmp_path)
    monkeypatch.setitem(e9.DESIGN, "n", 3)
    with pytest.raises(ValueError, match="changed"):
        e9.verify_manifest(tmp_path)
    monkeypatch.setitem(e9.DESIGN, "n", 2)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["source_sha256"]["src/jevbandits/sampling_followup.py"] = "tampered"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source_sha256 changed"):
        e9.verify_manifest(tmp_path)


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
            "probabilities": {
                arm: (0.2 if i == selected else 0.8) for i, arm in enumerate(ids)
            },
            "confidence": 0.8,
        }
    return httpx.Response(
        200, json={"model": MODEL, "usage": {"input_tokens": 100}, "answers": answers}
    )


def test_durable_checkpoint_replay_and_completed_skip(tmp_path, monkeypatch):
    small_design(monkeypatch)
    resumed, uninterrupted = tmp_path / "resumed", tmp_path / "uninterrupted"
    e9.prepare(resumed)
    e9.prepare(uninterrupted)
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
                await e9.execute(client, resumed)
            assert client.accounting()["inference_decisions"] == 8
            assert not read_jsonl(resumed / "episodes_jev.jsonl")
        finally:
            await client.close()
        # New client/connection proves durable disk replay rather than object memory.
        client = make_client(resumed, mock_response)
        try:
            await e9.execute(client, resumed)
            assert client.accounting()["inference_decisions"] == 16
            assert (
                client.accounting()["http_requests"] == 5
            )  # four successful, one interrupted
            before = client.accounting()["http_requests"]
            await e9.execute(client, resumed)
            assert client.accounting()["http_requests"] == before
        finally:
            await client.close()
        client = make_client(uninterrupted, mock_response)
        try:
            await e9.execute(client, uninterrupted)
        finally:
            await client.close()
        a = read_jsonl(resumed / "episodes.jsonl")
        b = read_jsonl(uninterrupted / "episodes.jsonl")
        assert a == b
        assert len(a) == 4
        assert {r["representation"] for r in a} == {"bayes"}
        assert {r["action_rule"] for r in a} == {"direct", "sample"}
        assert all(r["exact_loss"] is None and r["completed"] for r in a)

    asyncio.run(run())


def test_all_baselines_offline_and_merge(tmp_path, monkeypatch):
    small_design(monkeypatch)
    monkeypatch.setitem(e9.DESIGN, "n", 1)
    monkeypatch.setitem(e9.DESIGN, "horizon", 2)
    e9.prepare(tmp_path)
    asyncio.run(e9.execute(None, tmp_path, only_baselines=True))
    rows = read_jsonl(tmp_path / "episodes.jsonl")
    assert len(rows) == len(e9.BASELINES)
    assert {r["policy"] for r in rows} == set(e9.BASELINES)
    assert len({r["seed"] for r in rows}) == 1
    index = next(r for r in rows if r["policy"] == "finite_ap_index")
    assert index["index_tolerance"] == 1e-6
    assert all("index_ambiguous_ranking" in t for t in index["trace"])
    assert not (tmp_path / "ledger.sqlite").exists()
    asyncio.run(e9.execute(None, tmp_path, only_baselines=True))
    assert read_jsonl(tmp_path / "episodes.jsonl") == rows


def test_sampling_uses_policy_turn_rng_while_direct_retains_backend_choice(
    tmp_path, monkeypatch
):
    small_design(monkeypatch)
    e9.prepare(tmp_path)

    async def run():
        client = JevClient(
            tmp_path,
            key="mock-only",
            transport=httpx.MockTransport(mock_response),
            rate=1e9,
        )
        try:
            await e9.execute(client, tmp_path)
        finally:
            await client.close()

    asyncio.run(run())
    for row in read_jsonl(tmp_path / "episodes_jev.jsonl"):
        ep = experiments.Episode(
            e9.EXPERIMENT,
            "prior",
            2,
            int(row["episode_id"].split(":")[-1]),
            4,
            row["policy"],
        )
        for turn, trace in enumerate(row["trace"], 1):
            backend = (turn - 1) % 2
            probs = [0.2 if i == backend else 0.8 for i in range(2)]
            expected = (
                backend
                if row["policy"] == "bayes_direct"
                else int(ep.rng(turn).choice(2, p=probs))
            )
            assert trace["action"] == expected
            assert trace["choice_mismatch"] is True


def synthetic_rows(n=100):
    rows = []
    for k, effect in [(2, 1), (10, -4)]:
        for i in range(n):
            for policy in e9.POLICIES:
                rows.append(
                    {
                        "experiment": e9.EXPERIMENT,
                        "k": k,
                        "family": "prior",
                        "horizon": 100,
                        "episode_id": f"{e9.EXPERIMENT}:prior:{k}:{i}",
                        "policy": policy,
                        "pseudo_regret": 10
                        + i / 100
                        + (effect if policy == "bayes_sample" else 0),
                        "completed": True,
                    }
                )
    return rows


def test_single_interaction_orientation_and_stratified_paired_bootstrap():
    rows = synthetic_rows()
    result = e9.interaction_summary(rows)
    assert result["complete_design"] is True
    assert result["estimate"] == pytest.approx(-5)
    assert result["ci_low"] == pytest.approx(-5)
    assert result["ci_high"] == pytest.approx(-5)
    assert [
        r["sample_minus_direct"] for r in result["secondary_per_k"]
    ] == pytest.approx([1, -4])
    # Large task-specific nuisance cancels only if bootstrap retains policy pairing.
    for i, row in enumerate(rows):
        row["pseudo_regret"] += (i // 2) * 1000
    paired = e9.interaction_summary(rows)
    assert paired["ci_low"] == pytest.approx(-5)
    assert paired["ci_high"] == pytest.approx(-5)
    assert e9.interaction_summary(list(reversed(rows))) == paired


def test_interaction_bootstrap_is_nondegenerate_and_matches_declared_rng():
    rows = synthetic_rows()
    values = {}
    for k in (2, 10):
        values[k] = np.arange(100) / (10 if k == 2 else -5)
    for row in rows:
        if row["policy"] == "bayes_sample":
            i = int(row["episode_id"].split(":")[-1])
            row["pseudo_regret"] = 10 + i / 100 + values[row["k"]][i]
    draws = {}
    for k in (2, 10):
        rng = np.random.default_rng(
            e9.stable_seed(e9.SEED, e9.EXPERIMENT, "interaction_bootstrap", k)
        )
        # Analysis sorts scientific IDs lexically before resampling.
        ordered = values[k][
            sorted(range(100), key=lambda i: f"{e9.EXPERIMENT}:prior:{k}:{i}")
        ]
        draws[k] = ordered[rng.integers(100, size=(10000, 100))].mean(axis=1)
    low, high = np.quantile(draws[10] - draws[2], [0.025, 0.975])
    result = e9.interaction_summary(rows)
    assert result["ci_low"] == pytest.approx(low)
    assert result["ci_high"] == pytest.approx(high)
    assert result["ci_low"] < result["estimate"] < result["ci_high"]


def test_missing_pairs_incomplete_and_duplicate_records_rejected():
    rows = synthetic_rows()
    result = e9.interaction_summary(rows[:-1])
    assert result["complete_design"] is False
    assert result["analysis_status"] == "incomplete; descriptive only"
    assert result["secondary_per_k"][1]["missing_pairs"] == 1
    assert result["secondary_per_k"][1]["completed_direct"] == 100
    assert result["secondary_per_k"][1]["completed_sample"] == 99
    assert e9.interaction_summary([])["estimate"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        e9.interaction_summary(rows + rows[:1])
