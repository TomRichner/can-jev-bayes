"""Mock-only integration checks for API safety and scientific replay."""

import asyncio
import copy
import json

import httpx
import numpy as np
import pytest

from jevbandits import experiments
from jevbandits.client import MAX_INPUT, PRICE, BudgetExceeded, JevClient
from jevbandits.experiments import Episode, diagnostics, environment, online, read_jsonl
from jevbandits.prompts import MODEL, TASK, observation, question, validate_answer


def answer_for(q, choice=0, probs=None):
    ids = list(q["criteria"])
    probabilities = (
        probs
        if probs is not None
        else [1.0 if i == choice else 0.0 for i in range(len(ids))]
    )
    return {
        "type": "choice",
        "choice": ids[choice],
        "probabilities": dict(zip(ids, probabilities)),
        "confidence": 0.95,
    }


def response_for(request, *, choice=0, tokens=100):
    payload = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "model": MODEL,
            "usage": {"input_tokens": tokens},
            "answers": {
                key: answer_for(q, choice) for key, q in payload["questions"].items()
            },
        },
    )


def jobs(n=1):
    return [
        {"id": f"id-{i}", "question": question(observation([i, 0], [0, i], 2))}
        for i in range(n)
    ]


def client(tmp_path, handler, **kwargs):
    return JevClient(
        tmp_path,
        key="test-secret",
        transport=httpx.MockTransport(handler),
        rate=1e9,
        **kwargs,
    )


def test_batch_state_contains_only_shared_model_and_question_local_observations(
    tmp_path,
):
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["state"] == TASK
        assert "arms" not in payload["state"]
        for q in payload["questions"].values():
            assert "observation" in q["instructions"]
            assert set(q["criteria"]) == {"arm_00", "arm_01"}
        return response_for(request)

    async def run():
        c = client(tmp_path, handler, batch_size=2)
        try:
            result = await c.evaluate(jobs(5))
            assert len(result) == 5
            assert c.accounting()["http_requests"] == 3
            assert c.accounting()["inference_decisions"] == 5
            assert c.accounting()["known_cost_usd"] == pytest.approx(300 * PRICE)
            assert c.accounting()["uncertain_reserved_usd"] == 0
        finally:
            await c.close()

    asyncio.run(run())
    assert sorted(len(p["questions"]) for p in requests) == [1, 2, 2]
    assert "test-secret" not in (tmp_path / "accounting.json").read_text()


def test_durable_response_replay_never_reissues_successful_decision(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return response_for(request, choice=1)

    async def run():
        first = client(tmp_path, handler)
        original = await first.evaluate(jobs(3))
        await first.close()
        second = client(tmp_path, handler)
        try:
            replayed = await second.evaluate(jobs(3))
            assert replayed == original
            assert second.accounting()["http_requests"] == 1
        finally:
            await second.close()

    asyncio.run(run())
    assert calls == 1


def test_resume_rejects_changed_question_with_reused_decision_id(tmp_path):
    async def run():
        c = client(tmp_path, response_for)
        try:
            original = jobs()
            await c.evaluate(original)
            altered = copy.deepcopy(original)
            altered[0]["question"]["instructions"]["observation"][
                "remaining_pulls_including_this_one"
            ] = 9
            with pytest.raises(ValueError):
                await c.evaluate(altered)
            assert c.accounting()["http_requests"] == 1
        finally:
            await c.close()

    asyncio.run(run())


def test_budget_reserves_inflight_requests_before_network(tmp_path):
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return response_for(request, tokens=MAX_INPUT)

    async def run():
        reserve = MAX_INPUT * PRICE
        c = client(
            tmp_path,
            handler,
            cap=reserve + 0.00015456 + 1e-8,
            batch_size=1,
            concurrency=4,
        )
        try:
            with pytest.raises(BudgetExceeded):
                await c.evaluate(jobs(4))
            assert calls == 1
            budget = c.accounting()
            assert (
                budget["known_cost_usd"] + budget["uncertain_reserved_usd"] + 0.00015456
                <= c.cap
            )
        finally:
            await c.close()

    asyncio.run(run())


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_nonretryable_http_error_stops_without_policy_substitution(tmp_path, status):
    async def run():
        c = client(tmp_path, lambda request: httpx.Response(status))
        try:
            with pytest.raises(RuntimeError, match="Non-retryable"):
                await c.evaluate(jobs())
            assert c.accounting()["http_requests"] == 1
            assert c.accounting()["inference_decisions"] == 0
            assert c.accounting()["uncertain_reserved_usd"] > 0
        finally:
            await c.close()

    asyncio.run(run())


@pytest.mark.parametrize("failure", [429, 529, 500, "timeout"])
def test_retry_retains_uncertain_charge_and_records_success(
    tmp_path, monkeypatch, failure
):
    calls = 0
    waits = []

    async def no_sleep(delay):
        waits.append(delay)

    monkeypatch.setattr("jevbandits.client.asyncio.sleep", no_sleep)

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("mock timeout")
            return httpx.Response(failure, headers={"Retry-After": "2"})
        return response_for(request)

    async def run():
        c = client(tmp_path, handler)
        try:
            assert len(await c.evaluate(jobs())) == 1
            budget = c.accounting()
            assert budget["http_requests"] == 2
            assert budget["error_attempts"] == 1
            assert budget["uncertain_reserved_usd"] == MAX_INPUT * PRICE
            assert budget["known_cost_usd"] == 100 * PRICE
        finally:
            await c.close()

    asyncio.run(run())
    assert calls == 2
    assert max(waits) >= (1 if failure == "timeout" else 2)


@pytest.mark.parametrize("kind", ["model", "labels", "choice", "usage"])
def test_malformed_success_response_stops_before_any_action(tmp_path, kind):
    def handler(request):
        response = response_for(request)
        body = response.json()
        if kind == "model":
            body["model"] = "different-model"
        elif kind == "labels":
            body["answers"] = {"wrong": body["answers"]["q0"]}
        elif kind == "choice":
            body["answers"]["q0"]["choice"] = "hidden-arm"
        else:
            body.pop("usage")
        return httpx.Response(200, json=body)

    async def run():
        c = client(tmp_path, handler)
        try:
            with pytest.raises(ValueError):
                await c.evaluate(jobs())
            assert c.accounting()["inference_decisions"] == 0
            assert c.accounting()["http_requests"] == 1
        finally:
            await c.close()

    asyncio.run(run())


def test_probability_normalization_preserves_raw_answer_and_zero_mass():
    q = question(observation([0, 0, 0], [0, 0, 0], 1))
    answer = answer_for(q, 0, [0.50, 0.49, 0.0])
    original = copy.deepcopy(answer)
    action, probability, mass = validate_answer(answer, list(q["criteria"]))
    assert action == 0
    assert mass == 0.99
    assert probability.sum() == pytest.approx(1)
    assert probability[-1] == 0
    assert answer == original


@pytest.mark.parametrize(
    "probabilities",
    [[0, 0], [0.4, 0.4], [1.1, -0.1], [float("nan"), 1], [float("inf"), 0]],
)
def test_invalid_probabilities_rejected(probabilities):
    q = question(observation([0, 0], [0, 0], 1))
    with pytest.raises(ValueError):
        validate_answer(answer_for(q, 0, probabilities), list(q["criteria"]))


def test_direct_choice_must_be_argmax_and_confidence_finite():
    q = question(observation([0, 0], [0, 0], 1))
    with pytest.raises(ValueError, match="argmax"):
        validate_answer(answer_for(q, 0, [0.2, 0.8]), list(q["criteria"]))
    answer = answer_for(q)
    answer["confidence"] = float("nan")
    with pytest.raises(ValueError, match="confidence"):
        validate_answer(answer, list(q["criteria"]))


def test_observation_bayesian_summaries_and_label_reversal():
    obs = observation(
        [10, 0], [8, 0], 2, "dp_values", q_values=[1.1, 1.10833333], label_order=[1, 0]
    )
    first, second = obs["arms"]
    assert first["posterior_mean"] == 0.5
    assert second["posterior_mean"] == 0.55
    assert first["successes"] == 0
    assert second["successes"] == 10
    assert (
        first["expected_total_reward_if_chosen_then_optimal"]
        > second["expected_total_reward_if_chosen_then_optimal"]
    )
    rec = observation(
        [10, 0],
        [8, 0],
        2,
        "recommendation",
        q_values=[1.1, 1.10833333],
        label_order=[1, 0],
    )
    assert rec["recommended_arm"] == "arm_00"


@pytest.mark.parametrize(
    "successes,failures,remaining",
    [([-1, 0], [0, 0], 1), ([1], [0, 0], 1), ([0, 0], [0, 0], 0)],
)
def test_bad_counts_or_horizon_rejected(successes, failures, remaining):
    with pytest.raises(ValueError):
        observation(successes, failures, remaining)


@pytest.mark.parametrize(
    "successes,failures,remaining",
    [
        ([1.5, 0], [0, 0], 1),
        ([0, 0], [0, 0.25], 1),
        ([0, 0], [0, 0], 1.5),
        ([], [], 1),
    ],
)
def test_fractional_counts_or_horizon_and_empty_arm_set_rejected(
    successes, failures, remaining
):
    with pytest.raises(ValueError):
        observation(successes, failures, remaining)


def test_episode_public_question_independent_of_hidden_means_and_outcomes():
    ep = Episode("pilot", "prior", 3, 7, 8, "bayes_direct")
    original = ep.job(1)
    ep.theta[:] = [0.999, 0.888, 0.777]
    ep.outcomes[:] = 1
    assert ep.job(1) == original
    encoded = json.dumps(original)
    assert (
        "theta" not in encoded and "outcomes" not in encoded and "family" not in encoded
    )


def test_potential_outcomes_paired_by_arm_pull_count_not_global_turn():
    one = Episode("pilot", "prior", 2, 1, 5, "counts_direct")
    two = Episode("pilot", "prior", 2, 1, 5, "bayes_direct")
    np.testing.assert_array_equal(one.outcomes, two.outcomes)
    one.step(1, 0)
    one.step(2, 0)
    two.step(1, 1)
    two.step(2, 0)
    two.step(3, 0)
    assert [r["reward"] for r in one.trace] == [r["reward"] for r in two.trace[1:]]


def test_pseudoregret_reward_counts_and_exact_loss_match_known_crossover():
    ep = Episode("e3_exact_online", "prior", 2, 2, 2, "counts_direct")
    ep.s[:] = [10, 0]
    ep.f[:] = [8, 0]
    ep.theta[:] = [0.55, 0.5]
    # Step uses arm-indexed streams; expand only for this deliberately pre-observed state.
    ep.outcomes = np.ones((2, 20), dtype=int)
    ep.step(1, 0)
    assert ep.trace[0]["exact_loss"] == pytest.approx(1 / 120)
    assert ep.trace[0]["pseudo_regret"] == 0
    assert ep.trace[0]["optimal_agreement"] is False
    assert ep.trace[0]["posterior_mean_greedy"] is True
    assert ep.s.tolist() == [11, 0]
    ep.step(2, 1)
    result = ep.summary()
    assert result["reward"] == 2
    assert result["pseudo_regret"] == pytest.approx(0.05)
    assert result["switches"] == 1
    assert result["arms_visited"] == 2
    assert result["best_arm_fraction"] == 0.5


def test_environment_namespace_separates_experiments_and_is_reproducible():
    first = environment("pilot", "prior", 3, 0, 10)
    second = environment("pilot", "prior", 3, 0, 10)
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], second[2])
    assert first[0] != environment("e4_scaling", "prior", 3, 0, 10)[0]


class MemoryClient:
    """Mimic durable decisions while avoiding network and accounting dependencies."""

    def __init__(self, path, fail_call=None):
        self.run_dir = path
        path.mkdir(parents=True, exist_ok=True)
        self.records = {}
        self.calls = 0
        self.fail_call = fail_call

    async def evaluate(self, submitted):
        self.calls += 1
        if self.calls == self.fail_call:
            raise RuntimeError("intentional interruption")
        for job in submitted:
            self.records.setdefault(
                job["id"],
                {
                    "action": 0,
                    "probabilities": [0.6, 0.4],
                    "probability_mass": 1.0,
                    "request_id": job["id"],
                    "raw_answer": {"confidence": 0.6},
                },
            )
        return [self.records[job["id"]] for job in submitted]

    def accounting(self):
        return {"known_cost_usd": 0.0}


def test_online_interruption_replays_identical_sampled_actions_and_rewards(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(
        experiments.DESIGNS,
        "pilot",
        {
            "ks": [2],
            "families": ["prior"],
            "n": 2,
            "horizon": 4,
            "policies": ["counts_direct", "counts_sample"],
        },
    )
    resumed = MemoryClient(tmp_path / "resumed", fail_call=3)
    uninterrupted = MemoryClient(tmp_path / "uninterrupted")

    async def run():
        with pytest.raises(RuntimeError, match="intentional"):
            await online(resumed, resumed.run_dir, "pilot")
        assert len(resumed.records) == 8
        resumed.fail_call = None
        await online(resumed, resumed.run_dir, "pilot")
        await online(uninterrupted, uninterrupted.run_dir, "pilot")
        assert read_jsonl(resumed.run_dir / "episodes_jev.jsonl") == read_jsonl(
            uninterrupted.run_dir / "episodes_jev.jsonl"
        )
        assert len(resumed.records) == 16
        # A completed episode is skipped on subsequent invocation.
        previous = read_jsonl(resumed.run_dir / "episodes_jev.jsonl")
        await online(resumed, resumed.run_dir, "pilot")
        assert read_jsonl(resumed.run_dir / "episodes_jev.jsonl") == previous

    asyncio.run(run())


def test_diagnostic_reversal_maps_probabilities_and_choices_to_physical_arms(tmp_path):
    c = MemoryClient(tmp_path)
    asyncio.run(diagnostics(c, "e1_horizon", n=2))
    result = read_jsonl(tmp_path / "diagnostics.jsonl")
    assert len(result) == 2 * 4 * 3 * 2 * 2
    for row in result:
        assert row["canonical_action"] == row["label_order"]
        assert row["probabilities"] == (
            [0.6, 0.4] if row["label_order"] == 0 else [0.4, 0.6]
        )
        assert row["exact_loss"] == pytest.approx(
            max(row["q_values"]) - row["q_values"][row["canonical_action"]]
        )


def test_torn_final_jsonl_record_is_removed_before_resume(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    path.write_text('{"ok":1}\n{"torn":')
    assert read_jsonl(path) == [{"ok": 1}]
    assert path.read_text() == '{"ok":1}\n'
