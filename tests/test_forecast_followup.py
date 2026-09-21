import asyncio
import json

import httpx
import numpy as np
import pytest

from jevbandits.forecast_followup import (
    ForecastClient,
    best_probabilities,
    build_jobs,
    scoring,
)
from jevbandits.prompts import MODEL


@pytest.mark.parametrize("k", [2, 3, 5, 10, 15])
def test_exchangeable_posteriors(k):
    assert np.allclose(
        best_probabilities([3] * k, [7] * k), np.full(k, 1 / k), atol=1e-9
    )


def test_beta_two_one_vs_uniform_and_permutation():
    assert np.allclose(best_probabilities([1, 0], [0, 0]), [2 / 3, 1 / 3], atol=1e-9)
    assert np.allclose(best_probabilities([0, 1], [0, 0]), [1 / 3, 2 / 3], atol=1e-9)


def test_scoring_is_excess_proper_risk():
    exact = scoring([0.6, 0.4], [0.6, 0.4])
    assert exact["excess_brier"] == pytest.approx(0)
    assert exact["excess_log_loss"] == pytest.approx(0)
    wrong = scoring([0.6, 0.4], [0.8, 0.2])
    assert wrong["excess_brier"] == pytest.approx(0.08)
    assert wrong["excess_log_loss"] == pytest.approx(
        0.6 * np.log(0.6 / 0.8) + 0.4 * np.log(0.4 / 0.2)
    )


def test_zero_probabilities_are_counted_and_log_clipping_explicit():
    result = scoring([0.5, 0.5], [1.0, 0.0])
    assert result["zero_prediction_positive_reference"] == 1
    assert np.isfinite(result["excess_log_loss"])
    assert result["excess_brier"] == 0.5


def test_job_design_and_single_arm_forecast_payloads():
    jobs, metadata = build_jobs(n=1)
    assert len(jobs) == 240
    assert len({j["id"] for j in jobs}) == 240
    for job, record in zip(jobs, metadata, strict=True):
        assert job["id"] == record["decision_id"]
        obs = job["question"]["instructions"]["observation"]
        assert "theta" not in json.dumps(obs)
        if record["target"] == "next_reward":
            assert job["question"]["type"] == "noul"
            assert set(obs) == {"arm"}
            arm = obs["arm"]
            assert record["reference"][0] == pytest.approx(
                (arm["successes"] + 1) / (arm["successes"] + arm["failures"] + 2)
            )
        else:
            assert len(job["question"]["criteria"]) == record["k"]


def test_mixed_noul_choice_response_and_replay(tmp_path):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        answers = {}
        for key, q in body["questions"].items():
            if q["type"] == "noul":
                answers[key] = {"type": "noul", "noul": 0.6}
            else:
                names = list(q["criteria"])
                answers[key] = {
                    "type": "choice",
                    "choice": names[0],
                    "probabilities": {n: 1 / len(names) for n in names},
                    "confidence": 0.0,
                }
        return httpx.Response(
            200,
            json={"model": MODEL, "answers": answers, "usage": {"input_tokens": 100}},
        )

    async def run():
        jobs, _ = build_jobs(n=1)
        selected = [jobs[0], jobs[1]]
        client = ForecastClient(
            tmp_path,
            key="test-secret",
            transport=httpx.MockTransport(handler),
            rate=1000,
        )
        try:
            first = await client.evaluate(selected)
            second = await client.evaluate(selected)
            assert first == second
            assert first[1]["probabilities"] == [0.6, 0.4]
            assert len(calls) == 1
        finally:
            await client.close()

    asyncio.run(run())


def test_invalid_noul_rejected(tmp_path):
    client = ForecastClient(tmp_path, key="test-secret")
    jobs, _ = build_jobs(n=1)
    with pytest.raises(ValueError, match="Noul"):
        client._unpack(
            [jobs[1]],
            {"model": MODEL, "answers": {"q0": {"type": "noul", "noul": 1.1}}},
            "test",
            0,
        )
    asyncio.run(client.close())
