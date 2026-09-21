import json
import sqlite3

import pytest

from jevbandits.data_audit import (
    compare,
    ledger_audit,
    online_expected,
    primitive_audit,
    records,
)


def test_exact_identity_audit_detects_same_count_wrong_data():
    result = compare({"a", "b", "c"}, ["a", "a", "d"])
    assert result["observed"] == result["expected"] == 3
    assert result["missing"] == 2
    assert result["unexpected"] == 1
    assert result["duplicate_extra_rows"] == 1
    assert not result["passed"]


def test_parser_never_repairs_truncated_source(tmp_path):
    path = tmp_path / "data.jsonl"
    content = '{"ok":1}\n{"torn":'
    path.write_text(content)
    with pytest.raises(ValueError, match="Malformed JSONL"):
        list(records(path))
    assert path.read_text() == content


def test_cost_uses_ledger_rows_not_cumulative_accounting(tmp_path):
    path = tmp_path / "ledger.sqlite"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE attempts(cost REAL,reserved REAL,input_tokens INTEGER,status TEXT)"
        )
        db.execute("CREATE TABLE decisions(id TEXT)")
        db.executemany(
            "INSERT INTO attempts VALUES(?,?,?,?)",
            [(1, 0, 10, "200"), (0, 0.1, 0, "pending")],
        )
        db.execute("INSERT INTO decisions VALUES('a')")
    (tmp_path / "accounting.json").write_text(
        json.dumps({"project_cost_and_reserves_usd": 99})
    )
    result = ledger_audit(path, {"a"})
    assert result["known_cost_usd"] == 1
    assert result["cost_plus_reserves_usd"] == 1.1
    assert result["decision_ids"]["passed"]


def test_expected_online_ids_are_manifest_derived():
    manifest = {
        "online": {
            "x": {
                "families": ["prior"],
                "ks": [3],
                "n": 2,
                "horizon": 4,
                "policies": ["direct", "sample"],
            }
        },
        "classical": ["random"],
    }
    jev, classical, decisions = online_expected(manifest)
    assert len(jev) == 4 and len(classical) == 2 and len(decisions) == 16
    assert "x:prior:3:1:sample:4" in decisions


def test_e10_fixture_and_unnormalized_score_audit(tmp_path, monkeypatch):
    from jevbandits import primitive_followup as e10

    monkeypatch.setattr(e10, "N", 1)
    monkeypatch.setattr(e10, "KS", (2,))
    panel = e10.fixture_panel()
    jobs, contexts = e10.build_design(panel)
    manifest = {
        "master_seed": e10.SEED,
        "experiment": e10.EXPERIMENT,
        "ks": [2],
        "fixtures_per_k": 1,
        "representations": e10.REPRESENTATIONS,
        "repeats": 2,
        "fixture_sha256": e10.digest(panel),
    }
    rows = []
    for job, context in zip(jobs, contexts, strict=True):
        p = [0.7, 0.3] if context["arm"] is None else [0.8, 0.2]
        rows.append(
            e10.score(
                context,
                {
                    "probabilities": p,
                    "raw_answer": {},
                    "probability_mass": 1,
                    "request_id": job["id"],
                },
            )
        )
    (tmp_path / "fixtures.json").write_text(json.dumps(panel))
    output = tmp_path / "forecasts.jsonl"
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    expected, fixture_check, score_check = primitive_audit(tmp_path, manifest)
    assert len(expected) == 56
    assert fixture_check["passed"] and score_check["passed"]
    rows[0]["mean_binary_excess_brier"] += 0.001
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _, _, score_check = primitive_audit(tmp_path, manifest)
    assert not score_check["passed"] and score_check["reference_score_errors"] == 1
