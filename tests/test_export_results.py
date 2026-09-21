import csv
import gzip
import json
import sqlite3
import zlib

import pytest

from jevbandits.export_results import export, write_csv_gz


def test_export_preserves_large_seed_and_drops_trace(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    record = {
        "episode_id": "x",
        "policy": "p",
        "seed": 2**64 - 1,
        "theta": [0.25, 0.75],
        "trace": [{"reward": 1, "action": 0}],
        "pseudo_regret": 0.125,
    }
    (source / "episodes_jev.jsonl").write_text(json.dumps(record) + "\n")
    result = export(source, output)
    assert result["exports"]["episodes"]["rows"] == 1
    with gzip.open(output / "episodes.csv.gz", "rt") as file:
        row = next(csv.DictReader(file))
    assert row["seed"] == str(2**64 - 1)
    assert json.loads(row["theta"]) == [0.25, 0.75]
    assert "trace" not in row
    assert json.loads(row["actions"]) == [0]
    assert json.loads(row["rewards"]) == [1]
    assert float(row["pseudo_regret"]) == 0.125


def test_gzip_is_deterministic(tmp_path):
    first = write_csv_gz([{"x": 1}], tmp_path / "a.gz")
    second = write_csv_gz([{"x": 1}], tmp_path / "b.gz")
    assert first["sha256"] == second["sha256"]


def test_duplicate_episodes_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    row = json.dumps({"episode_id": "x", "policy": "p"}) + "\n"
    (source / "episodes_jev.jsonl").write_text(row)
    (source / "episodes_index.jsonl").write_text(row)
    with pytest.raises(ValueError, match="Duplicate"):
        export(source, tmp_path / "output")


def test_ledger_export_whitelists_fields(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with sqlite3.connect(source / "ledger.sqlite") as db:
        db.execute("CREATE TABLE decisions(id TEXT,record BLOB)")
        record = {
            "action": 0,
            "probabilities": [0.6, 0.4],
            "raw_answer": {
                "type": "choice",
                "choice": "a",
                "confidence": 0.2,
                "probabilities": {"a": 0.6, "b": 0.4},
            },
            "headers": {"authorization": "not-for-release"},
        }
        db.execute(
            "INSERT INTO decisions VALUES(?,?)",
            ("test", zlib.compress(json.dumps(record).encode())),
        )
    result = export(source, tmp_path / "output")
    assert result["exports"]["decision_distributions"]["rows"] == 1
    with gzip.open(tmp_path / "output" / "decision_distributions.csv.gz", "rt") as file:
        text = file.read()
    assert "not-for-release" not in text
    assert "backend_confidence" in text
