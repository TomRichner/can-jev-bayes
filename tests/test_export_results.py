import csv
import gzip
import json

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
        "trace": [{"reward": 1}],
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
