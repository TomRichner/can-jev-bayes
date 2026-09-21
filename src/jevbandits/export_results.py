"""Export compact synthetic evaluation records for offline reproducibility.

No credentials, HTTP headers, downloaded literature, or large request ledger are
copied. Seed integers are written as strings to avoid JSON/JavaScript precision loss.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path


def read_records(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def compact(record):
    result = {}
    for key, value in record.items():
        if key in {"trace", "raw_answer"}:
            continue
        if key == "seed":
            value = str(value)
        elif isinstance(value, (dict, list)):
            value = json.dumps(value, sort_keys=True, separators=(",", ":"))
        result[key] = value
    answer = record.get("raw_answer", {})
    for name in ("confidence", "choice", "noul"):
        if name in answer:
            result[f"backend_{name}"] = answer[name]
    if "probabilities" in answer:
        result["backend_probabilities"] = json.dumps(
            answer["probabilities"], sort_keys=True, separators=(",", ":")
        )
    return result


def write_csv_gz(records, path):
    records = [compact(r) for r in records]
    fields = sorted({field for record in records for field in record})
    with (
        path.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream,
    ):
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return {
        "rows": len(records),
        "fields": fields,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def export(run_dir, output_dir):
    source, output = Path(run_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    episodes = []
    # Read authoritative policy-specific outputs, not a possibly older merged view.
    for name in [
        "episodes_jev.jsonl",
        "episodes_baselines.jsonl",
        "episodes_index.jsonl",
    ]:
        episodes.extend(read_records(source / name))
    ids = [(r["episode_id"], r["policy"]) for r in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate episode/policy rows")
    groups = {
        "episodes": episodes,
        "diagnostics": read_records(source / "diagnostics.jsonl"),
        "forecasts": read_records(source / "forecasts.jsonl"),
    }
    manifest = {"source_run": str(source), "exports": {}, "provenance": {}}
    for name, records in groups.items():
        if not records:
            continue
        if name != "episodes":
            decisions = [r["decision_id"] for r in records]
            if len(decisions) != len(set(decisions)):
                raise ValueError(f"Duplicate {name} decision IDs")
        manifest["exports"][name] = write_csv_gz(records, output / f"{name}.csv.gz")
    for name in [
        "manifest.json",
        "forecast_manifest.json",
        "index_manifest.json",
        "baseline_provenance.json",
        "accounting.json",
    ]:
        path = source / name
        if path.exists():
            target = output / name
            target.write_bytes(path.read_bytes())
            manifest["provenance"][name] = hashlib.sha256(
                target.read_bytes()
            ).hexdigest()
    (output / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Recorded synthetic evaluation data\n\n"
        "These compressed CSVs reproduce reported episode/fixture analyses without new API calls. "
        "Read `seed` as a string or unsigned 64-bit integer; derive RNG seeds from the published generator when rerunning. "
        "Nested arrays/dictionaries are JSON strings. Losses, recommendations, and probabilities retain the recorded precision. "
        "Raw request/response ledgers and per-turn traces remain in the local ignored artifact directory; "
        "this compact release does not reproduce every latency or trajectory figure by itself.\n\n"
        "No external publication or upload is performed by this export. The source may be an interim snapshot; "
        "check row counts and protocol manifests before interpreting completeness. Accounting is run-wide and "
        "includes reserves and previous-run totals rather than verified account billing.\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    print(json.dumps(export(args.run_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
