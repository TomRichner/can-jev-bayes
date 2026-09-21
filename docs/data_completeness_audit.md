# Independent data completeness audit

Snapshot: 2026-09-21T06:42:27.059415+00:00.

This offline audit reconstructs expected decision and episode identities directly from frozen manifests and fixture labels. It reads JSONL strictly without repairing files, checks traces and paired worlds, compares ledger decision IDs, verifies scientific source hashes, and validates compressed export row counts and hashes. It makes no API calls.

| Run | Expected / observed API decisions | Authoritative episodes | Duplicates / missing IDs | Status |
|---|---:|---:|---:|---|
| overnight_v2 | 393,160 / 393,160 | 13,537 | 0 / 0 | PASS |
| followup_advice_v1 | 28,800 / 28,800 | 0 | 0 / 0 | PASS |
| forecast_v1 | 24,000 / 24,000 | 0 | 0 / 0 | PASS |
| evidence_v1 | 96,000 / 96,000 | 3,520 | 0 / 0 | PASS |
| sampling_v1 | 40,000 / 40,000 | 2,100 | 0 / 0 | PASS |
| batch_control_v1 | 1,920 / 1,920 | 0 | 0 / 0 | PASS |
| primitive_v1 | 16,560 / 16,560 | 0 | 0 / 0 | PASS |

The main API count includes 200 preflight questions and 2,160 pilot decisions; the five evaluation experiments alone contain 390,800 decisions. Pilot/preflight records are not independent evaluation tasks. E6 contains 28,800 fixed-state decisions, E7 24,000 forecasts, E8 96,000 online decisions, E9 40,000 online decisions, E10 16,560 forecasts, and the batch control 1,920 questions. All are complete.

Manifest-derived evaluation totals: 1,560 independent online environments; 5,960 Jev trajectories; 516,000 online actions, excluding pilot. Completed evaluation/control questions total 598,080, matching the 598,080 planned IDs. Including revised pilot/preflight gives 600,440 cached questions, matching 600,440 expected IDs. Archived development is excluded from these scientific totals.

E10 adds an independent fixture audit: all 90 fixture identities and their frozen hash, seeded counts, analytic next-reward means, target-to-reference mappings, probability ranges, and stored common binary Brier scores pass. Primary Noul best-arm scores are rechecked without normalizing their marginal vectors. Best-arm quadrature values match the frozen fixture hash and valid probability constraints; this audit does not independently rerun quadrature.

The original main and E9 episodes.jsonl files are core-only merged views. Supplemental finite-index and exact-horizon baseline files are authoritative separate outputs and are included by export_results. Their absence from the core merged file is not missing experiment data. Trace lengths, turn order, valid actions/rewards, summed rewards/pseudo-regret, and matching seeds/hidden means across policies are checked. This structural audit does not rederive every reward stream or every numerical reference value.

## Costs counted once

| Ledger | Known cost (USD) | Retained reserves (USD) | HTTP attempts |
|---|---:|---:|---:|
| overnight_v2 | 12.489054732 | 0.008257536 | 24,628 |
| followup_advice_v1 | 0.441406224 | 0.000000000 | 1,800 |
| forecast_v1 | 0.253560216 | 0.000000000 | 1,500 |
| evidence_v1 | 2.566184292 | 0.000000000 | 6,000 |
| sampling_v1 | 1.448456478 | 0.000000000 | 2,500 |
| batch_control_v1 | 0.043193808 | 0.000000000 | 1,020 |
| primitive_v1 | 0.793554888 | 0.000000000 | 1,067 |
| overnight_v1 (archived development) | 0.055884864 | 0.000000000 | 162 |

Included ledger totals: **$18.091295502 known cost**, **$0.008257536 reserves**, **$18.099553038 combined**. Adding the separate $0.000154560 preparation allowance gives **$18.099707598**. Each attempts table is summed once; cumulative previous-run totals in accounting.json are never added. Reserves are uncertain potential charges, not verified billing.

All completed project runs, final controls, and archived overnight_v1 development spending are included in the cost scope. Archived overnight_v1 is cost-only and does not enter evaluation completeness. No controls remain pending in this snapshot.

## Exports and reproduction

Completed follow-up exports are in results/followup_advice_v1, results/forecast_v1, results/evidence_v1, results/sampling_v1, results/batch_control_v1, and results/primitive_v1. The existing results/overnight_v2 export is validated in place. E9 interaction_summary.json is retained with a provenance hash. The batch-control report is copied to reports/batch_control_v1. Compressed CSV rows preserve decision IDs, probabilities, and compact episode trajectories; raw HTTP ledgers remain local. Source scientific hashes match the frozen manifest when a run is marked PASS. Exported accounting files may contain cumulative totals and must not be summed across runs.

Run `PYTHONPATH=src .venv/bin/python -m jevbandits.data_audit` from the repository root to reproduce results/data_audit.json and this document. The JSON includes per-file and per-cell/policy counts, missing/duplicate examples, source-hash checks, SQLite integrity results, HTTP status counts, and export checks.
