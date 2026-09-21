# Overnight execution log

All timestamps below are UTC unless stated otherwise. Work began September 20 evening CDT; the user clarified the deadline as September 21, 07:00 CDT. Changes and findings are recorded separately from the frozen evaluation protocol.

## Preparation and selected readings

- Read selected sections of the cleaned Krishnamurthy, Sun, Kaufmann, Thompson-sampling, and IDS manuscripts. The design review distinguishes information presentation, external computation, and exploration control.
- Implemented exact Beta-Bernoulli baselines, full-state DP, knowledge gradient, and 2,048-sample IDS, with rational/analytic cross-checks.
- Built isolated-question HTTP batching, compressed raw-response persistence, deterministic episode replay, strict public/evaluator data separation, and paired/clustered reporting.
- Initial runner validation: 100 tests passed. Commits preserve the scientific methods before live evaluation.

## Development run v1: observed interface discrepancy

The first 80-decision preflight used four target questions, ten repeats, alone and in four-question mixed batches. It passed the predeclared .10 probability-shift gate; largest observed shift .027. It cost $0.00201852.

The pilot stopped at turn 16 on a valid backend choice assigned .44 probability when another arm had .45. This was the only such mismatch in the 162 successful HTTP responses saved at that point. Pilot plus preflight used 1,330,592 input tokens, estimated $0.055884864, with no transport/status failures. The strict argmax assertion exposed a mismatch with the documented response semantics; it was not a reward-performance exclusion.

Retain the complete v1 ledger. The amended v2 protocol preserves backend choice, records the gap from the reported maximum, and does not silently execute host argmax. All scientific evaluation begins after this amendment. Budget accounting now includes prior sibling run namespaces, so the abandoned pilot is not hidden from spending totals.

## Revised run v2: preflight and pilot gate

The expanded preflight tests full 16-question batches: four target questions plus twelve unrelated decision contexts, with ten repeats and reversed batch ordering. It used 200 decisions across 50 requests, 94,060 input tokens, and estimated $0.00395052. Maximum target probability shifts were 0, .007, .003, and .018. This is evidence of no large shift on these fixtures, not a universal proof of batch independence.

The revised pilot completed all 108 episodes (nine task cells, four policies, three replicates each), 20 pulls per episode. At completion the v2 ledger held 2,360 decisions and 1,699,989 input tokens for $0.071399538, including preflight. No API errors or backend-choice discrepancies occurred in this revised pilot. Combined with v1 and the original preparation calls, estimated project usage was $0.127438962.

The budget gate regressed billed request tokens against question count and summed serialized question length, then projected the frozen matrix using dense late-posterior summaries. This is an engineering estimate, not a probabilistic cost interval. Projected online costs: E3 $0.30, E4 $7.46, E5 $5.09; with a 20% margin and $0.50 fixed-panel allowance, approximately $15.91. The $18 project cap remains active. Shared-request costs cannot be assigned exactly to individual questions; the earlier equal-share per-question diagnostic is not an estimate of each representation's intrinsic token cost.

Pilot mean pseudo-regrets (horizon 20): counts/direct 3.441, summaries/direct 3.076, counts/sample 2.995, summaries/sample 2.687. These are development checks from only three episodes per cell, not scientific conclusions. Sample sizes and prompts were not changed in response. The fixed N remains an exploratory effect-size design; the pilot is too small and short to power a definitive main-study superiority claim.

Gate decision: proceed with all five predefined experiments, preserving the full cell matrix and sample sizes.

## Offline baseline provenance

All 7,669 planned classical episodes completed before the v2 response-handling change: pilot 189; E3 1,600; E4 4,200; E5 1,680. Baseline policies, environmental seeds, and scientific inputs were unchanged, so the completed baseline JSONL was copied to v2 rather than simulated again.

SHA-256 of that artifact: `fc2a126e631950eb3f163fdc37afccd85ed771629184b25686cbd69d76e0f3bb`.

An independent analytic policy-value audit is underway to explain finite-horizon baseline behavior. In 200 E3 tasks, empirical greedy regret is below empirical exact-policy regret; this cannot establish superiority to the Bayesian optimum. Exact local Bellman loss is zero for the exact policy. Prior-integrated values will separate sampling variability from true expected policy gaps.

## Reporting discipline

Scientific changes after evaluation starts require a new experiment/run amendment and fresh evaluation tasks where appropriate. Report generation may improve without changing data. Numerical results from later stages will be recorded in the scientific report rather than silently rewriting this development history.

## E4 operational interruption and recovery

During automatic retry of a transport failure near turn 20, the monitoring worker misinterpreted an instruction to report errors as requiring interruption. It sent Ctrl-C; the client had not exhausted its retries. The same frozen run resumed from its recorded decisions, without changing prompts, tasks, sample sizes, or rewards. Pending/possibly billed attempts remain in the ledger and spending reserve. The monitor was instructed to leave ordinary bounded retries running and stop only on an actual terminal error, cap, or scientific validation exception. This is an operational deviation, not a reason to exclude any episode.

## Offline index and numerical-audit completion

The finite AP-index simulation completed 1,040 episodes: E3 200, E4 600, E5 240. One process ended during a worker handoff and resumed without duplicates. Five episodes had one bracket-overlap flag; four were terminal decisions, where exact ties can cause overlap. An overlap is not by itself numerical instability. The raw artifact SHA-256 is `ee99ad8ca5582edb282ef1eeceb5cf026deef991affd92512c8b6454ef37f16c`.

The IDS audit completed 200 posterior fixtures with 30 replicates at each of 128, 2,048, and 32,768 posterior samples. The 95th percentile excess true information ratio was 0.015571849, 0.00086055281, and 0.000054793577 respectively; the largest at 2,048 was 0.074298205. These describe approximation quality on this two-arm panel, not online regret or a guarantee for larger arm counts. Quadrature error estimates were below $8.33\times10^{-12}$. No Jev credits were used.

## Five-study completion and fresh follow-ups

All five original experiments completed, with 390,800 evaluation decisions plus 200 revised preflight questions and 2,160 revised pilot decisions. The v2 ledger records $12.489054732 in known usage, with conservative reserves retained for the interrupted/retried requests. No evaluation task was dropped or replaced. Core results and compact trajectory/probability exports were committed in `e14fdaa`.

After prospective protocol preparation, E6 completed 28,800 advice-format decisions, E7 completed 24,000 probability forecasts, E8 completed 96,000 adaptive evidence-ladder decisions, and E9 completed 40,000 adaptive sampling decisions. All four follow-up ledgers have zero failed attempts. Final reports distinguish newly specified primary contrasts from exploratory subgroup analyses.

## Final controls and operational spending amendment

The final-control cap amendment preceded all batch-control and E10 calls. It raises the operational cap to $19 while retaining the previously frozen questions and sample sizes. The batching CLI retained its original maximum-$18 argument guard, so the first `--cap 19` command exited before creating a request. Rather than changing frozen scientific source, execution used the existing `run(args)` entry point with the amended cap:

```python
import asyncio
from argparse import Namespace
from jevbandits.batch_control import run

asyncio.run(run(Namespace(
    run_dir="artifacts/batch_control_v1",
    prepare=False, report=False, run=True,
    cap=19.0, rate=5.0, concurrency=4, max_decisions=None,
)))
```

The first 32 decisions used `max_decisions=32`; an independent payload audit checked the frozen source/fixture hashes, exact expected question hashes, request question multisets, and task-only shared state. It confirmed sixteen standalone requests and one sixteen-question request. The complete batch validation then recorded 1,920 decisions across 1,020 requests for $0.043193808, with no errors. Its gross advice-format pattern persisted in standalone questions on the small reused fixture panel.

E10 similarly paused after its first 80 questions. The payload/hash audit passed; null `arm` entries occurred only for the joint best-arm Choice question, as required by its schema. The unchanged full run resumed under the amended cap. One Sol medium worker remained the sole API writer; offline auditing and interpretation proceeded separately.

E10 completed at 2026-09-21 06:38:56 UTC (01:38:56 CDT): all 16,560 questions, 1,067 requests, 18,894,164 input tokens, and $0.793554888 in estimated usage, with zero failed attempts. The final project estimate including all development usage, preparation, and conservative reserves is **$18.099707598**, below the amended $19 cap and the reported $20 credits. API collection stopped after this complete design; no additional credits were purchased. Final audit/export and interpretation followed offline.

Final independent data audit: all 598,080 evaluation/control questions complete; 600,440 cached questions including the revised pilot/preflight but excluding the archived development run. There are 1,560 independent online environments, 5,960 evaluation Jev trajectories, and 516,000 online actions. No expected IDs are missing or duplicated. Scientific hashes, paired environments, trace consistency, E10 reference/score calculations, SQLite checks, and export counts/hashes passed. The final full suite passed 319 tests in 25.21 seconds; Ruff and `git diff --check` were clean.
