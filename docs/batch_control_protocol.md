# Batch-control validation audit, version 1

This is a small quality check on HTTP batching, question scoping, and content
layout. E2/E6 advice-pointer results could depend on the interface. The original
batch preflight used ordinary Bayesian questions and did not cover all advice
formats. This audit is not a new primary experiment, a claim about an internal
mechanism, or a publication-level significance test. It does not modify E6–E9.

## Selection and exact inputs

Use namespace `batch_control` and directory `artifacts/batch_control_v1`.
Regenerate E6 using `advice_followup.fixture_panel/build_design`, verify the
prepared `artifacts/followup_advice_v1` manifest, and select fixture indices 0–9
in **each** cohort (`random`, `targeted`). Selection never reads Jev responses.
Only horizon 10 is included. Cross 20 fixtures with six formats, two label
schemes, two assignments, and both existing repeats: **960 source questions**.
Evaluate each once alone and once in a mixed request: **1,920 decisions**.
The source question object, model, and shared state are exactly identical in
both modes. API transport outer keys are `q0` alone and `q0`…`q15` mixed; these
are documented question identifiers, not question content. Scientific cache IDs
include the audit namespace, mode, and complete E6 source ID; these IDs are not
sent as model instructions.

The six formats are nested, path_explicit, inline, per_option, advice_only, and
dp_values. Both original arm IDs and A/B labels, cohort, assignment, repeat, and
source experiment are retained in diagnostics. Scoring copies the source context,
sets `experiment=batch_control`, and reuses `advice_followup.score`. Explicit
adherence analyses include only `advice_present=true`; dp_values remains a
positive numerical control with optimality and probability outcomes.

## Fixed order, requests, and checkpoints

Sort the source IDs, shuffle with `stable_seed(SEED, 'batch_control', 'execution')`,
and split into 60 fixed blocks of 16. Shuffle an exactly balanced vector of 30
alone-first and 30 mixed-first blocks with the same generator. Each block runs
both modes consecutively. Alone mode sends 16 separate one-question HTTP requests;
mixed mode sends one 16-question request. The planned total is 1,020 successful
HTTP requests before retries. One shared JevClient changes `batch_size` between
1 and 16 only at awaited mode-block boundaries. Rate limiting, concurrency,
reserves, replay cache, and global accounting remain shared. The original
question order within a block is identical in both modes. Random mixing is fixed,
not optimized against model results.

Preparation and reporting are offline. Running requires a matching prepared
manifest freezing source code, tests, protocol, dependencies and lock file,
source E6 manifest and fixtures, ordered jobs and contexts, and every question
hash. It also writes reviewable jobs, contexts, and fixtures. A changed scientific
input requires a new namespace. Only the runner's three new files may be edited;
frozen experiments are not altered.

Pause only between intact 16-decision mode-blocks. `--max-decisions` limits newly
scored rows and never changes planned N or repartitions mixed requests. A resume
replays saved decisions. An interruption during mixed-response unpacking replays
the already durable complete original response, recovering any missing records
without shrinking the request. A partial cache without a saved response fails.
Derived diagnostics can be reconstructed from the ledger without another API
call. Preserve the ledger and namespace on every resume.

Use the project cap of **$18**, including previous sibling ledgers and uncertain
reserves. One live writer is allowed across the project; root orchestration must
wait until E9 collection finishes before starting this audit. Do not run while
another experiment is collecting. Expected incremental spend is under $0.10,
subject to actual service accounting; there is no separate authorized spend cap
increase. Do not purchase credits. Stop on budget, input, or answer validation
failure. This build and its tests make no live API calls.

```sh
PYTHONPATH=src .venv/bin/python -m jevbandits.batch_control --prepare
# Only after E9 is finished and the project budget is checked:
PYTHONPATH=src .venv/bin/python -m jevbandits.batch_control --run --max-decisions 32
PYTHONPATH=src .venv/bin/python -m jevbandits.batch_control --run
PYTHONPATH=src .venv/bin/python -m jevbandits.batch_control --report
```

## Descriptive outcomes and uncertainty

Primary descriptive outcome is total variation of the canonical two-arm
probability vectors for each matched source question:
`TV = 0.5 * sum(abs(p_mixed - p_alone))`. Average the matched values over formats,
label schemes, assignments, and repeats within fixture, then average fixtures.
Report cohorts separately and also each cohort × format. Report probability TV,
backend-choice adherence and optimality differences, probability assigned to the
advised arm, and expected exact loss differences. Signed differences are always
mixed minus alone. Backend choices remain distinct from probability argmax.

Service nondeterminism contributes to matched TV. The two existing repeats permit
secondary TV between the mean probability vectors in each mode, plus within-mode
repeat TV. Neither is a noise-corrected causal estimate or proof of invariance.
Counterbalancing reduces broad time/order confounding but cannot eliminate service
changes. Fixed randomly mixed neighbors are themselves part of this audit.

Bootstrap whole fixture IDs within cohort, 10,000 deterministic resamples, with
95% percentile intervals. There are just ten fixtures per cohort. Labels,
assignments, formats, and repeats are not independent tasks. Only fixtures with
all expected matched cells for a reported metric contribute to that estimate;
report completed/planned decisions, matched questions, and contributing fixture
counts. An incomplete audit remains explicitly incomplete. Bootstrap intervals
condition on the fixed grouping and do not capture cross-fixture dependence from
shared mixed HTTP requests.

Report exploratory format-minus-nested contrasts of within-fixture mode effects
(mode × format), and analogous TV contrasts, in `report.json`. Format and cohort
summaries are also rendered in `report.md`. Unadjusted intervals are descriptive;
do not produce strong significance headlines, equivalence claims, or generalized
mechanism conclusions. Targeted fixtures are selected on exact exploration
crossover, and random fixtures follow E6's designed count mixture. This small
audit qualifies interpretations of interface robustness, not overall ability.
