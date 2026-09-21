# E6: advice binding follow-up (pre-registered before E6 API collection)

This follow-up is motivated by completed E2 results: supplied means and exact
action values supported perfect terminal decisions, whereas the nested optimal
recommendation condition achieved 89.25% optimal agreement at horizon 10 and
failed on six strict exploration fixtures. A manually inspected failure correctly
recommended `arm_01` in the payload but Jev chose `arm_00`. These observations
motivate the hypothesis that resolving advice pointers, option labels, or option
criteria can improve adherence separately from numerical planning. They do not
identify an internal mechanism. No E6 responses have been collected at registration.

## Fixed design and selection

Namespace: `e6_advice_binding`; run directory: `artifacts/followup_advice_v1`.
Use the existing pinned `jev-1.13.0` backend and unchanged shared task definition.
All intervention content is local to each question.

There are 200 fresh fixtures, with no Jev-response-based selection:

- 100 random states from `fixtures('e6_advice_random', 100)`. Each arm's count is
  drawn from {0,2,5,10,20}, followed by successes uniformly from 0 through its
  count. Conditional on count this is the Beta(1,1) prior predictive law;
  the count distribution is designed, not sampled from policy trajectories.
- 100 distinct targeted states from the stable seed
  `stable_seed(SEED, 'e6_advice_binding', 'targeted')`. Draw incumbent count
  uniformly from integers 10 through 80 and successes uniformly from integers
  yielding posterior means in [.51,.68]; the other arm has zero observations.
  Reject duplicate states and states without exact horizon-10 action-value
  advantage greater than 1e-4 for the lower-mean unobserved arm. Record the
  accepted draw number. A fixed 100,000-draw guard fails rather than adapting.

The targeted cohort is conditioned on exact-value crossover selection and is not
representative of all posterior states. Results must be reported separately by
cohort; no natural-population generalization follows from pooled performance.

Cross all fixtures with horizons 1, 2, 10; six formats; original `arm_00`/`arm_01`
and `A`/`B` labels; both physical-to-display assignments; and two independent API
repeats: **28,800 decisions**. Assignment changes the physical arm attached to
each display position, not the ordering of display IDs. Randomize execution order
once with `stable_seed(SEED, 'e6_advice_binding', 'execution')` and retain it on
resume. Scientific IDs include cohort, fixture, horizon, format, label scheme,
assignment, and repeat.

## Six frozen question formats

1. `nested`: original E2 recommendation payload and instruction.
2. `path_explicit`: add an explicit instruction to read
   `observation.recommended_arm` and select the criterion with that exact ID.
3. `inline`: add `Choose EXACTLY <ID>.` to the instruction.
4. `per_option`: mark the recommended option and nonrecommended option directly
   in their respective criteria, retaining the original recommendation instruction.
5. `advice_only`: retain the original instruction, arm IDs, horizon, and nested
   recommended ID, while removing counts and all posterior statistics.
6. `dp_values`: existing E2 exact-action-value payload and exploration instruction
   as the positive control, with no explicit recommended ID in the payload.

The first four retain counts, means, Beta parameters, standard deviations, and
95% credible intervals; `dp_values` retains these and exact Q values. All IDs,
criteria, and advice pointers are consistently relabeled. These are bundled
communication interventions; they do not isolate arithmetic as a causal factor.

Compute unrounded exact Q values for the same fixture and horizon. Define gold
advice as the lowest canonical arm index within 1e-10 of the maximum. Relabel that
same gold action for each assignment. Store gold advice for `dp_values` for a
common descriptive comparator, but flag `advice_present=false`: following a
particular advice pointer is not an intervention in that condition.

## Outcomes and analysis commitments

Primary outcome is chosen-ID advice adherence (the backend's valid `choice`).
Keep it distinct from tie-aware optimal-action agreement, exact local Q loss,
and probability-distribution expected Q loss. Choosing a different optimal arm
can have zero loss and optimal agreement while failing chosen-ID adherence.
Retain probability mass, raw answer, and backend-choice versus probability-argmax
gap; never replace backend choice with argmax. Also retain probability assigned
to the advised arm, posterior-mean greediness, and answer entropy.

Primary descriptive contrasts are each of path-explicit, inline, and per-option
against nested at horizon 10 within each cohort, paired within fixtures. Report
advice-only and exact-value comparisons and all horizon/label interactions as
secondary exploratory contrasts. Average assignments/repeats within fixture,
retain label schemes as declared comparison factors, and bootstrap whole fixture
IDs within cohort with 10,000 resamples and 95% percentile exploratory intervals.
Do not treat repeated calls, label assignments, or horizons as independent tasks.
No uncorrected formal significance headline or equivalence claim is planned.
Report missing/completed cells and selected-cohort limitations.

## Freeze, execution, and monitoring

`--prepare` is a separate offline-only step; `--run` requires an existing matching
manifest and frozen fixtures. The manifest records seeds, full panel and ordered
question hashes, model, shared/task instructions, and hashes of the new runner,
tests, protocol, original prompt/client/experiment/exact-method sources, and lock
file. Scientific source or payload changes require a new namespace and amendment.
Git commit is provenance, not a requirement that later unrelated commits stop a run.
The runner never tunes prompts, fixtures, or format allocation from responses.

After independent review and authorization of this build:

```sh
PYTHONPATH=src .venv/bin/python -m jevbandits.advice_followup --prepare
PYTHONPATH=src .venv/bin/python -m jevbandits.advice_followup --run --max-decisions 80
PYTHONPATH=src .venv/bin/python -m jevbandits.advice_followup --run
```

The first 80 decisions fit five 16-question requests and are part of the frozen
evaluation, not a prompt-selection pilot. Inspect mapping, validation, accounting,
and completion before handing monitoring to Sol. `--max-decisions` only pauses
the fixed permutation; it cannot redefine the sample. There is one live API writer
across all project ledgers. Reuse the existing SQLite replay cache, bounded retries,
batching, concurrency, and shared $18 cap, including previous run ledgers and
uncertain reserves. Stop on budget or validation failure without policy substitution.
No automatic credit purchase. Resume checkpoints without repeated successful calls.

Write compatible `diagnostics.jsonl` rows with additional `format`, `cohort`,
`label_scheme`, and adherence fields. Reporting changes are allowed after the freeze;
scientific input changes are not. The build and mock tests do not initiate API calls.
