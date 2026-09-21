# E9: confirm the sampling-by-arm-count interaction on fresh tasks

Preregistered before any E9 API collection. Completed E4 motivates this single
confirmatory contrast: on prior-family tasks at T=100, mean pseudo-regret for
Bayesian-summary sampling minus direct choice was +1.036, +0.155, -3.672, -4.825,
and -4.001 for K=2,3,5,10,15 respectively. E3 at K=2,T=20 gave +0.400. These
discovery results motivate an independent interaction test, not a posthoc claim
that a particular K is a universal crossover threshold. E9 is the final planned
confirmation of this pattern; its allocation does not adapt to observed outcomes.

## Fixed design

Use namespace `e9_sampling_regimes` and `artifacts/sampling_v1`. Draw 100 independent
tasks at each of K=2 and K=10, all with fixed horizon T=100. Each task has independent
Beta(1,1) arm success probabilities and the matching Beta(1,1) working prior.
The K strata are independently drawn, not paired to each other. There are 200
tasks, 400 Jev episodes, and **40,000 API decisions**.

Compare only existing `bayes_direct` and `bayes_sample`, using exactly the existing
Bayesian statistics and exploration prompt. Direct uses the valid backend choice,
including when it disagrees with the reported probability maximum. Sampling draws
once from the normalized reported categorical distribution using the existing
per-policy, per-turn deterministic RNG. It is not interpreted as Thompson sampling.
No prompt, global policy registry, or original design mutation is permitted.

Reuse `Episode` and `environment` directly: seeds separate experiment/family/K/task;
policies share each physical arm's nth potential reward on that arm's nth pull;
action RNG is separate and keyed by policy and turn. Hidden means and potential
rewards never enter prompts. Preserve question-hash-validated SQLite replay and
per-episode traces. Remaining pulls include the current decision. There is one
backend decision for each policy and turn, with no API repeats or prompt selection.

Include eight offline comparators on all 200 paired tasks: random, posterior-mean
greedy, Thompson sampling, Bayes-UCB, UCB1, knowledge gradient, Monte Carlo IDS,
and the finite-horizon average-productivity index heuristic. Keep existing methods,
IDS sample count, tie rules, and AP tolerance 1e-6; AP is not jointly optimal.
These 1,600 comparator episodes are secondary context and cost no Jev credits.

## One primary inferential contrast

For each task, calculate paired episode pseudo-regret difference
`d = sample - direct`. The sole primary contrast is

`mean(d at K=10) - mean(d at K=2)`.

The directional hypothesis is negative. Report a **two-sided 95% percentile
bootstrap interval**, using 10,000 resamples of whole paired tasks independently
within each K stratum, then subtracting the resampled cell means. Do not pool
turns as independent samples, pair unrelated tasks across K, or pool policies
before making within-task differences. Bootstrap seeds are
`stable_seed(SEED, 'e9_sampling_regimes', 'interaction_bootstrap', K)`.

Report each K-specific mean sample-minus-direct difference with a secondary,
pointwise 95% interval from those same bootstrap draws. The primary interaction
does not alone prove opposite signs in the two cells; discuss those secondary
estimates separately. A negative interaction with both effects having the same
sign is an interaction result, not confirmation of a sign reversal. No formal
K-threshold, equivalence, or additional multiplicity-unadjusted primary claims.

K changes together with the difficulty of identifying a good arm under a fixed
100-pull budget and the number of input options. This design confirms an
operational interaction, not a pure mechanistic effect of arm count. It applies
to matched-prior tasks and this model/prompt/budget, not all reward families.

Record complete episodes, missing policies, and missing pairs at each K. Retain
the full N=100 allocation when budget or time interrupts collection. Report
incomplete paired-subset estimates as descriptive only, never as the completed
confirmatory experiment; do not stop for significance or silently reduce N.
Reward, coverage, switching, greediness, best-arm fraction, abandonment, and
backend choice mismatch remain secondary descriptive outcomes in episode records.

## Freeze and execution

Freeze this protocol, the E9 module/tests, original prompt/client/experiment/
baseline/index sources, dependency lock, seed, design, evaluator-world hashes,
and representative public-payload hashes before collection. New scientific source
or parameter changes require a new namespace and amendment. Unrelated git commits
may follow the recorded provenance commit without modifying frozen inputs.

Preparation and planned reporting are offline and create no client. The builder
may prepare after passing mock tests and committing. Root conducts the live spend
gate after E1-E8: estimated incremental cost approximately $1.50, sharing the $18
project cap with all sibling ledgers, earlier charges, and conservative reserves.
This estimate is not an authorization to exceed the cap or buy credits. Only one
live project writer is allowed; offline baselines may run separately under Sol.

```sh
PYTHONPATH=src .venv/bin/python -m jevbandits.sampling_followup --prepare
PYTHONPATH=src .venv/bin/python -m jevbandits.sampling_followup --run --only-baselines
# Only after root's existing-project spending gate:
PYTHONPATH=src .venv/bin/python -m jevbandits.sampling_followup --run
PYTHONPATH=src .venv/bin/python -m jevbandits.sampling_followup --report
```

The CLI exposes no N, K, horizon, or policy overrides. Resume via the same `--run`.
`--report` writes the declared primary contrast, secondary effects, and completion
status to `interaction_summary.json`. Missing data must remain explicit.
