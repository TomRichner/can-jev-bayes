# E8: fresh online evidence ladder

Version 1, specified before collecting any E8 responses. This is a posthoc follow-up motivated by the completed E2 fixed-state experiment, with fresh held-out tasks for the online test. It is not a revision of the original five experiments and does not select prompts from E4 outcomes.

E2 observed 100% terminal optimal agreement with posterior means versus 91.5% with counts alone; at horizon 10, mean assistance achieved 93.5% versus 92.25% with full Bayesian summaries and 100% with exact action values. These designed fixed-state results motivate two questions: does supplying arithmetic help on adaptive trajectories, and do uncertainty summaries improve on supplying means alone? These observations do not establish either online effect.

## Frozen design

- Experiment and seed namespace: `e8_evidence_online`, using the existing `stable_seed(20260920, experiment, family, K, episode)` environment construction.
- Four cells: K = 3 or 10 crossed with close (.55 versus .50) or prior (independent Beta(1,1)) environments. The model is never told its family.
- Horizon 100; 80 independent paired tasks per cell, totaling 320 tasks.
- Three direct Jev policies: `counts_direct`, `means_direct`, `bayes_direct`. Each has the same exploration-benefit objective from the frozen original prompt module. This yields 960 complete Jev episodes and 96,000 decisions.
- Counts alone retain each arm's successes and failures. Means add the Beta(1,1) posterior mean. Full Bayesian summaries add posterior parameters, standard deviation, and 95% equal-tailed credible interval. Every condition retains counts and knows the same likelihood and prior; summaries are deterministic functions of that condition's own observations. No condition receives extra data or another learner's history. The means condition also retains the existing summary-definition sentence; this is a representation package, not a literal one-field causal manipulation.
- “Direct” means the valid backend choice, including recorded disagreements with the reported probability maximum. There is no sampled variant, prompt contest, selected stopping rule, or choice substitution on failures.
- Common random numbers share the nth potential outcome of each physical arm on its nth pull, not the outcome at a common global turn. Anonymous labels and winner randomization follow the existing environment. Experiment, policy-action, and job-order namespaces remain separate.

The same 320 evaluator-only worlds support uniform random, posterior-mean greedy, Thompson sampling, Bayes-UCB, UCB1, knowledge gradient, Monte Carlo IDS (2,048 posterior draws), and the finite-horizon average-productivity index. Bayes-UCB uses q_t = 1 - 1/t. The finite index uses the existing float64 calibration at tolerance 1e-6; midpoint ties are uniform within absolute 1e-12 and ambiguous distinct-state brackets are recorded. It is a single-arm stopping-index heuristic, not the joint Bayesian optimum. There is no exact joint reference at horizon 100. The offline panel contains 2,560 baseline episodes.

## Analysis fixed before collection

The endpoint is each episode's cumulative pseudo-regret, sum_t(max_a theta_a - theta_action_t). Negative left-minus-right differences favor the first policy. There are exactly two overall primary contrasts:

1. `means_direct` minus `counts_direct`.
2. `bayes_direct` minus `means_direct`.

For each contrast, compute paired episode differences within each of the four cells and average the four cell means with equal weight. Use 10,000 percentile bootstrap resamples of whole paired episodes independently within each cell, preserving both policies' outcomes for each sampled task. Resample all 80 tasks per cell with replacement, recompute each cell mean, then average cells equally. Use deterministic reporting seeds derived from the experiment namespace and contrast. Neither turns nor API calls are independent replicates.

Report the point estimate and both the ordinary pointwise 95% interval (quantiles .025, .975) and a Bonferroni 97.5% interval for each of the two prespecified contrasts (quantiles .0125, .9875). The latter target simultaneous 95% coverage for this two-contrast family, subject to the bootstrap approximation; they do not cover all reported subgroup or baseline comparisons. Per-cell contrasts and every other comparison use exploratory pointwise 95% intervals. Do not choose the favorable interval, add primary comparisons after viewing results, or treat an interval crossing zero as equivalence.

Report completeness against 80 paired tasks per cell and distinguish missing left/right episodes from missing pairs. A pooled estimate from an incomplete design is descriptive only, with every missing cell and pair disclosed. Do not silently omit cells, replace tasks, or extend a favorable condition. Supplemental descriptive outcomes are reward, best-arm fraction, posterior-mean-greedy fraction, coverage, switching, and last-20-pull abandonment. All classical comparisons are exploratory. Hypotheses are informed by E2, while their evaluation uses fresh seeds; this does not prove broad superiority, infer hidden reasoning, or establish publication readiness.

## Reproducibility and execution gate

`python -m jevbandits.evidence_followup --prepare` freezes `artifacts/evidence_v1/manifest.json`, evaluator-only `tasks.json`, and representative `prompt_fixtures.json`, without constructing an API client or reading credentials. The manifest includes the full design and policy list, model `jev-1.13.0`, task/prompt hashes, dependency-lock hash, this protocol and its tests, its own source, and all scientific dependencies including the finite-index implementation. Task fixtures include seeds, hidden means, and potential-outcome hashes solely for evaluator reproducibility. Source hashes freeze the adaptive prompt generator; sample prompt fixtures are not the entire online trajectory.

`--run` verifies every frozen component before initializing the existing audited Jev client. It uses the durable SQLite response cache, question-hash validation, deterministic environment replay, and common outcome streams. Incomplete episodes are reconstructed from cached responses; already completed episode/policy pairs are skipped. Completed records use the original `episodes_jev.jsonl`, `episodes_baselines.jsonl`, and merged `episodes.jsonl` schema. Raw transport records and conservative cost reserves remain in the ledger. A source or fixture change requires a new namespace and amendment, not an in-place update.

The run is **not started by preparation**. Live E8 collection waits until E1–E5 and E6/E7 finish and the parent explicitly checks the remaining project budget. About $3 incremental cost is a rough planning estimate, not a guarantee. The default $18 cap includes all sibling run ledgers and conservative reserves through the original client. Maintain one live writer across sibling ledgers; do not purchase credits automatically. Offline execution is separate: `python -m jevbandits.evidence_followup --run --only-baselines`. Long baseline work also waits for the monitoring handoff. Preparation and mock-only tests are permitted before that gate.

The protocol freezes the full N. If credits, time, or errors prevent completion, preserve cached data and report the unfinished design; do not silently redefine the sample size.
