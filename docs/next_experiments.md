# Research continuation after the overnight studies

Updated September 21, 2026. This plan responds to E1–E10 and the numerical/interface audits. It preserves the frozen first-study protocols and datasets. These are proposed next allocations, not completed experiments or permission requests. Reasoning-model harnesses remain out of scope for the present phase.

## Decision on the present evidence

The initial five experiments were followed by five focused studies because the early results left interpretable questions: recommendation binding (E6), probability fidelity (E7), the evidence ladder on fresh trajectories (E8), a sampling interaction (E9), and primitive/target confounding with supplied-probability controls (E10). Additional numerical audits checked exact finite-horizon values, the AP approximation, and IDS integration. The next useful step is replication and isolation of these effects, rather than further unstructured prompt search.

The [research report](../research_report.md) is the interpretation; the source protocols remain the authority for what was specified before each run. The existing data should be treated as development data for selecting the following hypotheses. Each new study needs fresh seeds, a versioned protocol, a small primary contrast family, and a sealed analysis script before calls begin.

## 1. Replicate the strongest adaptive contrasts

Repeat E8's means-versus-counts and full-summary-versus-means contrasts, E9's sampling interaction, and E10's two elicitation contrasts on another service date, recording the exact model identifier. Preserve the original tasks' distribution, horizon where applicable, and question text. Use fresh tasks, and report both replication effects and differences from the first wave. Do not pool across dates without showing their effects separately.

For planning, use the existing **paired task-level differences**, not the number of turns, to estimate variance. Simulate the planned stratified bootstrap under plausible effects and choose N to target a confidence-interval half-width of 0.5 reward units for the E8 overall contrasts and 1.5 units for the E9 interaction. Publish the chosen N before collection. A provisional N of 160 per E8 cell and 200 per E9 arm count would double the original samples: 192,000 and 80,000 questions, respectively. This is a precision starting point, not a demonstrated power calculation.

For E10, a provisional 60 fixtures per arm-count stratum preserves its full factorial design at 33,120 questions. Replicate its two primary contrasts and report coherence and known-probability controls again, without turning the current best format into the only retained condition.

At the measured first-wave rates, these E8/E9/E10 allocations project roughly $5.13, $2.90, and $1.59 in Jev input charges, respectively: **about $12.50 together with a 30% allowance**. Re-estimate from payload lengths and current documented pricing before running. Existing account credits may not cover this next wave; costs here describe a proposal, not an automatic purchase.

Primary estimands and multiplicity should mirror the original studies, with the replication interaction signed in advance. Count a changed model version as a transfer/robustness study rather than an exact service replication. An unchanged name does not establish unchanged backend weights, so preserve dates and request metadata.

## 2. Isolate why fuller summaries can hurt

The E8 intervention packages uncertainty, extra fields, and length together. Use a factorial ablation on counts plus posterior means:

- Add Beta parameters only.
- Add posterior standard deviation only.
- Add the equal-tailed interval only.
- Add all uncertainty fields, matching the existing full-summary format.
- Add length-matched irrelevant formatting or redundant mean/count text, with no new statistical quantity.

Develop the exact length-control construction without evaluation responses, then freeze it. Do not call a comparison “length matched” merely because its mean token count is similar; record per-question lengths and residual imbalance. Initially use exact two-arm fixtures with a balanced mix of greedy-optimal and strict-exploration states. Keep a separate randomly drawn fixture bank for population estimates. Subsequently validate selected hypotheses online on fresh 3/10-arm prior-drawn tasks.

The primary question is whether a specified uncertainty field changes exact decision loss or online regret relative to means alone. Adherence, attention to a field, or an inferred internal reasoning process is not directly observed. This design should also cross numeric versus letter labels where inexpensive, since E6 showed substantial label dependence under explicit lookup.

## 3. Treat advice use as a decision-interface problem

E6's numeric-values success and recommendation-pointer failure warrant a targeted experiment. Cross correct advice with agreement versus disagreement with posterior-mean greedy, while balancing the action-value gap and horizon. Compare an explicit action ID, a per-option recommendation marker, a complete Q-value vector, and a minimal instruction-only presentation. Include deterministic parsers/argmax as the appropriate zero-model baselines.

Add controlled advice errors in a separately specified condition: small perturbations that change the recommended action and larger errors that conflict with clearly superior reward. This asks whether Jev follows advice mechanically, rejects it usefully, or inconsistently combines sources. Score the actual chosen action against independently computed Q values. Incorrect advice must never be mixed silently into the correct-advice estimates. Include independently worded templates held out from prompt development.

The practical aim is to find when a model usefully arbitrates between evidence and an external algorithm. Perfect execution of a supplied answer alone does not add value over directly executing that answer.

## 4. Link probability fidelity to sequential reward

E7 and E10 separate predictive reward, latent-best events, and API elicitation. In E10, Noul was more faithful on both targets, but its full-summary best-arm probabilities summed to 1.466 on average at 15 arms. Even supplying exact probabilities did not make either primitive an exact numeric copier. Consequently the next experiment should evaluate **whether correcting the probability distribution actually improves the control policy**. On held-out tasks compare:

- Classical Thompson sampling from the exact Beta posterior.
- Sampling exact posterior probabilities of being best (an independently implemented TS-equivalent control, up to numerical error).
- Sampling Jev's explicitly elicited best-arm distribution.
- Sampling normalized separate Noul best-arm forecasts, with raw marginal coherence reported separately.
- Sampling a Jev forecast corrected by a calibration map fitted only on development fixtures.
- Direct Jev under the same evidence representation.

Choose the correction family on development data and freeze it. A simple temperature transform is a candidate, not an assumed solution; it may fail to repair ranking or joint-coherence errors. Report proper-scoring excess on held-out fixtures and online regret separately. Improved posterior fidelity need not improve finite-horizon reward, because Thompson sampling itself is not finite-horizon optimal.

Keep reward binary primitives under identical full-arm context. Compare separate Noul marginals with joint Choice on a common mean binary-risk scale, retain unnormalized Noul probabilities for fidelity evaluation, and explicitly report any normalization used to turn them into a policy. Include a cost and latency accounting by interface. Do not retrofit E7's single-arm reward questions into a supposedly identical E10 condition.

## 5. Transfer to Gaussian-process Bayesian optimization

After replication, use low-dimensional continuous optimization as a distinct transfer test before high-dimensional or model-training tasks. Start with known analytic objective functions, a fixed noisy-observation model, and 2/5/10 dimensions. Use common initial designs and equal evaluation budgets. Distinguish cumulative regret from simple regret of the final recommendation; the latter is usually the relevant endpoint for hyperparameter optimization.

Build a candidate-set decision interface so Jev chooses among exactly the same proposed points available to comparison selectors. Compare random selection, GP expected improvement, GP-UCB, Thompson sampling from the GP posterior, Jev supplied only past evaluations, and Jev supplied GP means/uncertainty or acquisition values. Separate candidate generation from selection: if a Bayesian optimizer proposes all promising points, credit that computation explicitly. Include a direct acquisition-argmax baseline and make inference cost visible.

The candidate restriction, GP kernel assumptions, hyperparameter fitting, normalization, initial design, observation noise, and duplicate-point policy all require an explicit protocol. Use synthetic objectives before optimizing an LLM's post-training hyperparameters. The latter adds noisy expensive evaluations, asynchronous execution, and difficult budget comparability, so it should not be inferred from bandit results.

## Publication gates

Before preparing a submission, obtain an independent code/data audit; validate a clean offline reproduction from the compact exports; verify references and exact API semantics; document service dates and all amendments; and repeat the central effects. Inspect classically competitive baselines at the intended horizon rather than assuming Thompson sampling is the strongest reference. Report negative and positive interventions together, along with the unsuccessful formats that motivated follow-ups.

A potential paper should center on the relationship between probability estimation, evidence presentation, and sequential execution. Claims about internal mechanisms, broad model families, reasoning-LLM collaboration, or practical hyperparameter optimization require additional evidence. No external submission is authorized or performed merely by this planning document.
