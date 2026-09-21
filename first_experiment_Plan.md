# Five experiments on Jev, Bayesian assistance, and sequential control

Revised 2026-09-20/21 CDT, before evaluation. The user authorized implementation, execution, analysis, and follow-up experiments through 07:00 CDT September 21. Reasoning-model harnesses, real-money gambling, and post-training optimization are outside this series. This supersedes the [archived proposal](docs/initial_experiment_proposal.md).

## Aim and literature-driven revisions

Characterize where a typed decision model benefits from explicit objectives, Bayesian summaries, and external calculations. Separate numerical inference, using supplied evidence, valuing information, and following advice. Do not presume that Jev can or cannot perform Bayesian decision-making.

The [selected literature review](docs/literature_design_review.md) motivated three changes: use Kaufmann's experimental Bayes-UCB schedule $q_t=1-1/t$; distinguish abandonment from persistent undirected exploration; and include knowledge gradient and Monte Carlo information-directed sampling (IDS). Sampling Jev's answers is not automatically Thompson sampling.

## Frozen five-experiment design

| ID | Question | Design | Jev decisions |
|---|---|---|---:|
| E1 horizon/framing | Does horizon and wording change choices in the direction of exact Bayesian action values? | 100 two-arm fixtures; horizons 1/2/5/10; neutral/exploration/regret; two label assignments; two independent API repeats | 4,800 |
| E2 assistance | Does accessible arithmetic, uncertainty, or optimal advice help? | 100 new fixtures; horizons 1/2/10; five evidence levels; two assignments; two repeats | 6,000 |
| E3 exact online | Do fixed-state abilities translate into adaptive behavior? | 200 paired prior-drawn two-arm tasks; horizon 20; four core Jev policies plus exact-value-assisted Jev | 20,000 |
| E4 online scaling | How do representation and host sampling interact with arm count and difficulty? | 2/3/5/10/15 arms; clear/close/prior families; horizon 100; 40 episodes/cell; four core policies | 240,000 |
| E5 strategy interventions | Can a Bayesian instruction or supplied index improve held-out behavior? | 3/10 arms; close/prior families; horizon 100; 60 fresh episodes/cell; five interventions | 120,000 |

Total: **390,800 evaluation decisions**, plus 200 batching-preflight decisions and 2,160 pilot decisions in the revised run. The aborted development run is accounted separately. Each decision is one question; independent questions may share an HTTP request. The executable run manifest records every prompt, generator, seed, method, scientific source hash, and dependency-lock hash before data collection.

E1 includes six illustrative posterior pairs: Beta(1,1)/Beta(1,1), Beta(11,9)/Beta(1,1), Beta(19,3)/Beta(1,1), Beta(10,10)/Beta(1,1), Beta(2,1)/Beta(60,40), Beta(1,2)/Beta(6,4). Its other 94 fixtures and all 100 E2 fixtures independently draw each arm's observation count from {0,2,5,10,20}, then successes uniformly from zero through that count. These designed posterior-state panels are not samples from a natural policy's trajectory. No fixture is selected based on Jev's answer or a favorable crossover.

E2 retains counts at every level: counts alone; posterior means; means plus Beta parameters, standard deviation, and 95% equal-tailed credible intervals; those summaries plus exact action values; and summaries plus an exact optimal recommendation with an explicit instruction to follow it. The last two test external-computation use, not autonomous planning. E2 recommendation uses a stronger instruction as part of the bundled advice intervention; it is not an isolated information-only causal contrast.

Core policies: counts/direct choice, Bayesian summaries/direct choice, counts/categorical sampling, Bayesian summaries/categorical sampling. All use the same exploration-benefit instruction. E3 adds direct choice with exact values. E5 fixes its interventions now: Bayesian summaries with exploration, neutral, regret, or Thompson-strategy wording (all direct choice), plus Bayes-UCB indices with an instruction to choose the maximum. E5 is not a post-hoc best-prompt contest. Later discovered prompts require a new namespace and amendment.

## Environments, baselines, and normative references

Rewards are independent Bernoulli observations with fixed unknown means. All learners know the likelihood and independent Beta(1,1) working prior. Families: one .70 arm versus .30 alternatives; one .55 arm versus .50 alternatives; independent Beta(1,1)-drawn means. Learners are not told their family. Only the last is a matched-prior Bayesian benchmark.

Seeds separate experiment/family/arm-count/episode, actions, and reporting. Policies share each physical arm's $n$th potential outcome on its $n$th pull. Winning labels are randomized and anonymous. Jev receives only its own counts and functions of those counts; hidden means, future outcomes, and other episodes' histories remain evaluator-only. Remaining pulls include the current action. Reward maximization versus expected cumulative-regret minimization is a framing contrast, not a different mathematical objective.

Classical comparisons on the same tasks:

- Thompson sampling with exact Beta draws.
- Bayes-UCB at quantile $1-1/t$, first decision $t=1$; its initial zero indices tie uniformly. This reproduces the paper's practical $c=0$, not the $c\ge5$ theorem setting.
- Posterior-mean greedy, uniform random, and UCB1 with its own unobserved-arm initialization.
- Knowledge gradient $m_a+(h-1)\operatorname{KG}(a)$, exactly integrating one further Bernoulli observation. This is a longer-horizon approximation.
- Monte Carlo IDS with 2,048 joint posterior samples per decision, conditional Bernoulli entropy, and optimal two-arm mixture search. Rare best-arm categories may be missed; sample-count sensitivity is a candidate follow-up.
- Exact full-state dynamic programming in E3. No finite-horizon index approximation is called exact.

The exact reference is

$$
V_0(s)=0,\quad Q_h(s,a)=m_a+m_aV_{h-1}(s^{a,+})+(1-m_a)V_{h-1}(s^{a,-}),\quad V_h(s)=\max_aQ_h(s,a).
$$

For Beta(11,9) versus Beta(1,1), one pull favors the .55-mean arm; two pulls favor the unobserved arm ($Q_A=1.10$, $Q_B=1.108\overline3$). Tests verify this and compare DP with rational exhaustive enumeration. Bayesian method details and references are in [research_notes.md](research_notes.md); the original files' user edits are preserved.

## Analysis

E1/E2: exact loss $V_h(s)-Q_h(s,A)$, tie-aware optimal-action agreement, and expected one-decision loss of the normalized answer distribution. Stratify by horizon; average repeats/assignments within a fixture, then bootstrap whole fixtures. Describe the designed-state population and separate illustrative fixtures when needed.

Online primary endpoint is episode pseudo-regret:

$$\bar R_T=\sum_{t=1}^{T}(\theta^*-\theta_{A_t}).$$

Also report reward, arm coverage, switching, posterior-mean-greedy fraction, best-arm fraction, and last-20-pull abandonment. E3 includes cumulative local Bellman loss and agreement. E5 advice adherence uses indices from the recipient policy's own evidence, not a classical policy's different trajectory.

Use paired episode effects with 10,000 percentile bootstrap resamples and 95% exploratory intervals. Pool cells equally and resample within cells, retaining per-cell effects. Repeated calls, labels, and turns are not independent tasks. Report missing pairs/incomplete episodes. A CI crossing zero is not equivalence. Primary contrasts: summaries versus counts, sampling versus direct at matched representation, Jev versus TS, and E5 interventions versus exploration-worded summaries. Additional comparisons are exploratory; formal headline hypothesis tests would need declared multiplicity correction.

Interpret observable behavior, not unobserved reasoning. Action probabilities, predictive success probabilities, and posterior best-arm probabilities are distinct. Report model/version/prompt/task scope and both favorable and unfavorable results.

## Execution and gates

Pin `jev-1.13.0`. Shared state contains only the task definition; each Choice question's structured instructions contain exactly one decision's observation. Batch at most 16 questions and conservatively limit serialized bytes. Ten repeats of four targets alone versus mixed/order-reversed questions test the documented independence mechanism; a probability shift above .10 stops for investigation. This preflight does not prove independence universally. Batching is reported as an interface choice.

Start at five HTTP requests/second and four concurrent requests; respect the documented token/request limits. A compressed SQLite ledger records raw payloads/responses, usage, retries, and hashes, never keys or headers. Same-decision replay requires a matching question hash. Source changes affecting scientific input require a new run namespace/amendment; reporting can improve without changing collected data. Use only one live API writer per project ledger; offline baselines can run in parallel.

Validate choice IDs, finite/ranged probabilities, model, and usage. Retain the backend's valid choice even when it is not the reported probability maximum; flag the gap and mismatch frequency. Normalize only small rounding discrepancies, retaining raw values. Bound retries and conservatively reserve unknown charges. Never substitute a classical action on API failure. Deterministic per-turn action RNGs and common reward streams allow checkpoint replay without duplicated rewards.

Pilot: 2/5/15 arms, all three families, three episodes/cell, horizon 20, four core policies. Exclude it from evaluation. Inspect tokens, failures, trajectory correctness, and variance before committing to the main run. Keep the frozen N unless an explicitly documented amendment precedes evaluation; do not silently delete cells or choose favorable seeds.

Budget: initially cap Jev at **$18**, including preparation, previous run namespaces, and conservative reserves. At $0.042/million input tokens, 393,160 decisions average $16.51 at 1,000 tokens/decision or $33.03 at 2,000, plus aborted development usage. Measured batched usage controls execution. The user authorizes up to $200 for the project but reports $20 current credits; do not automatically purchase credits. If spending/balance prevents completion, preserve balanced completed data and identify unfinished work. Raw records remain ignored under `artifacts/`; download/OCR files stay in ignored `pdfs/`.

### Pre-evaluation amendment: backend choice discrepancy

At pilot turn 16, one response selected arm_03 with probability .44 although arm_04 had .45. This was the only discrepancy in the 162 HTTP responses then saved. The strict argmax validator stopped the development run, after $0.055884864 including preflight. Preserve that ledger as `overnight_v1`. All scientific evaluation runs use `overnight_v2`, which accepts any valid returned choice and explicitly records its gap from the probability maximum. “Direct” means backend choice throughout; it is not silently replaced by host argmax. A future host-argmax condition requires its own online evaluation. The complete offline baselines are unchanged and may be copied with hashes/provenance. The preflight now includes the full 16-question batch size, with twelve unrelated distractors alongside four targets. All sibling ledger costs count toward the same cap.

## Deliverables and continuation

Commit tested implementation, protocol changes, aggregate CSVs, standalone plots, reproducibility metadata, and an interpreted Markdown report at logical milestones with the user's identity and no AI attribution. Raw data can support a later publication release audit. No claim of publication readiness follows merely from running five tests.

If the five experiments finish before 03:00 CDT September 21, specify further rigorous experiments from their findings using new seeds and recorded hypotheses. Potential directions: IDS sample-count sensitivity, horizon-aware policy corrections, probability-estimation tasks, and uncertainty-format ablations. By 07:00 CDT deliver the strongest completed evidence and remaining limitations. Sol medium handles monitoring/literature retrieval; Astra handles building and interpretation.

## Follow-ups specified after E1–E3, before collecting follow-up responses

The completed first three experiments motivated the following additional studies. They preserve the original five-study data and use new evaluation namespaces. Live collection waits until E1–E5 finish; E8 also waits for E6/E7 and an updated budget check. Preparations and offline numerical controls may run concurrently with the original studies.

| Study | Motivation and intervention | Planned Jev decisions | Protocol |
|---|---|---:|---|
| E6 advice binding | Correct action values were used reliably, while an optimal recommendation pointer was not. Compare six advice layouts, numeric/letter labels, and random versus deliberately exploration-requiring new fixtures. | 28,800 | [E6 protocol](docs/e6_advice_protocol.md) |
| E7 probability forecasts | Good action ranking does not establish Bayesian uncertainty. Ask Noul reward forecasts and Choice latent-best-arm questions; compare with analytic conditional probabilities. | 24,000 | [E7 protocol](docs/e7_forecast_protocol.md) |
| E8 online evidence ladder | Means alone sometimes beat fuller summaries in E2. Compare counts, means, and full summaries on 320 fresh adaptive tasks; two overall primary contrasts receive Bonferroni intervals. | 96,000 | [E8 protocol](docs/e8_evidence_protocol.md) |

Additional offline controls cost no Jev credits: the [exact prior policy-value audit](docs/theory_audit.md), a [finite-horizon AP-index comparator](docs/finite_index_implementation.md), and the [IDS Monte Carlo sensitivity audit](docs/ids_numerical_audit.md). The index is a documented posthoc comparator rather than an originally frozen primary policy. The IDS audit compares 128/2,048/32,768 samples with quadrature references on two-arm states; it does not establish identical approximation accuracy at 15 arms.

The first-three results and their interpretation are committed in [the interim report](reports/interim_first_three/interpretation.md). Follow-up hypotheses were recorded without changing E4/E5 prompts or sample sizes. All new live studies remain subject to the same project-wide cap, including previous and interrupted runs.
