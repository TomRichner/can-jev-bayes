# Archived initial proposal: Can Jev use Bayesian evidence to explore?

Superseded by [the revised protocol](../first_experiment_Plan.md); retained for provenance.

Date: 2026-09-20. **Proposal for discussion; the experiment has not been implemented or run.**

## 1. Research question and recommendation

Test whether Jev can choose actions that maximize **expected cumulative reward over a stated finite horizon**, how its behavior differs from Bayesian bandit policies, and whether externally computed Bayesian summaries help. Start with independent stationary Bernoulli arms: simple enough to calculate the posterior exactly, yet rich enough to expose exploration failures.

The main comparison is **Jev's direct choice under an explicit long-term objective**. Taking the highest-probability answer does not necessarily mean taking the arm with the highest immediate expected reward. The question defines what an answer means. An exploratory action can be Jev's highest-probability answer to a long-term reward question. Probability sampling is a separate ablation, not a prerequisite for purposeful exploration. The API defines `choice` as the highest-probability option. [API reference](https://docs.typesafe.ai/api)

Predefined questions:

1. How much cumulative reward does Jev lose relative to Thompson sampling and Bayes-UCB as the number of arms increases?
2. Do posterior means and uncertainty summaries help it choose better actions than successes/failures alone?
3. Does its choice change appropriately when learning has more time to pay off?
4. Does sampling its answer distribution improve outcomes, or merely increase randomness?
5. Does it prematurely abandon promising arms, respond to uncertainty, or exhibit label/order bias?

Treat the repository's current “Can Jev Bayes? No.” tagline as a hypothesis, not a finding. Neither this plan nor the API smoke tests establish the answer.

## 2. Setup and feasibility already completed

- Created `.venv` using Python 3.12.9, with dependencies declared in `pyproject.toml` and resolved in `uv.lock`.
- Installed NumPy, SciPy, pandas, Matplotlib, HTTPX, python-dotenv, Pydantic, pytest, and Ruff. Scientific imports and a Beta posterior smoke check passed. No bandit framework or heavyweight GP stack is necessary yet.
- Loaded `jev_key` from `.env` with python-dotenv without printing its value. Used the direct official API only.
- One model-list request and four inference requests returned HTTP 200 using pinned model `jev-1.13.0`. Total reported input usage: 3,680 tokens; estimated inference cost: **$0.00015456**.
- Local request/response records and the one-off probe script are in ignored `artifacts/`. Downloaded papers and OCR outputs are in ignored `pdfs/`. Neither directory should be committed.

To recreate the environment:

```bash
uv sync --frozen --python 3.12
source .venv/bin/activate
```

In this agent's restricted environment, the working invocation also used `UV_CACHE_DIR=/private/tmp/can-jev-bayes-uv-cache`; package downloads required network permission. This is an execution-environment detail, not a requirement for the research.

Two live API observations must influence the harness: repeated identical inputs changed probabilities slightly, and one 15-arm response summed to 0.99. Preserve raw responses; do not assume deterministic outputs or an exactly normalized distribution. Details are in [research_notes.md](../research_notes.md).

## 3. Experimental design

### 3.1 Common environment and information boundary

At pull $t$, action $A_t$ receives $Y_t \sim \operatorname{Bernoulli}(\theta_{A_t})$. Reward probabilities stay fixed within an episode. Every reward has the same value and every action costs one pull. There is no discounting, switching cost, or separate exploration reward.

All policies know the reward model, arm count, remaining budget, and independent $\operatorname{Beta}(1,1)$ working priors. They observe only their own selected-arm rewards. The evaluator alone knows the true $\theta_i$. The policy must never receive hidden means, future rewards, an oracle ranking, or the benchmark family name.

Use anonymous arm IDs and randomly permute their assignment to reward streams once per episode. Hold that mapping and display order fixed throughout the episode. Test order changes separately, rather than accidentally making order a source of exploration in the main run.

For paired comparisons, generate independent reward streams indexed by **episode, physical arm, and that arm's pull count**. The $n$th pull of a physical arm reveals the same pre-generated outcome to every policy. Action-randomization RNGs are separate from environment RNGs. Histories will diverge; only starting tasks and potential outcomes are paired.

### 3.2 Four Jev policies

| Policy | Visible evidence | Action rule | Role |
|---|---|---|---|
| J-counts-direct | Successes and failures for every arm | Returned `choice` | Primary unassisted policy |
| J-Bayes-direct | Counts plus posterior summaries | Returned `choice` | Primary assisted policy |
| J-counts-sample | Same counts input | Local categorical draw from normalized probabilities | Sampling ablation |
| J-Bayes-sample | Same assisted input | Local categorical draw from normalized probabilities | Sampling ablation |

For assisted inputs, compute posterior $\alpha_i,\beta_i$, mean, standard deviation, and equal-tailed 95% credible interval in SciPy. Serialize derived numbers to six decimal places. Explain that a credible interval concerns an arm's unknown success probability, rather than the next binary outcome. The prior plus counts already contains this information; this intervention changes accessibility and arithmetic burden, not the underlying observations.

All four policies get the same objective, one Choice question, and all available arms:

> Choose the arm to pull now to maximize expected total reward across all remaining pulls, including this one. Pulling an uncertain arm may improve later choices. Explore only when the expected future benefit justifies the immediate reward tradeoff. Arm labels carry no reward information.

Include `remaining_pulls_including_this_one` explicitly. Do not supply a prescribed exploration rate, a TS recommendation, posterior draws, or an index ranking. Those would turn this into algorithm execution or policy imitation. Do not add arbitrary temperature adjustments or smooth zero-probability choices in the main comparison.

The counts representation is not a raw-history memory test. For this stationary model it is a sufficient statistic. Full histories can be a later ablation.

### 3.3 Classical comparison policies

All use the same observed evidence. Resolve exact numerical ties uniformly with seeded local randomness, except the direct Jev policy, whose returned choice is retained.

| Policy | Specification | Why include it? |
|---|---|---|
| Beta-Bernoulli Thompson sampling | Draw each $\tilde\theta_i \sim \operatorname{Beta}(1+s_i,1+f_i)$; choose the largest | Main Bayesian probability-matching baseline |
| Bayes-UCB variant | Choose the largest posterior quantile at $q_t=1-1/(t+1)$, with first pull $t=1$ | Bayesian optimism versus probability matching |
| Posterior-mean greedy | Choose the largest $(1+s_i)/(2+s_i+f_i)$ | Is Jev effectively exploiting posterior means? |
| UCB1 | Pull unobserved arms first in seeded random order; then maximize $\hat\theta_i+\sqrt{2\log(t)/n_i}$ | Familiar non-Bayesian optimism control |
| Uniform random | Choose every arm with probability $1/K$ | Lower-complexity reference |
| Clairvoyant oracle | Always choose a true best arm | Evaluation ceiling only; not an implementable learner |

The Bayes-UCB schedule above is an explicitly specified practical variant, not a claim that every theorem in the original paper applies unchanged. Record it by its full variant name in results. No tuning on held-out episodes. Do not force an initial round-robin on all policies; UCB1's initialization is part of that policy. [Bayes-UCB paper](https://proceedings.mlr.press/v22/kaufmann12.html), [Thompson sampling tutorial](https://arxiv.org/abs/1707.02038)

Neither Thompson sampling nor Bayes-UCB is the exact finite-horizon Bayesian optimum. Use dynamic programming in the small panel below to make a genuine optimality comparison. Reserve information-directed sampling and finite-horizon index approximations for follow-up work; adding them now would expand implementation effort more than it clarifies the first result.

### 3.4 Diagnostic panel: explicitly test the value of exploration

Use six two-arm posterior fixtures, each starting from the same uniform prior and consistent success/failure counts:

| Fixture | Arm A posterior | Arm B posterior | Purpose |
|---|---|---|---|
| Symmetric unknown | Beta(1,1) | Beta(1,1) | Label bias under equal evidence |
| Close mean, uncertain alternative | Beta(11,9) | Beta(1,1) | Exact short-horizon exploration crossover |
| Clear incumbent | Beta(19,3) | Beta(1,1) | Avoid exploration simply because uncertainty exists |
| Equal means, unequal precision | Beta(10,10) | Beta(1,1) | Uncertainty sensitivity |
| Lucky small sample | Beta(2,1) | Beta(60,40) | Distinguish high mean from strong evidence |
| Unlucky small sample | Beta(1,2) | Beta(6,4) | Recovery from poor early outcomes |

Cross these with both A/B label assignments, horizons $h\in\{1,2,10,100\}$, both evidence formats, and three objective wordings:

1. Neutral: maximize expected total reward over the remaining pulls.
2. Explicit exploration: the main instruction above.
3. Regret framing: minimize expected cumulative regret relative to always pulling the arm with the highest fixed success probability; those probabilities are unknown.

Repeat each identical request five times, without cross-call caching: **1,440 requests**. Compare means/distributions at the fixture level; repeats measure service variability and are not independent environments. Compute exact action values with memoized dynamic programming for $h\leq10$. The $h=100$ arm choices are behavioral diagnostics only; do not label them optimal without solving them.

There is a useful analytic anchor. For A = Beta(11,9), B = Beta(1,1), one pull favors A ($0.55>0.50$). With two pulls:

$$
Q_A=0.55+0.55=1.10,
\qquad
Q_B=0.50+\frac12\frac23+\frac12(0.55)
=1.108\overline{3}.
$$

B is optimal with two pulls because a success makes it attractive for the last pull. This tests the user's central question without conflating long-term choice with randomization.

Additionally, run an **exact-policy episode panel** with $K=2$, $T=20$, 100 independent prior-drawn tasks, all four Jev variants, and the classical learners plus the exact dynamic-programming policy. This adds **8,000 Jev calls**. Score action-value loss against the exact policy at the states Jev actually visits; label this separately from regret against a clairvoyant oracle.

### 3.5 Pilot and main matrix

| Dimension | Pilot | Main exploratory comparison |
|---|---|---|
| Arm counts | 2, 3, 5, 10, 15 | 2, 3, 5, 10, 15 |
| Families | All three below | All three below |
| Episode horizon | 50 | 100 |
| Independent episode seeds per cell | 5 | 40 |
| Jev variants | Four | Four |
| Jev decisions | 15,000 | 240,000 |

Environment families:

- **Clear winner:** one arm with $\theta=0.70$, all others $0.30$; randomize the winner's label.
- **Close winner:** one arm with $\theta=0.55$, all others $0.50$; randomize the winner's label.
- **Prior-drawn:** sample all $\theta_i$ independently from Beta(1,1) for every episode, then keep them fixed.

The first two are fixed-instance stress tests under a working prior. Only the last is a matched-prior Bayesian benchmark. Report them separately. Do not expose their generative rules to the learners.

Use master seed 20260920 and distinct seed namespaces for pilot, main, exact-panel, diagnostics, label permutations, policy sampling, and bootstrap. Freeze prompts, configuration, and seed manifests before the main run. Exclude pilot episodes from main estimates. The main run is an exploratory effect-size study, not a pre-powered equivalence or superiority trial.

The main horizon remains 100 to preserve replication under the initial budget. Within it, use remaining-horizon bins to describe behavior. Fresh 300/1,000-turn studies are follow-ups; a prefix of a long-horizon run is not equivalent to a short-horizon experiment.

## 4. Metrics and interpretation

Primary endpoint: expected-reward loss along the chosen action sequence, usually called cumulative pseudo-regret:

$$
\bar R_T=\sum_{t=1}^{T}(\theta^*-\theta_{A_t}),
\qquad \theta^*=\max_i\theta_i.
$$

Also report realized cumulative reward, pseudo-regret per pull, and best-arm pull fraction for environments with a unique best arm. Expected cumulative reward and expected regret relative to the fixed best arm differ by a policy-independent constant. Therefore “maximize reward” versus “minimize regret” is a framing comparison here, not a new mathematical objective. Maximizing the probability of winning or exceeding a reward threshold would be a different objective and is outside this experiment.

For each $K$/family cell, report paired episode-level differences between J-counts-direct and TS, J-Bayes-direct and TS, and J-Bayes-direct and J-counts-direct. Give mean differences and paired percentile bootstrap 95% intervals using 10,000 resamples. Secondary sampling comparisons get the same treatment. Bootstrap whole episodes, never individual pulls. Any pooled result equally weights the 15 cells and resamples within cells; retain cell-level plots. Intervals are exploratory and not simultaneous claims across all comparisons.

Exploration diagnostics:

- Fraction of choices outside the set of posterior-mean maximizers; separate zero-information ties.
- Pulls of unseen/less-observed arms, switching rate, and number of arms visited.
- Choices of lower-mean/higher-uncertainty arms, with posterior mean differences controlled through the fixed diagnostic fixtures.
- Never revisiting an early-disfavored arm; reward and pseudo-regret in the last 20 pulls.
- Change in action probabilities and choices with horizon, objective wording, and label reversal.
- Exact-panel one-step decision loss $\max_a Q_h(s,a)-Q_h(s,A_t)$ and tie-aware optimal-action agreement.

Log Jev confidence and distribution entropy descriptively. **Do not call its action probabilities Bayesian posterior probabilities that an arm is best, or evaluate them as predicted reward probabilities.** Those are different questions. A later, separate probability-estimation task can test calibration with proper scoring rules.

A nonsignificant gap is inconclusive, particularly with 40 episodes per cell. Infer behavior from observable decisions; the experiment cannot establish Jev's internal algorithm or reasoning process.

## 5. Implementation after discussion

Build a small Python package with a CLI for `smoke`, `diagnostics`, `pilot`, `run`, and `report`. It should share an environment, observation representation, policy interface, recorder, and report generator. No experiment modules are being added in this preparation stage.

The policy interface takes public observations, horizon information, and its own RNG, and returns an arm ID plus diagnostic metadata. Hidden reward parameters and reward streams belong exclusively to the simulator/evaluator. The Jev adapter sends `state`, pinned `model`, and one `questions.next_arm` Choice to `POST https://api.typesafe.ai/v1/systemone`.

Use direct HTTPX rather than an SDK to make retries, requests, accounting, and response validation explicit. Load `.env` internally; never store headers or credentials. Record configuration/prompt hashes, dependency-lock hash, model requested/returned, timestamps, episode/turn IDs, RNG information, counts, raw answer probabilities, normalization mass, selected arm, observed reward, posterior summaries, latency, usage, and all attempt statuses. Evaluator-only quantities remain in distinct records inaccessible to policy code.

Validation and failure handling:

- Require precisely the offered arm IDs, a valid returned choice, finite probabilities in $[0,1]$, positive total mass, and choice probability tied for maximum within $10^{-8}$.
- Preserve raw probabilities; normalize for local categorical sampling when $|\sum_i p_i-1|\leq\max(0.02,0.005K)+10^{-8}$. This accommodates the observed two-decimal-style rounding but is a provisional tolerance, not an official precision guarantee. Flag every correction. Larger discrepancies stop that episode for inspection.
- Do not renormalize the returned confidence or invent probability mass for zero-probability arms.
- Start at four concurrent episodes and five requests/second, with a shared limiter. After 1,000 successful calls, permit up to eight concurrent episodes and 15 requests/second if the 429/529 rate is below 1%; otherwise remain at five requests/second. Each episode stays sequential.
- Use 30-second timeouts. Retry 429, 529, other 5xx, and transport failures at most three times with jittered exponential backoff starting at one second, capped at 30 seconds, honoring a longer `Retry-After` deadline. Reduce shared rate on throttling. Stop on 401/403 or request-validation errors. Stop on model-version mismatch.
- Never silently substitute a classical policy on Jev failure. Record incomplete episodes and resume from the last durable observation; report missingness and paired-analysis exclusions. Mark ambiguous timeout retries as possibly billed, even without a usage response.
- Checkpoint each successful decision and resulting observation with RNG state before moving on. Reuse stored responses only to recover an interrupted same episode/turn; do not deduplicate requests across independent episodes or diagnostic repeats.

Required validation before spending on the pilot:

1. Posterior updates, Beta means/quantiles, and seeded reward-stream replay match analytic fixtures.
2. Exact DP agrees with exhaustive enumeration for tiny horizons, the two-pull crossover above, and terminal posterior-mean maximization.
3. TS action frequencies match posterior probability-of-being-best on a fixed fixture within Monte Carlo error; symmetric-arm ties do not favor a label.
4. Equal-mean environments have zero pseudo-regret; oracle pseudo-regret is zero; cumulative metrics and remaining-pull counters are consistent.
5. Recorded HTTP fixtures cover rounded sums, out-of-range/NaN probabilities, invalid IDs, rate limiting, timeouts, missing usage, version drift, and checkpoint recovery without duplicated rewards.
6. Hidden means and unrevealed outcomes cannot reach policy payloads. Budget stop works with requests in flight and possible duplicate billing.
7. Offline replay reproduces actions for seeded classical and sampled policies; report generation needs no new API calls.

Deliver tables and plots for regret curves, paired effects, horizon-sensitive choices, arm coverage, and latency/cost, plus a Markdown report with limitations and all configurations. Do not call a failed comparison “Bayes optimal” merely because TS was the strongest baseline.

## 6. Cost, run gates, and review point

The direct rate checked on 2026-09-20 is **$0.042 per million input tokens; output is free**. The small counts probe used 478 input tokens; the 15-arm assisted probe used 2,248. Actual average cost will depend on arm count and representation. [Official model/pricing page](https://docs.typesafe.ai/models)

At a planning average of 1,000 input tokens per call:

| Component | Calls | Estimated Jev cost |
|---|---:|---:|
| Preparation probes, already done | 4 | $0.00015456 actual-usage estimate |
| Fixed diagnostic panel | 1,440 | $0.06048 |
| Exact-policy episode panel | 8,000 | $0.336 |
| Pilot matrix | 15,000 | $0.630 |
| Main matrix | 240,000 | $10.080 |
| Planned suite total | 264,440 | $11.10648 |
| With 10% retry/token contingency | — | **$12.22** |

At averages of 750–1,400 tokens per call, the same suite with contingency is approximately **$9.16–$17.10**. This range is a scenario estimate, not a confidence interval. If every call were as long as the 2,248-token probe, the estimate would rise to roughly $27.46 including contingency, so the pilot gate matters.

Set a **project-local Jev cap of $18**, including preparation calls and conservative reserves for uncertain/in-flight charges. Do not automatically purchase credits or treat the user's $200 maximum as the first-run budget. The stated $20 balance comes from the user; no account-balance endpoint was verified.

Before a large run, project costs from observed pilot usage separately by arm count and evidence format, including question overhead. Proceed only if the planned remaining suite fits the cap; otherwise stop after the pilot with its results and a revised design for discussion. Do not silently reduce sample sizes or remove unsuccessful policies. At 5–15 requests/second, 264,440 requests imply roughly 5–15 hours before retries and analysis; token affordability does not imply an instant experiment.

OCR uses a separate Mistral account and is not charged to Jev credits; document its use separately in the research notes.

**Current review point:** discuss the proposed matrix, four Jev variants, exact-policy panel, and spending cap before implementing the harness. GP optimization, semantic arms, and reasoning-model orchestration remain later studies described in the research notes.
