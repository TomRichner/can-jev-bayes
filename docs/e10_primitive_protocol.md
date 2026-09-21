# E10: forecast target, elicitation primitive, and known-probability controls

Version 1, specified before any E10 API responses. This posthoc follow-up uses fresh fixtures to address an interpretation gap in E7. E7 observed relatively accurate Noul next-reward forecasts (full-summary mean absolute error about .029) but poor Choice latent-best distributions at K = 15 (categorical excess Brier about .403 and KL about 4.67). Target and primitive were confounded. Those results alone cannot establish a general failure of joint Bayesian inference. E10 is a necessary control, not a reasoning-model test or an online action-policy comparison.

## Frozen factorial design

The namespace is `e10_primitive_bridge`. Independently for K = 2, 5, 15, generate 30 fresh fixtures using `stable_seed(20260920, experiment, "fixtures", K)`. Each arm's observation count is drawn uniformly from {0,2,5,10,20}; successes are uniform integers from zero through that count. Priors are independent Beta(1,1). Every target and condition shares the same fixture. No state is selected by a Jev answer, previous study outcome, or favorable reference value.

There are four representations:

1. `counts`: each arm's successes and failures.
2. `bayes`: counts plus posterior mean, Beta parameters, standard deviation, and 95% equal-tailed credible interval, using the existing E7 observation builder.
3. `oracle_probs_full`: full summaries plus each arm's exact probability of the target event.
4. `oracle_probs_only`: arm IDs and exact target-event probabilities, with no counts or posterior moments.

Both oracle conditions say explicitly that the supplied values are known conditional probabilities of the named events and should be reported directly. Reward fields are named `exact_probability_next_reward_is_one`; best-arm fields are named `exact_probability_true_mean_is_largest`. Only probabilities for the current target are supplied. Float64 analytic/quadrature values are serialized without extra decimal rounding. These controls introduce external calculations and instruction content; they do not test autonomous inference from counts.

Each representation has two repeats of these four target/elicitation packages:

| Target | Elicitation | Questions per fixture/repeat |
|---|---|---:|
| Next reward is 1 | Noul binary event for each arm | K |
| Next reward is 1 | Choice over yes/true and no/false for each arm | K |
| Named arm's fixed unknown mean is largest | Noul binary event for each arm | K |
| Identity of arm with largest fixed unknown mean | One K-option Choice | 1 |

The two reward primitives use exactly identical question text, observation, and criterion rubric; only `type` changes. The binary criteria IDs are `true` and `false`, with explicit Yes and No wording. Choice probabilities, not its selected category, are the forecast. All questions expressly request forecasts, not arm recommendations. They retain the original shared Bernoulli likelihood/prior context. Each question contains its own complete observation; repeats and unrelated questions contribute no shared observations.

The best-arm comparison necessarily changes scope: one joint categorical distribution versus separate marginal event forecasts. It therefore contrasts elicitation packages, including joint-coherence demands, not an isolated internal architecture. Marginal Noul questions retain all arms' public evidence, so they are not restricted to the queried arm's counts. Ties have probability zero under independent continuous Beta posteriors.

There are 90 independent fixtures and `30 × 4 × 2 × ((3×2+1)+(3×5+1)+(3×15+1)) = 16,560` questions. The execution order is a fixed independent permutation seeded from the experiment namespace. Repeats have identical payloads but unique decision IDs. No prompt search, extra model, sampled action, or adaptive question selection is included.

## Reference quantities and scoring

The next-reward reference for arm i is `(successes_i + 1)/(successes_i + failures_i + 2)`. The best-arm reference is the posterior probability that its latent mean is largest, computed by the existing E7 adaptive quadrature and its error checks. These are different target quantities even though both are probabilities.

The common primary metric is **mean binary excess Brier risk per fixture**. For event probabilities p_i and references q_i, it is `2/K × sum_i (p_i-q_i)^2`, averaged over repeats. For reward Noul, reward binary Choice, and best-arm Noul, calculate `2(p_i-q_i)^2` per arm then average arms and repeats. For the K-option best-arm Choice, multiply its ordinary categorical squared-error sum by `2/K`, then average repeats. This equals the average risk of the K associated binary events and supplies the same scale for the primitive comparisons at a fixed K. It does not turn the latent-best and next-reward targets into the same event, nor make raw risks equally difficult across K.

Do **not** normalize the separate Noul best-arm probabilities before primary scoring, even if their sum differs from one. The scores are excess expected proper-scoring risk conditional on the specified model, not observed predictive accuracy, empirical outcome losses, or policy reward. API repeats and arm components are not independent sampling units.

Secondary outputs include mean absolute event error, mean binary excess log risk, Noul best-arm probability sum and absolute deviation from one, and an explicitly labeled normalized Noul best-arm Brier risk. If the Noul sum is zero, normalized risk is undefined; do not substitute a uniform distribution. Binary log scoring uses the frozen E7 routine: clip the two class probabilities at 1e-6 and renormalize, then compute KL(reference || clipped prediction). This is secondary and does not change the primary Brier values. Any API probability-mass rounding adjustment for Choice follows the original validated client and retains the raw answer.

## Prespecified inference

Exactly two overall primary contrasts use **full Bayesian summaries only**:

1. Next-reward Choice risk minus next-reward Noul risk.
2. Best-arm Noul risk minus best-arm Choice risk.

Negative differences favor the first package. Within each K, pair complete fixture risks and average the differences. Pool the three K means with equal weight. Use 10,000 bootstrap resamples of whole paired fixtures independently within each K, recomputing the equal-K average on every draw. Reporting seeds are deterministic functions of the experiment and contrast.

Report both pointwise 95% percentile intervals (quantiles .025, .975) and Bonferroni 97.5% intervals (.0125, .9875) for each of the two primary contrasts. The latter target simultaneous 95% coverage for this two-contrast family subject to bootstrap approximation. Per-K primitive contrasts for every representation receive exploratory paired-fixture pointwise 95% bootstrap intervals. Other representations, per-K comparisons, normalized Noul analyses, and coherence findings are secondary or exploratory; they do not enlarge the prespecified primary family. Do not select favorable endpoints or intervals after seeing results, and do not call an interval crossing zero equivalence.

Only complete arm/repeat packages enter fixture risks. Disclose missing questions and conditions, per-K left/right fixture counts, and complete pairs against 30 expected fixtures per K. Do not treat incomplete packages as if they had the same arm/repeat composition. A pooled estimate from an incomplete design is descriptive; do not silently drop K strata. Fewer than two pairs in any K gives no uncertainty interval. The fixed N does not change because of time, credits, or results.

## Reproducibility and execution gate

`python -m jevbandits.primitive_followup prepare` freezes `artifacts/primitive_v1/manifest.json` and `fixtures.json` without an API client or credential access. The manifest records the design, exact reference fixture hash, all ordered question payloads' hash, scoring-context hash, model `jev-1.13.0`, source/dependency hashes (including this module, protocol, tests, E7 client/reference implementation, original scientific dependencies, and `uv.lock`), primary contrasts, and interval quantiles.

`python -m jevbandits.primitive_followup run` regenerates and verifies the frozen design before creating the existing E7 `ForecastClient`. It inherits the durable SQLite cache, question-hash checks, bounded retries, cost reservations, and previous-sibling-ledger accounting. The original shared project cap defaults to $18. The rough incremental planning estimate is under/about $0.60, but actual measured token use and budget reserves determine feasibility. Live E10 waits for the parent budget gate after E9. Maintain one live API writer across sibling ledgers and do not purchase credits automatically.

`--max-questions` pauses the fixed pending permutation without changing the design. A restart reuses validated cached responses, including responses saved before a scoring interruption; completed records are skipped. No live calls or long jobs occur merely from preparation or mock tests. Scientific-source or payload changes require a new namespace/amendment. `python -m jevbandits.primitive_followup report` writes fixture risks, completeness, primary intervals, descriptive summaries, secondary coherence records, and a standalone Markdown interpretation.
