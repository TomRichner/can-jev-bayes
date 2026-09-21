# E7 protocol: probability forecasts are not action probabilities

Proposed after E1/E2, before collecting any E7 responses. Run only after completing the first five experiments and checking the remaining project budget. Reasoning-model access remains out of scope.

## Motivation and questions

E2 showed that providing posterior means can improve action ranking, while providing exact action values can eliminate measured choice loss on a finite panel. These results do not establish that Jev's probability outputs match Bayesian uncertainty. E7 explicitly asks about two different probability targets, without reinterpreting action-choice probabilities as forecasts.

1. **Next reward:** Noul asks whether one additional pull of a specified arm yields reward 1. The reference is the exact posterior predictive mean.
2. **Latent best arm:** Choice asks which arm has the largest unknown success probability. The reference is its probability of being best under independent Beta posteriors, computed by adaptive vector quadrature.

## Frozen design

Use 100 independent fixtures for each $K\in\{2,3,5,10,15\}$, disjoint seeds from all earlier studies. For every arm, draw its observation count uniformly from {0,2,5,10,20} and its successes uniformly from zero through that count. Conditional on the externally assigned count, this is exactly the Beta(1,1)-Binomial prior-predictive marginal. It is not an adaptive policy's visited-state distribution.

Cross counts, means, and full Bayesian summaries with two API repeats. For each representation/repeat ask one best-arm Choice and one Noul per arm: **24,000 questions**, on 500 underlying fixtures. Reward questions receive only their specified arm's evidence to avoid irrelevant-arm interference; best-arm questions receive the whole fixture. This interface difference is declared rather than attributed to the statistical target alone.

Keep the same model, isolated-question batching, response normalization, credentials handling, project cap, and raw-response ledger as the main study. A dedicated adapter adds the documented Noul answer shape; no frozen main-study source is modified. Freeze every payload and source hash before running. Exact instructions are in `forecast_followup.py` and the run manifest. No prompt optimization on E7 outcomes.

## References and scoring

For arm $i$, with successes $s_i$ and failures $f_i$:

$$q_i^{\mathrm{reward}}=\frac{1+s_i}{2+s_i+f_i}.$$

For independent posterior densities $f_i$ and CDFs $F_i$:

$$q_i^{\mathrm{best}}=\int_0^1 f_i(x)\prod_{j\ne i}F_j(x)\,dx.$$

Quadrature uses absolute and relative tolerances $10^{-10}$; reject references if its error estimate or probability-mass deviation exceeds $10^{-7}$. Tests check symmetric posteriors, analytic Beta(2,1)-versus-uniform values, permutation, and the mixed Noul/Choice transport.

Report excess multiclass Brier risk $\sum_i(p_i-q_i)^2$ and excess log loss $D_{\mathrm{KL}}(q\Vert p)$. For a binary event the two-class Brier excess is twice the scalar squared error. These equal excess expected proper-scoring losses under the analytic conditional target; no noisy future outcomes need to be fabricated. For log loss, clip predictions below $10^{-6}$ and renormalize, reporting how often positive-reference events received zero model probability. This prevents hiding extreme overconfidence behind undefined log scores.

Simple controls: empirical success rate with .5 for an unobserved arm, and normalized posterior means for best-arm beliefs. The latter is intentionally a simple non-Bayesian shortcut, not a correct best-arm probability. The analytic reference is the scoring optimum by construction.

## Inference and interpretation

Average arms and repeats within each fixture; give each fixture equal weight. Report per-$K$/target/representation means and 10,000-resample fixture bootstrap 95% intervals. Primary paired contrasts are means-minus-counts, summaries-minus-counts, and summaries-minus-means for excess Brier risk; log loss and other comparisons are secondary. Intervals are exploratory and pointwise, without equivalence or broad superiority claims.

Reliability plots compare predicted probabilities with analytic conditional probabilities, not imagined observed outcomes. Pooling probability components for visualization is descriptive; they are not independent inferential units. Differences may reflect numerical copying, probability semantics, or evidence interpretation. They do not directly imply an online reward advantage and do not identify Jev's internal inference mechanism.

Expected additional cost is below $2 at 2,000 input tokens/question; use actual ledger usage and the same overall $18 cap. No new credits are purchased automatically.
