# Interim interpretation: Bayesian assistance changes what Jev can do

E1, E2, and E3 are complete. E4/E5 remain in progress, so this is an interim interpretation rather than the final scientific report. Numerical tables and plots are in [the generated report](report.md); methods and later studies are in the [protocol](../../first_experiment_Plan.md). All intervals below are pointwise, exploratory 95% episode/fixture bootstrap intervals unless stated otherwise. The generated report's accounting table is a run-wide snapshot and includes some in-progress E4 usage; it is not a cost allocation to the filtered first-three results.

## The result is already more nuanced than a failure-to-explore story

Jev's choice quality depends on which calculation the interface exposes. With raw successes/failures it makes some final-pull mistakes. Supplying posterior means removes those observed mistakes in E2. Supplying exact finite-horizon action values yields optimal choices throughout the E2 and E3 samples. These are useful capabilities, but using external calculations is different from computing them autonomously, and a deterministic argmax could execute supplied action values without an AI model.

Neither additional statistics nor stochastic action selection is uniformly beneficial. Full uncertainty summaries sometimes perform worse than means alone on fixed states. Sampling the answer distribution reduces performance in the short closed-loop panel. Those observations motivate separate tests of probability semantics and advice representation rather than a universal judgment about Bayesian competence.

## E1: explicit exploration wording has a measurable but limited effect

On the 100 designed posterior fixtures, at horizon 10 optimal-action agreement was 84.5% with explicit exploration wording, 77.0% with neutral expected-reward wording, and 78.25% with regret wording. Agreement alone can overweight tiny mistakes, so the companion mean action-value losses are 0.01386, 0.02070, and 0.02136 reward units.

A secondary classification finds only eight fixtures at horizon 10 where every posterior-mean-greedy action is strictly suboptimal. Within that small subset, agreement was 31.25% for exploration wording, 3.125% for neutral wording, and 12.5% for regret wording. Explicit wording increased the non-greedy choice rate from horizon 1 to horizon 10 by 28.1 percentage points, with a fixture bootstrap interval of approximately [6.25,56.25]. The small selected stratum and posthoc classification limit generalization.

On the illustrative Beta(11,9) versus Beta(1,1) crossover, all three wordings chose the incumbent at horizons 2, 5, and 10, despite the unobserved arm being optimal. Its two-pull loss is only 0.00833; at horizon 10 it is approximately 0.07636. This is a controlled limitation, not evidence that all non-greedy choices are absent or that the whole policy is poor.

## E2: means help, action values work, and a recommendation pointer is not enough

| Evidence format | Optimal choices, one pull | Optimal choices, ten pulls | Mean action-value loss, ten pulls |
|---|---:|---:|---:|
| Counts | 91.5% | 87.75% | 0.00778 |
| Counts + means | 100% | 93.5% | 0.00256 |
| Full posterior summaries | 100% | 92.25% | 0.00366 |
| Supplied exact action values | 100% | 100% | 0 |
| Supplied optimal-arm recommendation | 98.75% | 89.25% | 0.00674 |

The 400 calls per row/horizon are four repeated/display variants of 100 fixtures, not 400 independent environments. Zero error fixtures among 100 random fixtures gives a two-sided exact 95% upper bound of 3.62% on the fixture-level probability of at least one error under this test protocol. An empirical bootstrap interval of [100%,100%] does not establish perfect population performance.

Six horizon-10 fixtures strictly require exploration. Means and full summaries had zero optimal selections in that subset; supplied action values had all optimal selections. This suggests that improved numerical accessibility can improve exploitation without solving the value-of-information decision.

The recommendation result has been checked against raw payloads. In a representative failure, `recommended_arm` correctly named arm_01 and its posterior mean was .5 versus arm_00's .363636, yet Jev chose arm_00. The source-to-label mapping was correct. A fresh preregistered follow-up will test explicit field references, inline instructions, option-specific advice, distraction removal, and label conventions. It should not assume that the present field layout is the best way to communicate advice.

## E3: strong performance against TS does not establish optimal planning

In 200 two-arm, 20-pull, prior-drawn episodes:

| Policy | Mean pseudo-regret | Mean summed exact action loss |
|---|---:|---:|
| Jev, counts/direct | 0.99457 | 0.18517 |
| Jev, summaries/direct | 0.94430 | 0.06489 |
| Jev, summaries/sampled | 1.34417 | 0.41521 |
| Jev, supplied exact values | 0.89174 | 0 |
| Exact Bayesian DP | 0.87915 | 0 |
| Finite-horizon AP index | 0.87113 | 0.00164 |
| Knowledge gradient | 0.94532 | 0.03160 |
| Monte Carlo IDS | 0.90271 | 0.03615 |
| Bayes-UCB | 0.91146 | 0.05803 |
| Posterior-mean greedy | 0.77784 | 0.06235 |
| Thompson sampling | 1.45571 | 0.50251 |

Summaries versus counts reduced mean summed action loss by 0.12028, interval [-0.15974,-0.08368]. The paired pseudo-regret difference was only -0.05027, interval [-0.17944,0.06415]. These estimators have different sampling variability; the latter is inconclusive at this sample size.

Summaries/direct versus TS reduced pseudo-regret by 0.51141, interval [-0.65389,-0.36550]. However, an independent prior-integrated calculation shows that greedy is already extremely close to optimal at this horizon: its expected total reward is 12.36535 versus optimum 12.43126. TS's expected total reward is 11.93179. Beating TS here is therefore compatible with mostly greedy behavior. See the [analytic audit](../../docs/theory_audit.md).

Summaries/sampled versus summaries/direct increased pseudo-regret by 0.39987, interval [0.27597,0.52191], and summed action loss by 0.35033, interval [0.29541,0.40844]. Host randomization is not itself useful information seeking.

The supplied-value Jev policy incurred zero exact action loss across 4,000 decisions. Its observed reward/regret differs slightly from the exact policy because equally optimal actions can lead to different realized trajectories. This establishes successful use on sampled states, not a universal guarantee on unseen states.

### Why the two loss measures can tell different finite-sample stories

For a matched-prior episode, define $\delta_t=V_{h_t}(S_t)-Q_{h_t}(S_t,A_t)$. The Bellman identity gives

$$
\mathbb E\left[\sum_t\delta_t\right]
=V_T(S_0)-\mathbb E\left[\sum_tY_t\right].
$$

Thus averaging summed local action losses estimates the policy's prior-expected gap from the optimum. It is not the realized shortfall in an individual episode. Independent policy-value calculations and rational tests verify this identity. The lower empirical pseudo-regret of greedy than exact DP in this particular 200-task panel is sampling variation, not a contradiction of Bayesian optimality. Its exact population gap is approximately 0.06591.

## What remains open

The larger-arm, longer-horizon tasks may reward different exploration behavior. The first three experiments do not establish superiority on those settings, calibration of probability outputs, or transfer to semantic tasks. The AP-index comparator was added as a documented posthoc baseline and remains a multi-arm heuristic. All broad comparisons are exploratory, and small strata should not become headline population claims.

The next steps already specified use fresh tasks: advice-interface ablation, explicit predictive/best-arm probability questions, and an online counts/means/full-summary comparison. Any final paper should emphasize the boundary between numerical accessibility, normative planning, and reliable use of external computation.
