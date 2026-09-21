# Analytic audit of the two-arm finite-horizon benchmark

Date: 2026-09-20. This additional scientific check was prompted by the finite-sample E3 classical results. It does not alter the frozen experimental policies, prompts, seeds, or runs. No API calls were made.

## Question and method

A simulated panel can rank posterior-mean greedy above the Bayes-optimal policy by realized pseudo-regret. Optimality applies to the expectation over the declared prior and outcome process; it does not guarantee a ranking on every finite collection of tasks and reward streams. We therefore calculate population policy values directly for two independent $\operatorname{Beta}(1,1)$ arms.

For a fixed policy $\pi$, let $p_\pi(a\mid s,h)$ be its action distribution given posterior state $s$ and $h$ remaining decisions. Its value satisfies

$$
V^\pi_0(s)=0,
$$

$$
V^\pi_h(s)=\sum_a p_\pi(a\mid s,h)
\left[m_a+m_aV^\pi_{h-1}(s^{a,+})+(1-m_a)V^\pi_{h-1}(s^{a,-})\right].
$$

Replacing the action-distribution average with a maximum gives the Bayes-optimal value. The independent implementation in `src/jevbandits/policy_values.py` exhausts this recursion using memoization and symmetric-state canonicalization. It does not call the experiment's baseline selector or optimal-value recursion. The results integrate outcome and policy randomness; they are not Monte Carlo estimates. Numerical arithmetic is float64, with SciPy Beta quantiles.

Deterministic policies reproduce the experimental indices and uniformly randomize ties within the same $10^{-12}$ absolute tolerance. In particular, Bayes-UCB uses $1-1/t$, $t=T-h+1$, including the all-zero quantile tie at $t=1$. UCB1's unobserved-arm initialization and knowledge gradient's $m_a+(h-1)\operatorname{KG}(a)$ score are retained.

For Thompson sampling, with $X\sim\operatorname{Beta}(\alpha_x,\beta_x)$ and $Y\sim\operatorname{Beta}(\alpha_y,\beta_y)$ independent and integer parameters,

$$
\Pr(X>Y)=\sum_{i=0}^{\alpha_x-1}
\frac{B(\alpha_y+i,\beta_x+\beta_y)}
{(\beta_x+i)B(1+i,\beta_x)B(\alpha_y,\beta_y)}.
$$

The finite sum is accumulated in log space and independently checked against numerical quadrature of $f_X(x)F_Y(x)$. It integrates the continuous posterior sampling law. The experimental selector's roundoff-sized tolerance also treats samples differing by at most $10^{-12}$ as ties; the resulting distinction is below the precision reported here. Monte Carlo IDS is intentionally excluded because its action law also depends on finite posterior-sample estimation, which this audit does not integrate.

## Population results

Values are expected cumulative rewards, not success probabilities. All rows start with two independent uniform priors. The final column measures the horizon-20 shortfall from the exact optimum.

| Policy | Reward, $T=5$ | Reward, $T=10$ | Reward, $T=20$ | Optimality gap, $T=20$ |
|---|---:|---:|---:|---:|
| Bayes-optimal DP | 2.888888889 | 6.021785714 | 12.431264906 | 0.000000000 |
| Knowledge gradient | 2.888888889 | 6.002407407 | 12.399618913 | 0.031645993 |
| Bayes-UCB, $c=0$ | 2.888888889 | 6.004505772 | 12.376397778 | 0.054867128 |
| Posterior-mean greedy | 2.884722222 | 6.005810786 | 12.365351889 | 0.065913017 |
| Thompson sampling | 2.716379630 | 5.704544183 | 11.931791513 | 0.499473392 |
| UCB1 | 2.727777778 | 5.600919312 | 11.523745226 | 0.907519680 |
| Uniform random | 2.500000000 | 5.000000000 | 10.000000000 | 2.431264906 |

For two independent uniforms, $\mathbb{E}[\max(\theta_1,\theta_2)]=2/3$. Consequently the population Bayesian regret is $2T/3-V^\pi_T$. At $T=20$, it is **0.902068427** for the optimum, **0.967981444** for greedy, and **1.401541820** for Thompson sampling.

These results establish that greedy is unusually competitive in this short-horizon matched-prior benchmark: its expected gap is only **0.065913017 reward units over 20 decisions**. Its advantage over Thompson sampling is real for this task distribution and horizon. Its apparent advantage over exact DP in a finite simulated panel is not evidence against DP optimality. None of these facts establishes the ranking at longer horizons, larger arm counts, or other task distributions.

## Local decision loss and interpretation

Define local Bayes-optimal decision loss at a visited state by

$$
\ell_t=V^*_{h_t}(S_t)-Q^*_{h_t}(S_t,A_t).
$$

The finite-horizon performance-difference identity gives

$$
\mathbb{E}_\pi\left[\sum_{t=1}^T\ell_t\right]
=V^*_T(S_1)-V^\pi_T(S_1).
$$

Thus the mean summed Bellman loss in the sampled E3 panel can be checked against the exact population gaps above. The initially observed approximate sums, 0.0623 for greedy and 0.5025 for Thompson sampling, are close to the population values 0.065913017 and 0.499473392. They support the interpretation that the small-panel pseudo-regret ranking reflects sampling variation, while TS's larger short-horizon planning loss is substantive. Final experiment estimates and uncertainty belong in the experiment report, not this audit.

For Jev, this diagnostic helps distinguish matching a good simple policy from outperforming Bayesian planning. If Jev resembles greedy and achieves low loss here, that is meaningful competence on the stated task; a larger gap from TS is not by itself a failure to explore appropriately. The fixed-state horizon crossover and broader sequential panels remain necessary to identify where useful exploration emerges or fails.

## Reproduction and checks

```bash
PYTHONPATH=src .venv/bin/python -c \
  'from jevbandits.policy_values import prior_value_table; import json; print(json.dumps(prior_value_table(), indent=2))'
.venv/bin/pytest -q tests/test_policy_values.py
```

Tests cover Beta comparison against independent quadrature, rational exhaustive greedy values on tiny histories, analytic one- and two-turn values, action-law agreement with the experimental selector, independent optimal DP agreement, permutation invariance, optimal dominance through $T=20$, and the expected cumulative Bellman-loss identity. The audit uses the same mathematical assumptions as E3 and requires no external data or API credits.
