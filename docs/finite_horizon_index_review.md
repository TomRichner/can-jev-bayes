# Finite-horizon indices for the Bernoulli study

Date: 2026-09-20

## Recommendation

Add one **finite-horizon average-productivity (AP) index** baseline to the later Bernoulli analysis. In bandit language it is commonly called the finite-horizon Gittins index strategy. It is a worthwhile comparator because it is deterministic, explicitly uses the remaining horizon, and assigns exploration value by solving a separate optimal-stopping problem for each arm. That mechanism is different from Thompson sampling, Bayes-UCB, knowledge gradient, IDS, and posterior-mean greedy.

Do not add a second policy under the name “Whittle index.” Whittle's label is normally used for a subsidy index in a restless bandit, and importing it here would require a separate time-augmented model and indexability argument. Niño-Mora's earlier route to the finite-horizon result used machinery related to restless-bandit indexation, but the 2011 paper names and defines the object as the finite-horizon AP counterpart of the Gittins index. “Finite-horizon AP index” is the least ambiguous result label for this study.

This baseline is especially useful in E3, where exact full-state dynamic programming can measure its multi-arm approximation error, and in E4, where exact global DP is unavailable. It can be computed offline without more Jev calls. It should be treated as a sixth policy with a prespecified directional question: **does a horizon-aware per-arm stopping index improve on Bayes-UCB and one-step knowledge gradient in close and prior-drawn problems, especially early in an episode, while approaching greedy behavior near the terminal pull?**

A seventh policy is not justified from these papers. If a separate follow-up is desired, make it a diagnostic rather than another baseline: enumerate or sample two-arm states through horizon 20, compare AP-index choices with exact joint-state DP, and report disagreement and exact Bellman loss by remaining horizon. That directly measures the price of index decomposition. Running both an exact RAG implementation and a calibrated implementation as separate policies would mostly measure numerical error and pad the policy list.

## What the index is

For an arm in posterior state $i$ with $d$ pulls remaining, Niño-Mora defines

$$
\lambda^*(d,i)=\max_{1\leq\tau\leq d}
\frac{\mathbb E_i^\tau[\sum_{t=0}^{\tau-1}\beta^tR(X_t)]}
{\mathbb E_i^\tau[\sum_{t=0}^{\tau-1}\beta^t]}.
$$

Here $\tau$ is a stopping time after at least one play. In this study $\beta=1$, $i=(a,b)$ is a Beta posterior, and $R(i)=a/(a+b)$. The index is the largest constant per-pull reward $\lambda$ for which it remains worthwhile to start the unknown arm and retain the option to stop later. It equals the posterior mean at $d=1$ and is nondecreasing in $d$.

The associated multi-arm policy computes $\lambda^*(d,a_i,b_i)$ for every arm using the same global remaining horizon $d$, then selects the largest value. Ties should use the study's seeded tie rule.

## A practical Bernoulli computation

The simplest implementation for this project is calibration by a reward threshold. For a candidate safe-arm reward $\lambda\in[0,1]$, define the excess value of optimally continuing an arm rather than taking $\lambda$ forever:

$$
D_0(a,b;\lambda)=0,
$$

$$
C_d(a,b;\lambda)=\frac{a}{a+b}-\lambda
+\frac{a}{a+b}D_{d-1}(a+1,b;\lambda)
+\frac{b}{a+b}D_{d-1}(a,b+1;\lambda),
$$

$$
D_d(a,b;\lambda)=\max\{0,C_d(a,b;\lambda)\}.
$$

Then $\lambda^*(d,a,b)$ is the threshold at which $C_d(a,b;\lambda)$ crosses zero. Search on $[a/(a+b),1]$ by bisection, evaluating the triangular Beta-state recursion by backward induction. The lower endpoint is valid because an option to stop cannot make the arm worth less than its immediate posterior mean; the upper endpoint is valid because Bernoulli reward is bounded by one.

For an index tolerance $\varepsilon$, bisection needs $\lceil\log_2(1/\varepsilon)\rceil$ DP evaluations: 20 evaluations give interval width below $10^{-6}$ and 24 below $10^{-7}$. Cache completed indices by $(d,a,b)$ across episodes. At each fixed $\lambda$, cache the continuation recursion within the search. Record the final lower and upper brackets, tolerance, tie rule, and arithmetic type so close index comparisons are auditable.

An alternative is Niño-Mora's calibration grid: solve the stopping DP for a common increasing grid $\{\lambda_l\}_{l=1}^L$ and take the first grid value at which stopping is optimal. Its index error is controlled by the grid spacing, and its matrix/block form can be faster when a large table is needed. A uniform $10^{-4}$ grid gives at most $10^{-4}$ one-sided discretization error in each reported index; an arm ranking is numerically unresolved when the index intervals overlap. Bisection is a better first implementation for sparse, on-demand states; a vectorized grid is attractive if most reachable states will be precomputed.

Niño-Mora's recursive adaptive-greedy (RAG) algorithm computes the finite-horizon discrete-state index exactly in the paper's arithmetic-operation model. For a Beta-Bernoulli arm reachable from one starting prior it computes $T(T+1)(T+2)/6=O(T^3)$ relevant index values, with worst-case $O(T^6)$ time and $O(T^5)$ working storage. The reported implementation was faster than five-significant-digit calibration through $T=70$, but its storage rose to about 15.7 GB at $T=90$. Substituting $T=100$ into the paper's storage formula gives roughly 28 GB of doubles. The paper concludes that three- or four-significant-digit calibration is preferable when the horizon or state set is large. This makes RAG suitable as a small-horizon validation oracle, not the default for E4 at horizon 100.

“Exact” needs two qualifiers here:

- RAG is exact for the **single-arm finite-horizon stopping index**, apart from ordinary finite-precision arithmetic in an implementation. Calibration or bisection approximates that same index to a declared tolerance.
- Even perfectly computed per-arm indices give a **heuristic multi-arm policy** for finite, undiscounted horizons. The index is exactly optimal only for the one-unknown-arm problem with a known constant alternative. The classical Gittins optimality theorem applies to the infinite-horizon, geometrically discounted multi-arm problem, not this study.

## Difference from the study's exact DP

The study's exact DP evaluates

$$
V_h(s)=\max_i\left[m_i+m_iV_{h-1}(s^{i,+})+(1-m_i)V_{h-1}(s^{i,-})\right]
$$

on the **joint state of every arm**. It accounts for all future switching and for the opportunity costs created by the other uncertain arms. Its state space grows combinatorially with arm count and horizon.

The AP index instead solves $K$ separate one-arm stopping problems against a hypothetical constant alternative and ranks their scalar reservation rewards. This decomposition is cheap enough to use when joint-state DP is not. It loses exact Bayesian optimality because the actual alternatives also learn and change. Lattimore gives an explicit two-unknown-arm Gaussian counterexample, while finding the index and Bayesian-optimal policies empirically almost indistinguishable in a two-arm experiment. Niño-Mora likewise describes the multi-arm finite-horizon rule as a suboptimal heuristic and cites strong small-horizon Bernoulli performance.

Consequently, E3 should report AP index versus exact DP using cumulative reward, pseudo-regret, optimal-action agreement, and local Bellman loss. E4 should call it a horizon-aware heuristic, never “Bayes optimal.” A useful mechanism check is the exploration premium

$$
\lambda^*(d,a,b)-\frac{a}{a+b},
$$

which must be zero at $d=1$ and should generally shrink as the episode ends or the posterior concentrates.

## Expected computational cost

For a single requested state, the Beta recursion has $O(d^2)$ reachable posterior states per candidate $\lambda$. Bisection therefore costs $O(d^2\log(1/\varepsilon))$ arithmetic operations per uncached index, with small constants and two successors per state. Selecting among $K$ arms multiplies this by at most $K$, although cross-episode caching should remove many repeated $(d,a,b)$ requests. The online lookup after precomputation is $O(K)$.

The main practical risk is eager construction of every RAG value through $T=100$, whose $O(T^5)$ storage is unnecessarily large. Start with memoized bisection or a moderate calibration grid, validate it against exact full DP only as a policy comparison, and validate a small subset of index values against RAG or tighter bisection. Sensitivity at $10^{-3}$, $10^{-4}$, and $10^{-6}$ is more informative than presenting three numerical versions as separate algorithms. If policy choices and episode outcomes are unchanged between $10^{-4}$ and $10^{-6}$ except for declared near-ties, $10^{-4}$ is adequate for the large run.

## Sources and scope

- José Niño-Mora, [“Computing a Classic Index for Finite-Horizon Bandits”](https://doi.org/10.1287/ijoc.1100.0398), *INFORMS Journal on Computing* 23(2), 2011. This is the primary source for the discrete-state AP definition, calibration DP, exact RAG algorithm, Beta-Bernoulli state counts, complexity, and computational comparison.
- Tor Lattimore, [“Regret Analysis of the Finite-Horizon Gittins Index Strategy for Multi-Armed Bandits”](https://proceedings.mlr.press/v49/lattimore16.html), COLT/PMLR 49, 2016. This is the primary source for the exact-versus-global-policy distinction, regret results and computational evidence in the Gaussian model, and the warning that finite-horizon Gittins is not generally Bayesian optimal. Its closed-form approximation is Gaussian-specific and should not be transplanted into the Bernoulli experiment.

The regret theorem and closed-form exploration bonus in Lattimore are for Gaussian rewards and Gaussian priors. They motivate the comparator but do not establish the same finite-time guarantee for Beta-Bernoulli rewards. For Bernoulli arms, use the discrete Beta-state stopping DP from Niño-Mora rather than Lattimore's Gaussian spline or closed-form approximation.
