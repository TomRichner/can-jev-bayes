# Exact two-arm Bayesian evaluation through horizon 100

Date: 2026-09-21. This is a posthoc normative audit and additional offline baseline. It changes no Jev inputs, original manifests, source trajectories, or frozen scientific modules, and uses no API credits.

## Correction to the earlier computational scope

The original recursive solver imposed a horizon-20 guard to prevent accidental expensive general multi-arm jobs. That was an **implementation guard**, not a fundamental intractability result for two arms. A compact bottom-up table solves every two-arm state through horizon 100 with about 35 MiB of arrays. Therefore E4's two-arm conditions, and E9's two-arm confirmation, can receive exact finite-horizon Bayesian evaluation. Larger-arm E4/E9 conditions remain outside this solver.

This upgrade strengthens the comparison without changing what Jev saw. It also permits evaluation of the finite-horizon AP index against exact multi-arm planning at horizon 100, instead of inferring its accuracy from its narrow single-arm numerical brackets.

## Compact Bellman table

The prior is independent $\operatorname{Beta}(1,1)$ for each arm. For terminal horizon $T$, write $n$ for total observations, $n_1$ for observations of arm 1, $n_2=n-n_1$, and $s_i$ for each arm's successes. The remaining horizon is determined by the counts, $h=T-n$. Store

$$
V[n][n_1][s_1,s_2],\qquad
0\le s_1\le n_1,\quad 0\le s_2\le n_2.
$$

All entries at $n=T$ are zero. At earlier layers, with $m_i=(s_i+1)/(n_i+2)$,

$$
Q_1=m_1+m_1V[n+1][n_1+1][s_1+1,s_2]
 +(1-m_1)V[n+1][n_1+1][s_1,s_2],
$$

$$
Q_2=m_2+m_2V[n+1][n_1][s_1,s_2+1]
 +(1-m_2)V[n+1][n_1][s_1,s_2],
\qquad V=\max(Q_1,Q_2).
$$

NumPy vectorizes each $(n,n_1)$ matrix. Retaining all compact layers supports constant-time state lookup and action-value reconstruction when scoring existing trajectories. There are

$$
\sum_{n=0}^T\sum_{n_1=0}^n(n_1+1)(n-n_1+1)
=\binom{T+4}{4}
$$

float64 values. At $T=100$ this is **4,598,126 entries**, or **36,785,008 array bytes** (about 35.1 MiB), plus small Python/container and transient-array overhead. Both arithmetic work and retained values scale as $O(T^4)$ in this two-arm uniform-prior state representation. The public implementation restricts $T\le150$ and applies an explicit 256 MiB array-storage guard.

There is no stochastic approximation, policy rollout, or per-arm index decomposition. “Exact” means exhaustive Bellman optimization under the stated model, subject to ordinary floating-point arithmetic. Public `TwoArmTable.q(successes, failures)` returns both action values with remaining horizon fixed as $T-\sum_i(s_i+f_i)$. It does not accept an unrelated remaining horizon or arbitrary nonuniform starting priors.

## Validation and population anchor

Unit tests compare **every reachable state** through small terminal horizons with the independent recursive solver, verify symmetry and memory accounting, and reproduce the rational two-pull exploration crossover at posterior states Beta(11,9) and Beta(1,1). They also verify the existing horizon-20 prior value, seeded ties, terminal values, replayed reward streams, and recovery of interrupted audit writes without modifying live source records.

The initial horizon-100 table benchmark took approximately **0.063 seconds** on this workspace. It gives

$$
V^*_{100}(\operatorname{Beta}(1,1),\operatorname{Beta}(1,1))
=64.918420651023.
$$

Since the prior expected best-arm mean is $2/3$, its population Bayesian regret is

$$
100\cdot\frac23-V^*_{100}=1.748246015644.
$$

These are prior expectations, not estimates from the selected simulation seeds. They provide a useful anchor when interpreting finite-panel pseudo-regret: a policy can appear better than the optimum on a small fixed collection of worlds without exceeding its population Bayesian value.

## Scoring existing episodes and adding the exact policy

The standalone CLI uses original manifests to identify E4's three two-arm families with 40 episodes each, and E9's two-arm prior family with 100 episodes. It reuses the original `Episode` environment and potential-outcome streams. The added policy name is `exact_long_horizon`; its per-turn action RNG is separately namespaced. Ties within $10^{-12}$ are uniformly randomized.

For completed Jev, classical, AP-index, and added exact-policy trajectories, the scorer replays successes/failures and records

$$
\ell_t=\max_aQ_{h_t}(S_t,a)-Q_{h_t}(S_t,A_t).
$$

It verifies the world seed, hidden means, turn order, and every observed reward against the physical-arm/pull-index outcome stream. Each scored record includes the original episode's hash. It reports cumulative loss and tie-aware optimal-action agreement with tolerance $10^{-10}$. A subsequent scoring invocation rejects changed source episodes and adds newly completed episodes, making it safe to run again after ongoing Jev collection ends.

On the matched-prior E4/E9 tasks, expected summed Bellman loss is the policy's expected reward shortfall from the Bayes-optimal policy. For E4's fixed clear/close families, it remains a model-based decision diagnostic under the declared uniform working prior; it is not the same as frequentist pseudo-regret or proof of frequentist optimality for those fixed mean vectors. Retain both metrics and their distinct interpretations.

Separate outputs in each original run directory are:

- `two_arm_audit_manifest.json`: source-method hash, original manifest hash, prior, arithmetic, and tie conventions.
- `episodes_long_exact.jsonl`: added exact-policy episodes, explicitly marked posthoc/exploratory.
- `two_arm_bellman_audit.jsonl`: episode-level exact losses plus per-decision action values.
- `two_arm_audit_summary.json`: descriptive means and completed counts by experiment, family, and policy.

These files supplement the original analysis; the original manifests and trajectory records are not rewritten. Missing or still-running episode groups remain missing and must not be interpreted as completed comparisons. The summary is descriptive; any paired uncertainty analysis should use whole episodes and the original task identities.

## Commands

```bash
.venv/bin/pytest -q tests/test_two_arm_table.py
PYTHONPATH=src .venv/bin/python -m jevbandits.two_arm_table --benchmark --horizon 100
PYTHONPATH=src .venv/bin/python -u -m jevbandits.two_arm_table --baseline --score
# Repeat scoring after additional Jev episodes finish; it adds only new records.
PYTHONPATH=src .venv/bin/python -u -m jevbandits.two_arm_table --score
```

Default run directories are `artifacts/overnight_v2` and `artifacts/sampling_v1`. The initial construction executed only unit tests and the horizon-100 benchmark; the complete offline baseline/scoring job is ready for delegated execution. This scope correction should supersede earlier statements that exact DP is unavailable throughout E4: it is now available for E4 and E9 **when $K=2$**.
