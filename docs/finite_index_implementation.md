# Finite-horizon AP index: implementation and posthoc evaluation

Date: 2026-09-20. This comparator was added after the five-study protocol was frozen and its Jev runs had begun. It is an **exploratory, posthoc classical baseline**, using no additional Jev requests. The original manifest and scientific implementation remain unchanged. The literature and mathematical distinctions are recorded in [finite_horizon_index_review.md](finite_horizon_index_review.md).

## Numerical method

For an arm with posterior $\operatorname{Beta}(a,b)$ and $h$ remaining decisions, the stopping calibration is

$$
C_h(a,b;\lambda)=m-\lambda+mD_{h-1}(a+1,b;\lambda)+(1-m)D_{h-1}(a,b+1;\lambda),
\qquad D_h=\max(0,C_h),\quad D_0=0,
$$

where $m=a/(a+b)$. Bisection locates the root of $C_h$ in $[m,1]$. At each candidate $\lambda$, the reachable Beta states form a triangular array: depth $j$ has $j+1$ success counts, with posterior means $(a+s)/(a+b+j)$. NumPy vectorizes each depth, and backward induction retains only the next value row. Root continuation is returned before applying the optional-stopping maximum.

The default stopping criterion is bracket width at most $10^{-6}$. Every result exposes the lower endpoint, upper endpoint, midpoint, iteration count, and requested tolerance. Thus its midpoint error is at most half the bracket width, apart from floating-point error. Completed results are cached by $(a,b,h,\varepsilon)$ with a bounded 200,000-entry cache; triangular working arrays are not cached. Computation uses float64. No eager multi-gigabyte index table is constructed.

The multi-arm policy computes these indices using the global remaining horizon and chooses the largest midpoint. As in the classical baselines, numerical ties within $10^{-12}$ are broken uniformly with the episode/turn action RNG. Every trace records all index brackets. Overlap between the selected bracket and another arm's bracket with **different posterior parameters** is flagged as an unresolved numerical ranking. Identical posteriors are ordinary symmetric ties. This distinction supports a later tolerance-sensitivity audit.

This index solves a one-unknown-arm retirement problem. Ranking multiple uncertain arms by these indices is a **heuristic**, not exact undiscounted finite-horizon Bayesian control. Its exactness should therefore never be inferred from narrow numerical brackets. E3's full-state DP measures its actual multi-arm planning loss.

## Validation and feasibility

The unit tests check:

- $h=1$ gives the posterior mean exactly.
- With $m^+=(a+1)/(a+b+1)$, the two-pull index is $m(1+m^+)/(1+m)$.
- The index increases with horizon on fixed fixtures and its exploration premium is smaller for a concentrated symmetric posterior.
- Final bracket endpoints have opposite stopping-continuation signs; tighter bisection lies inside the default bracket.
- An independent full DP with an unknown arm and a reusable known safe arm selects the unknown arm below the reservation threshold and the safe arm above it.
- Symmetric ties are unbiased, replay is seeded, terminal actions are greedy, and invalid inputs are rejected.

The initial local benchmark computed uncached $h=100$ indices in about **5 ms each**: Beta(1,1) gave 0.868456364, Beta(10,10) gave 0.602193356, and Beta(19,3) gave 0.915409695. These are midpoint estimates at $10^{-6}$ bracket tolerance. Cross-episode reuse makes the additional offline panel feasible without large allocations or API spending.

## Paired offline panel and provenance

The standalone CLI reads E3, E4, and E5 dimensions from the original run's manifest and reuses its `Episode` class, world seeds, and physical-arm/pull-index reward streams. It verifies the frozen experiment, baseline, and prompt source hashes before running. Policy randomization uses the additional name `finite_ap_index`, providing its own action RNG namespace while retaining paired worlds.

Results go to `artifacts/overnight_v2/episodes_index.jsonl`, separate from the frozen panel. A separate `index_manifest.json` records the original manifest hash, this implementation's hash, prior, arithmetic, index tolerance, tie rule, and exploratory status. Completed episode identifiers are skipped on resume. Analysis can explicitly merge these records as an exploratory comparator; the main manifest is not rewritten.

```bash
PYTHONPATH=src .venv/bin/python -m jevbandits.finite_index --benchmark
PYTHONPATH=src .venv/bin/python -m jevbandits.finite_index \
  --run-dir artifacts/overnight_v2
.venv/bin/pytest -q tests/test_finite_index.py
```

There are 200 E3 episodes, 600 E4 episodes, and 240 E5 episodes, totaling 88,000 additional offline decisions. The E3 outcomes should be interpreted against exact Bellman loss and paired pseudo-regret. E4/E5 comparisons with Bayes-UCB, knowledge gradient, and IDS test whether the arm-specific stopping calculation helps in settings where the full joint DP is unavailable. Any favorable comparison remains exploratory because the comparator was added after initial results were inspected.

## A concrete counterexample to global optimality

The original two-pull diagnostic anchor also distinguishes the finite index from joint planning. For Beta(11,9) versus Beta(1,1), the two-pull indices are

$$
\lambda_A=\frac{(11/20)(1+12/21)}{1+11/20}=\frac{121}{217}\approx0.557604,
\qquad
\lambda_B=\frac{(1/2)(1+2/3)}{1+1/2}=\frac59\approx0.555556.
$$

Thus the index policy chooses A, whereas the exact joint values are $Q_A=1.10$ and $Q_B=1.108\overline3$, favoring B. The index brackets at tolerance $10^{-6}$ do not overlap, so this is a planning approximation rather than numerical ambiguity. Knowledge gradient has the exact action ranking at two pulls and chooses B on this fixture.

This does not imply that knowledge gradient dominates the index in aggregate: the index's observed E3 loss is smaller over the 200 prior-drawn episodes. The example illustrates why both exact fixtures and representative closed-loop evaluations are needed, and why no single Bayesian heuristic should be treated as universal ground truth.
