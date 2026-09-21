# Numerical audit of the Monte Carlo IDS baseline

Date: 2026-09-20. This audit was added after the five-study protocol was frozen. It makes no Jev calls and does not alter the original IDS implementation or any experimental policy. Its purpose is to determine whether the 2,048 posterior draws used by the existing IDS baseline provide an adequate approximation on the study's two-arm posterior fixtures.

## Quadrature reference

For two independent Beta posteriors, define $A^*=\arg\max_i\theta_i$, $w_i=\Pr(A^*=i)$, $m_i=\mathbb{E}[\theta_i]$, and

$$
c_{ij}=\mathbb{E}[\theta_j\mathbf{1}\{A^*=i\}].
$$

Writing $f_i$ and $F_i$ for the Beta density and CDF gives one-dimensional integrals:

$$
w_i=\int_0^1 f_i(x)F_j(x)\,dx,
\qquad c_{ii}=\int_0^1 x f_i(x)F_j(x)\,dx,
$$

$$
c_{ij}=\int_0^1 f_i(x)m_j I_x(a_j+1,b_j)\,dx,\quad i\ne j,
$$

where $I_x$ is the regularized incomplete beta function. The conditional probability of a success at arm $j$, given $A^*=i$, is $c_{ij}/w_i$. The IDS posterior regret and information gain are therefore

$$
\Delta_j=\sum_i c_{ii}-m_j,
$$

$$
g_j=H(m_j)-\sum_i w_iH(c_{ij}/w_i)
=\sum_i w_i\operatorname{KL}\!\left(\operatorname{Bern}(c_{ij}/w_i)\,\|\,\operatorname{Bern}(m_j)\right).
$$

Here $H(p)=-p\log p-(1-p)\log(1-p)$, using natural logarithms, so $g_j$ is in nats. The implementation evaluates both information identities and uses the conditional-KL form for scoring to avoid cancellation of nearly equal entropy terms when one best-arm category is rare.

The reference uses SciPy adaptive quadrature with absolute tolerance $10^{-13}$ and relative tolerance $10^{-11}$. This is a high-accuracy numerical reference, **not symbolic exactness**. Records retain integration-error estimates, raw probability-mass and first-moment residuals, and entropy/KL disagreement. Residuals above $10^{-10}$ stop the audit. Tiny integration residuals are corrected by normalizing $w$ and scaling each first-moment column to its known analytic marginal mean. A numerically lost best-arm category stops the audit instead of being silently declared impossible.

For two uniform priors, the analytic checks are $w=(1/2,1/2)$, $\Delta=(1/6,1/6)$, $g_j=\log 2-H(2/3)$, and

$$
c=\begin{pmatrix}1/3&1/6\\1/6&1/3\end{pmatrix}.
$$

Additional tests compare $w$ with the independent finite Beta-function probability formula, compare $\mathbb{E}[\max_i\theta_i]$ with $\int_0^1[1-F_1(x)F_2(x)]\,dx$, verify label symmetry and rare-category behavior, and compare the reference mixture optimizer with a dense independent one-dimensional grid.

## Sensitivity design

The full audit uses the existing **100 E1 and 100 E2 posterior fixtures**, preserving their identities and duplicates. At each fixture, estimate IDS statistics with $M\in\{128,2048,32768\}$ posterior samples, using **30 independently seeded repetitions per sample count**. The resulting 18,000 fitted action mixtures are scored under the quadrature reference.

For mixture $p$, define

$$
\Psi(p)=\frac{(p^\top\Delta)^2}{p^\top g},
\qquad
\operatorname{Excess}(p)=\Psi(p)-\min_{q\in\mathcal{S}_2}\Psi(q).
$$

The reference optimizer is the already tested two-action IDS mixture optimizer, supplied with the quadrature statistics. The primary diagnostics are the median, 95th percentile, mean, and maximum excess true information ratio. Comparing total-variation distance with one arbitrary reference mixture would be misleading: when both arms have identical posterior distributions, every mixture has the same information ratio. An approximate policy should receive zero objective penalty if it chooses another minimizer.

The audit defines $0/0=0$ when expected regret is zero; positive regret with zero information has infinite ratio. The finite Beta fixtures have positive posterior uncertainty under the reference, but their Monte Carlo approximation can assign all sampled optimality mass to one arm. For each replicate the audit records missing optimal-arm categories, the zero-information fallback, and whether a category has expected sample count below one, $M\min_iw_i<1$. Very small negative excess caused by optimizer tie tolerance is retained as a raw diagnostic and clipped to zero for summary tables; an apparent improvement over the reference beyond $10^{-9}\max(1,|\Psi^*|)$ stops the run.

Every record also includes the expected one-step exact Bayesian action-value loss at remaining horizon 10. This is a **different objective**: minimizing IDS's information ratio need not minimize finite-horizon Bellman loss. Its inclusion must not turn an IDS approximation audit into a claim that exact IDS is Bayes-optimal.

The fixtures are selected states, not fresh independent bandit environments. Aggregate quantiles describe approximation error across this state panel and Monte Carlo randomness. They are not confidence intervals for sequential regret, and adequate performance on two-arm fixtures does not certify the 10- or 15-arm approximation.

## Running and provenance

Only unit tests and a bounded four-fixture smoke run were executed during construction. The smoke used three repetitions per sample count, completed successfully, and verified the complete record/report path. Its results are not a sensitivity-study conclusion.

```bash
.venv/bin/pytest -q tests/test_ids_audit.py
PYTHONPATH=src .venv/bin/python -u -m jevbandits.ids_audit \
  --output-dir artifacts/overnight_v2/ids_audit
```

The full run is ready for delegated monitoring. It writes a separate manifest, resumable fixture-level `sensitivity.jsonl`, and generated `report.md` under the ignored output directory. The manifest records all sample sizes, repeats, quadrature tolerances, fixture-list hash, seed, and source hashes. It rejects changed configurations or methods on resume. The original Jev run manifest and baseline data remain unchanged.

The methodological basis is Russo and Van Roy's [IDS paper](https://arxiv.org/abs/1403.5556), particularly the Beta-Bernoulli information-ratio calculation and SampleIR approximation in Section 6. This audit directly checks the approximation employed by this project rather than substituting a weaker comparison policy after observing results.
