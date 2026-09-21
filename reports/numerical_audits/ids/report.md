# Monte Carlo IDS numerical sensitivity

Completed fixtures: **200**; Monte Carlo replicates: **18000**.

Excess information ratio is evaluated with quadrature-derived true posterior moments, relative to the minimum of that same objective. It is not total-variation distance to an arbitrary optimal mixture. Fixtures may repeat posterior states; these are descriptive numerical diagnostics, not independent environment effect estimates.

| Posterior draws | Median excess | 95th percentile | Mean excess | Maximum excess | Missing category | Rare-category rows |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0 | 0.015571849 | 0.0045677069 | 0.84398174 | 9.93% | 11.00% |
| 2048 | 0 | 0.00086055281 | 0.00026144833 | 0.074298205 | 4.62% | 5.00% |
| 32768 | 0 | 5.4793577e-05 | 2.3431176e-05 | 0.0055393181 | 1.25% | 1.50% |

A rare category has fewer than one expected posterior sample at the indicated draw count. Missing categories can force an empirical zero-information fallback even when the true posterior retains uncertainty.

## Largest excess-ratio cases at 2,048 draws

- e2_assistance:45, replicate 23: excess 0.074298205; mixture [0.055835162549086756, 0.9441648374509133]; missing category False.
- e1_horizon:17, replicate 19: excess 0.043773617; mixture [0.7458641183597561, 0.2541358816402439]; missing category False.
- e2_assistance:29, replicate 19: excess 0.031415704; mixture [1.0, 0.0]; missing category False.
- e2_assistance:30, replicate 9: excess 0.029918308; mixture [0.06932646913847262, 0.9306735308615274]; missing category False.
- e2_assistance:29, replicate 4: excess 0.027946311; mixture [0.11361494655035888, 0.8863850534496411]; missing category False.
- e2_assistance:29, replicate 17: excess 0.021260011; mixture [0.9529696568198278, 0.047030343180172185]; missing category False.
- e1_horizon:55, replicate 19: excess 0.020694413; mixture [0.6270614920965382, 0.37293850790346184]; missing category False.
- e2_assistance:70, replicate 16: excess 0.019038693; mixture [0.4772979605477467, 0.5227020394522532]; missing category False.
- e2_assistance:29, replicate 2: excess 0.018600283; mixture [0.9378766589504097, 0.062123341049590275]; missing category False.
- e2_assistance:30, replicate 8: excess 0.018522121; mixture [0.10035426670959036, 0.8996457332904096]; missing category False.

Maximum quadrature error estimate: 8.33e-12. Maximum raw first-moment residual: 5.55e-16.

The records also contain expected one-step Bayes-optimal decision loss at horizon 10. That is a separate planning objective; minimizing the IDS information ratio need not minimize this loss.
