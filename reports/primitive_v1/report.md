# E10: forecast target and elicitation format

Recorded 16,560 questions; 1,440 complete fixture/condition packages; 0 observed incomplete packages. Planned: 16,560 questions and 30 fixtures per K = 2, 5, 15. Entirely absent conditions appear in primary completeness counts, not the observed-incomplete table.

The common endpoint is mean binary excess Brier risk: 2 times mean squared event-probability error across arms and repeats. The best-arm Choice vector receives 2/K times its squared-error sum. Separate Noul best-arm probabilities remain unnormalized for every primary calculation. This is excess conditional proper-scoring risk, not realized reward or empirical outcome accuracy.

## Prespecified primary contrasts under full Bayesian summaries

| target | left | right | representation | metric | mean_difference | ci95_low | ci95_high | ci97_5_low | ci97_5_high | complete | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| next_reward | choice | noul | bayes | mean_binary_excess_brier | 0.0508 | 0.0465 | 0.0552 | 0.0459 | 0.0558 | True | prespecified full design |
| best_arm | noul | choice | bayes | mean_binary_excess_brier | -0.0572 | -0.0702 | -0.0452 | -0.0721 | -0.0437 | True | prespecified full design |

| target | k | n_left | n_right | n_pairs | expected |
| --- | --- | --- | --- | --- | --- |
| next_reward | 2 | 30 | 30 | 30 | 30 |
| next_reward | 5 | 30 | 30 | 30 | 30 |
| next_reward | 15 | 30 | 30 | 30 | 30 |
| best_arm | 2 | 30 | 30 | 30 | 30 |
| best_arm | 5 | 30 | 30 | 30 | 30 |
| best_arm | 15 | 30 | 30 | 30 | 30 |

Differences are left minus right; lower risk is better. Bootstrap whole paired fixtures within K, then weight the three K strata equally. Pointwise 95% and Bonferroni 97.5% intervals are shown for the two-contrast family; the latter target simultaneous 95% coverage subject to bootstrap approximation. Incomplete-design estimates are descriptive. A zero-crossing interval is not equivalence.

## Descriptive risks

| target | primitive | representation | k | mean_binary_excess_brier | mean_absolute_error | n_fixtures |
| --- | --- | --- | --- | --- | --- | --- |
| best_arm | choice | bayes | 2 | 0.0858 | 0.1668 | 30 |
| best_arm | choice | bayes | 5 | 0.0922 | 0.1423 | 30 |
| best_arm | choice | bayes | 15 | 0.0458 | 0.0692 | 30 |
| best_arm | choice | counts | 2 | 0.0696 | 0.1461 | 30 |
| best_arm | choice | counts | 5 | 0.0989 | 0.1395 | 30 |
| best_arm | choice | counts | 15 | 0.0302 | 0.0569 | 30 |
| best_arm | choice | oracle_probs_full | 2 | 0.1330 | 0.2042 | 30 |
| best_arm | choice | oracle_probs_full | 5 | 0.1072 | 0.1553 | 30 |
| best_arm | choice | oracle_probs_full | 15 | 0.0511 | 0.0731 | 30 |
| best_arm | choice | oracle_probs_only | 2 | 0.1321 | 0.2032 | 30 |
| best_arm | choice | oracle_probs_only | 5 | 0.1100 | 0.1569 | 30 |
| best_arm | choice | oracle_probs_only | 15 | 0.0523 | 0.0736 | 30 |
| best_arm | noul | bayes | 2 | 0.0108 | 0.0539 | 30 |
| best_arm | noul | bayes | 5 | 0.0213 | 0.0628 | 30 |
| best_arm | noul | bayes | 15 | 0.0199 | 0.0474 | 30 |
| best_arm | noul | counts | 2 | 0.0216 | 0.0717 | 30 |
| best_arm | noul | counts | 5 | 0.0440 | 0.0919 | 30 |
| best_arm | noul | counts | 15 | 0.0198 | 0.0494 | 30 |
| best_arm | noul | oracle_probs_full | 2 | 0.0076 | 0.0436 | 30 |
| best_arm | noul | oracle_probs_full | 5 | 0.0027 | 0.0284 | 30 |
| best_arm | noul | oracle_probs_full | 15 | 0.0017 | 0.0259 | 30 |
| best_arm | noul | oracle_probs_only | 2 | 0.0088 | 0.0533 | 30 |
| best_arm | noul | oracle_probs_only | 5 | 0.0054 | 0.0453 | 30 |
| best_arm | noul | oracle_probs_only | 15 | 0.0041 | 0.0403 | 30 |
| next_reward | choice | bayes | 2 | 0.0598 | 0.1576 | 30 |
| next_reward | choice | bayes | 5 | 0.0583 | 0.1522 | 30 |
| next_reward | choice | bayes | 15 | 0.0442 | 0.1302 | 30 |
| next_reward | choice | counts | 2 | 0.0356 | 0.1132 | 30 |
| next_reward | choice | counts | 5 | 0.0204 | 0.0806 | 30 |
| next_reward | choice | counts | 15 | 0.0212 | 0.0852 | 30 |
| next_reward | choice | oracle_probs_full | 2 | 0.0186 | 0.0791 | 30 |
| next_reward | choice | oracle_probs_full | 5 | 0.0129 | 0.0659 | 30 |
| next_reward | choice | oracle_probs_full | 15 | 0.0087 | 0.0531 | 30 |
| next_reward | choice | oracle_probs_only | 2 | 0.0312 | 0.1066 | 30 |
| next_reward | choice | oracle_probs_only | 5 | 0.0215 | 0.0830 | 30 |
| next_reward | choice | oracle_probs_only | 15 | 0.0188 | 0.0782 | 30 |
| next_reward | noul | bayes | 2 | 0.0040 | 0.0379 | 30 |
| next_reward | noul | bayes | 5 | 0.0041 | 0.0404 | 30 |
| next_reward | noul | bayes | 15 | 0.0021 | 0.0281 | 30 |
| next_reward | noul | counts | 2 | 0.0030 | 0.0332 | 30 |
| next_reward | noul | counts | 5 | 0.0030 | 0.0330 | 30 |
| next_reward | noul | counts | 15 | 0.0026 | 0.0281 | 30 |
| next_reward | noul | oracle_probs_full | 2 | 0.0044 | 0.0383 | 30 |
| next_reward | noul | oracle_probs_full | 5 | 0.0030 | 0.0296 | 30 |
| next_reward | noul | oracle_probs_full | 15 | 0.0031 | 0.0326 | 30 |
| next_reward | noul | oracle_probs_only | 2 | 0.0057 | 0.0429 | 30 |
| next_reward | noul | oracle_probs_only | 5 | 0.0048 | 0.0351 | 30 |
| next_reward | noul | oracle_probs_only | 15 | 0.0038 | 0.0343 | 30 |

Per-K primitive contrasts for every representation, with exploratory pointwise 95% intervals and missing-pair counts, are in secondary_primitive_effects.csv. They are outside the two overall primary contrasts.

## Interpretation limits

Reward Choice and Noul use identical binary event wording and rubric, with the type field changed. Best-arm Choice elicits one joint categorical distribution, whereas Noul elicits separate marginal events; this contrast includes question scope and joint-coherence demands as well as primitive. Known-probability controls test following explicit event probabilities, with and without counts and moments. These bundled changes cannot identify an internal architectural mechanism or prove that all joint inference fails. No action recommendation is tested.

Raw Noul probability sums and optional normalized best-arm risks are secondary outputs in best_noul_coherence.csv. Zero-sum vectors have undefined normalized risk and are not replaced with a uniform distribution. Secondary log risk clips each binary distribution at 1e-6 and renormalizes, as in E7; this clipping does not affect Brier risk. Repeated calls and arms are dependent and are averaged within independent fixtures.
