# E7 repeat variability versus reference error

This posthoc numerical audit uses the two independently requested repeats of each identical forecast question. It does not query the API or infer an internal sampling mechanism.

| target | representation | raw_excess_brier | repeat_variability | cross_repeat_error_product | variability_fraction |
| --- | --- | --- | --- | --- | --- |
| best_arm | bayes | 0.230945 | 0.000403 | 0.230542 | 0.001746 |
| best_arm | counts | 0.195382 | 0.001298 | 0.194084 | 0.006644 |
| best_arm | means | 0.233372 | 0.000471 | 0.232901 | 0.002018 |
| next_reward | bayes | 0.002256 | 0.000067 | 0.002189 | 0.029611 |
| next_reward | counts | 0.004733 | 0.000126 | 0.004607 | 0.026609 |
| next_reward | means | 0.002981 | 0.000067 | 0.002914 | 0.022368 |

For vectors $p_1,p_2$ and analytic reference $q$, the exact arithmetic identity is

$$rac{\|p_1-q\|^2+\|p_2-q\|^2}{2}=rac{\|p_1-p_2\|^2}{2}+(p_1-q)^	op(p_2-q).$$

Under independent, stationary repeat errors conditional on a fixed question, the first right-hand term estimates summed response variance and the second estimates squared deviation of the mean response from the reference. The cross-product estimate may be negative in a finite sample and is not clipped. If repeat errors are correlated, its expectation also includes that covariance. Repeat variability may include service or batch effects; it is not automatically Monte Carlo sampling inside Jev.

Arms are averaged within each fixture before 10,000-resample bootstrap intervals. The overview equally weights arm-count cells and is descriptive. The decomposition does not turn repeated API calls into independent task samples. Near-zero repeat variability cannot rule out shared service drift or systematic batch effects.

The observed Noul probabilities lie on a .01 grid. Nearest rounding of an otherwise exact scalar reference to that grid would change it by at most .005; this alone does not explain larger errors. No particular internal quantization or sampling procedure is assumed.
