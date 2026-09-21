# Follow-up interpretation: assistance, probability elicitation, and sampling

Analysis snapshot: 2026-09-21. E6–E10 and the dedicated batch-control audit are complete.

The follow-ups show that Jev's use of supplied analysis depends on its presentation, the probability question and API primitive, and the rule that converts its output into an action. Explicit posterior means improved online decisions on fresh tasks; additional uncertainty summaries worsened the prespecified average relative to means alone. Exact action values remained an effective aid in fixed-state tasks. Advice identifiers were sensitive to presentation, while randomized selection from Jev's reported action distribution helped in the ten-arm regime. Probability fidelity also differed substantially between elicitation packages, even when exact target probabilities were supplied. These are distinct capabilities and interventions, rather than one measure of “Bayesian ability.”

All findings concern pinned `jev-1.13.0`, stationary Bernoulli arms, and the stated independent Beta priors. Intervals below are 95% unless explicitly marked 97.5%. Repeated questions and arm components are averaged within fixtures; online comparisons pair whole tasks. Neither API calls nor individual pulls increase the inferential sample size.

| Study | Completed evaluation | Main inferential unit |
|---|---|---|
| E6: advice presentation | 28,800 decisions across formats, horizons and label assignments | 100 random fixtures and 100 separately analyzed targeted fixtures |
| E7: probability forecasts | 24,000 questions | 100 fixtures at each of five arm counts |
| E8: online evidence ladder | 960 Jev episodes, 96,000 decisions | 80 paired tasks in each of four cells; 320 tasks total |
| E9: sampling regimes | 400 Jev episodes, 40,000 decisions | 100 paired tasks at each of two arm counts |
| E10: elicitation and known-probability controls | 16,560 questions | 30 fixtures at each of three arm counts; 90 total |
| Batch control | 1,920 decisions, 960 matched source questions | Ten reused E6 fixtures per cohort |

## E6: the presentation of valid advice matters

E6 separated a nested recommendation from an explicit reference to its field, an inline instruction, per-option advice, advice without numerical evidence, and exact action values. At horizon 10, explicitly naming the recommendation's path increased adherence over nested advice in the random cohort by **12.75 percentage points [7.75, 18.25]** with the original arm IDs and **12.50 points [7.50, 18.00]** with A/B labels. These are the prespecified descriptive, paired-fixture comparisons. [E6 contrasts](e6_advice/advice_contrasts.csv)

The targeted cohort makes the distinction clearer. Its states were selected using exact values so that learning from the lower-mean, unobserved arm is optimal at horizon 10. They were not selected using Jev responses and are not representative of general online states.

| Format, targeted cohort at horizon 10 | Adherence: original IDs | Adherence: A/B labels |
|---|---:|---:|
| Nested recommendation | 0% | 0% |
| Inline “choose exactly” instruction | 0% | 0% |
| Advice in each option | 1.00% | 1.00% |
| Explicit reference to the recommendation field | 56.75% | 84.25% |
| Advice without numerical evidence | 100% | 100% |

For the explicit-path intervention, the paired adherence gains over nested advice were **56.75 points [49.75, 63.50]** and **84.25 points [80.25, 88.25]**, respectively. The secondary, within-fixture A/B-minus-original label contrast was **27.50 points [22.50, 32.75]**. Exact-value inputs achieved 100% optimal-action agreement on these targeted states; they contain no advice pointer and therefore have **no advice-adherence endpoint**. [E6 summaries](e6_advice/advice_summary.csv), [label contrasts](e6_advice/advice_interactions.csv)

Binary failure rates should be read alongside the size of the decision error. In the A/B targeted condition, nested advice incurred mean local exact loss **0.05954 reward units [0.05633, 0.06276]**; the explicit-path version reduced this to **0.00727 [0.00552, 0.00910]**. These values concern one action followed by optimal continuation, not a measured ten-step Jev trajectory. Observed 0% or 100% rates and degenerate bootstrap intervals do not establish certainty on new fixtures.

The supported inference is that these communication packages matter: an explicit field reference helped where merely restating an ID inline did not, and removing numerical evidence also changed behavior. This is consistent with an interaction between advice presentation and other content. It does not identify an internal pointer-resolution mechanism or isolate numerical evidence, length, layout, and instruction wording as separate causes.

The completed batch audit reduces one concern. On its ten targeted fixtures, nested and inline advice had 0% adherence both alone and in mixed requests; advice-only had 100% adherence and exact-value inputs had 100% optimal agreement in both modes. Explicit-path adherence was 78.75% alone and 77.50% mixed. Thus the gross format pattern persisted in standalone calls on this slice. Across all formats, mean canonical probability-vector total variation was **0.02667 [0.02508, 0.02808]** in the targeted cohort and **0.01435 [0.00965, 0.01938]** in the random cohort. These quantities include service variability. The audit reused only twenty fixtures; its fixture bootstrap does not model cross-fixture dependence from shared mixed requests. It cannot establish general batching invariance or independence. [Batch-control audit](batch_control_v1/report.md)

## E7: useful reward forecasts and problematic best-arm distributions

E7 asked explicitly for probabilities, rather than treating action-choice probabilities as beliefs. Next-reward questions used the binary Noul primitive and only the queried arm's evidence. Latent-best questions used a joint, multi-option Choice with all arms' evidence. This target/primitive/input-scope confounding is central to interpretation.

Reward forecasts were reasonably close to the specified posterior predictive probabilities. The descriptive equal-arm-count mean absolute error was about **0.0350 with counts, 0.0337 with means, and 0.0292 with full summaries**. At 15 arms, full summaries reduced binary excess Brier risk relative to counts by **0.002751 [0.002367, 0.003153]**. Because the supplied mean already determines the reward probability, this shows an effect of the representation package rather than additional statistical evidence. [Forecast summaries](forecast_v1/forecast_summary.csv), [paired contrasts](forecast_v1/forecast_paired_effects.csv)

For the latent-best Choice target, full summaries did not reliably improve distributional accuracy. At 15 arms, categorical excess Brier risk was **0.2863 [0.2476, 0.3296]** with counts, **0.3826 [0.3374, 0.4285]** with means, and **0.4030 [0.3594, 0.4493]** with full summaries. Full summaries minus counts increased risk by **0.1167 [0.0826, 0.1484]** on paired fixtures. The simple normalized-posterior-means heuristic had risk **0.1363 [0.1186, 0.1557]** in this cell, although that heuristic is not the correct Bayesian best-arm distribution. [Forecast report and controls](forecast_v1/report.md)

The reliability summaries show overconcentration in the joint Choice package: for example, the full-summary highest-probability bin averaged prediction 0.983 versus analytic reference 0.633. This is a pooled, descriptive component-level comparison, not an independent-arm statistical test. Categorical and binary Brier scores should not be compared as a common difficulty scale. These scores measure discrepancies from analytic conditional probabilities; they are not observed outcome accuracy or online regret. [Conditional reliability](forecast_v1/conditional_reliability.csv)

The repeat audit argues against ordinary repeat-to-repeat fluctuation being the dominant observed error component. For references \(q\) and repeat outputs \(p_1,p_2\),

$$
\frac{\lVert p_1-q\rVert_2^2+\lVert p_2-q\rVert_2^2}{2}
=\frac{\lVert p_1-p_2\rVert_2^2}{2}
+(p_1-q)^\top(p_2-q).
$$

The first term on the right accounted descriptively for **0.17%–0.66%** of latent-best excess Brier risk, depending on representation, and **2.24%–2.96%** for next-reward forecasts. Most observed error therefore persisted across these two repeats. Calling the cross-product “squared bias” requires conditional independence and stationarity; correlated errors add covariance. The decomposition does not reveal internal Monte Carlo sampling, rule out common service drift, or prove a deterministic mechanism. [Repeat audit](forecast_v1/repeats/report.md)

E7 alone supports **good relative performance on this binary reward-forecast package, with substantial reference error on this joint latent-best Choice package**. E10, below, addresses the primitive confound and substantially narrows what can be claimed about joint inference.

## E8: means help online, while richer summaries are not monotonically better

E8 evaluated the E2-motivated evidence ladder on fresh trajectories: three or ten arms, close-winner or matched-prior environments, and 100 pulls. Its two overall contrasts were fixed before collection and equally weight the four cells.

| Prespecified contrast, cumulative pseudo-regret | Estimate | Pointwise 95% interval | Bonferroni 97.5% interval |
|---|---:|---:|---:|
| Means minus counts | -2.784 | [-3.925, -1.748] | [-4.136, -1.618] |
| Full Bayesian summaries minus means | +0.833 | [+0.375, +1.356] | [+0.321, +1.434] |

Negative differences favor the first policy. Both family-adjusted intervals retain their respective directions. The 97.5% intervals target simultaneous 95% coverage for these **two** contrasts, subject to the bootstrap approximation; they do not cover all secondary comparisons. [E8 primary results](e8_evidence/evidence_primary.csv)

The effects were concentrated in the prior-family tasks. At ten arms, means minus counts was **-9.379 [-13.534, -5.583]**, while full summaries minus means was **+2.266 [+0.912, +3.903]**. Close-winner cell intervals included zero. These subgroup intervals are exploratory, and their differences should not be promoted to an additional confirmatory interaction test. [E8 per-cell results](e8_evidence/evidence_per_cell.csv)

The descriptive behavior is also instructive. In the ten-arm prior cell, means-direct chose a posterior-mean maximizer on 94.25% of pulls and visited 2.74 arms on average; counts-direct was greedy on 77.18% and visited 2.09 arms. Full summaries gave 90.74% and 2.50 arms. Thus “more non-greedy” did not automatically mean broader exploration. These policies followed different histories, so the summaries are not a matched-state mechanism test. [Episode summaries](e8_evidence/evidence_summary.csv)

Means assistance did not eliminate the classical-method gap: means-direct minus Bayes-UCB was **+2.346 [+1.319, +3.431]**, while its difference from Thompson sampling was **+0.412 [-0.576, +1.462]**. The latter is inconclusive, not evidence of equivalence. Full uncertainty summaries could be less useful than means alone because of their contents, layout, length, or induced action preferences; this experiment does not isolate which. [Secondary baseline comparisons](e8_evidence/evidence_baseline_comparisons.csv)

## E9: a sampling interaction replicates, without proving a sign reversal

E9 held the Bayesian-summary prompt fixed and changed only how its response became an action: valid backend choice versus a local draw from the normalized reported distribution. The fresh matched-prior tasks used 100 pulls and 100 paired tasks at each arm count. The two arm-count strata were independently drawn.

With \(\Delta_K\) denoting sample-minus-direct mean pseudo-regret, the sole primary contrast was \(\Delta_{10}-\Delta_2\):

| Contrast | Estimate | 95% interval | Status |
|---|---:|---:|---|
| Sample minus direct, two arms | +0.843 | [-0.372, +1.875] | Secondary |
| Sample minus direct, ten arms | -5.441 | [-7.960, -3.113] | Secondary |
| Ten-arm effect minus two-arm effect | **-6.284** | **[-8.986, -3.700]** | Sole planned primary interaction |

The operational interaction was reproduced on fresh tasks. The two-arm interval includes zero, so these results do **not** establish a confirmed sign reversal or a crossover threshold. Increasing the number of arms also changes options in the input and learning difficulty under a fixed budget. [E9 protocol](../docs/e9_sampling_protocol.md), [recorded primary analysis](../results/sampling_v1/interaction_summary.json)

At ten arms, sampling increased descriptive arm coverage from 2.55 to 7.14 and switching from 1.75 to 16.66 changes per episode. Mean pseudo-regret fell from 18.265 to 12.824. This is consistent with sampling mitigating narrow commitment, but is not a mediation analysis. The sampled policy still had more regret than Bayes-UCB (8.926) and the finite-horizon AP index (7.745). Its improvement over direct Jev does not establish superiority over classical Bayesian algorithms or imply that its probabilities implement Thompson sampling. [E9 episode records](../results/sampling_v1/episodes.csv.gz)

![Fresh online evidence and sampling follow-ups](followup_comparison.png)

*The panels use separate task samples and different interventions. Their magnitudes are not a head-to-head comparison. E8 shows both specified interval levels; E9 distinguishes its single primary interaction from secondary per-arm-count effects.*

## E10: elicitation explains an important part of forecast fidelity

E10 used 90 fresh fixtures and scored all packages on the common mean binary-event excess Brier scale,

$$
\frac{2}{K}\sum_i(p_i-q_i)^2.
$$

Under full Bayesian summaries, its two prospectively specified, equal-arm-count contrasts were:

| Primary contrast | Estimate | Pointwise 95% interval | Bonferroni 97.5% interval |
|---|---:|---:|---:|
| Next-reward binary Choice minus Noul | +0.05075 | [+0.04649, +0.05516] | [+0.04595, +0.05584] |
| Latent-best marginal Noul minus joint Choice | -0.05721 | [-0.07024, -0.04519] | [-0.07213, -0.04369] |

Lower risk is better, so both comparisons favor Noul in their respective packages. The reward bridge holds the event question, evidence, and binary rubric fixed and changes only the request primitive. TypeSafe explicitly recommends **Noul for yes/no questions**; binary Choice is an intentional cross-interface control, not the recommended way to implement that binary task. [E10 primary results](primitive_v1/primary_effects.csv), [official Noul documentation](https://docs.typesafe.ai/primitives/noul), [Choice documentation](https://docs.typesafe.ai/primitives/choice)

The latent-best comparison also changes scope: one joint categorical distribution versus separate marginal-event questions. It requires **K Noul questions instead of one joint Choice** per fixture/repeat, so improved fidelity comes with a different query and computation budget; the measured risk contrast is not a cost-matched superiority claim. The reward bridge uses the same number of questions. Separate Noul probabilities were not normalized before primary scoring.

Known-probability controls strengthen the interface interpretation. Supplying exact latent-best probabilities alongside the summaries did not eliminate Choice's discrepancy: its descriptive mean risks were **0.1330, 0.1072, and 0.0511** at 2, 5, and 15 arms, versus **0.0858, 0.0922, and 0.0458** without those supplied probabilities. Providing only the exact probabilities likewise left substantial error. For Noul, supplying probabilities with summaries changed the corresponding means from **0.0108, 0.0213, 0.0199** to **0.0076, 0.0027, 0.0017**. These are secondary descriptive control results, not additional adjusted primary tests. [E10 risk summary](primitive_v1/risk_summary.csv)

Since the target quantities were supplied in those controls, failure to compute them from counts cannot be the sole explanation for the remaining output discrepancy. That does **not** prove an arithmetic failure, identify an internal architecture, or show that all joint Bayesian inference fails. The controls also change instructions and representation, and the returned probabilities may behave differently across elicitation interfaces.

Better marginal scores do not guarantee a coherent categorical forecast. With full summaries at 15 arms, the Noul latent-best probabilities summed to **1.466 on average**, although the reference events are mutually exclusive and exhaustive. Normalized marginal forecasts are a separate, secondary analysis; they are not the primary result and have not been shown here to improve online control. The batch validation examined E6 advice formats, not this full mixed-primitive forecast design. [Noul coherence diagnostics](primitive_v1/best_noul_coherence.csv)

![Forecast elicitation and known-probability controls](elicitation_controls.png)

*Panel A shows the two primary contrasts and both specified interval levels. Panel B shows secondary descriptive latent-best risks, without inferential intervals. Lower risk is better; separate Noul outputs are unnormalized. The known-probability conditions test the use of supplied event probabilities, not access to the hidden winning arm.*

## Synthesis with the core experiments and remaining limits

The core study's strongest positive control was successful use of supplied exact action values, including in sequential two-arm tasks. E6 strengthens the practical point that external analysis can be useful while showing that a valid recommendation identifier is not automatically used. E8 establishes an online benefit from posterior means on fresh tasks, with a measurable penalty from the particular richer summary package relative to means alone. E9 establishes that the action-selection rule can materially change performance in a task-dependent way. E7 and E10 keep probability elicitation separate from these decision results and show why the API primitive belongs in the experimental specification.

Together, these findings favor evaluating a harness as a combination of statistical computation, representation, elicitation, and action selection. They do not justify assigning one global capability label to Jev or identifying its internal algorithm. In particular:

- Exact-value assistance supplies planning results; success with it is not autonomous computation of those values.
- Sufficient statistics and their summaries contain the same conditional evidence, but adaptive policies subsequently observe different data because their actions differ.
- The batch audit shows that the large advice-format contrast survives standalone calls on a small reused subset. It does not prove independence across questions or general batch invariance.
- E10 controls the binary reward primitive cleanly at the request level; its latent-best contrast still changes joint versus marginal elicitation, coherence requirements, and question count. Marginal accuracy alone does not make a coherent or cost-matched categorical predictor.
- Length-matched or content-matched controls would be needed to attribute E8's richer-summary penalty specifically to uncertainty information.
- E9 confirms a fixed-budget operational interaction, not an isolated effect of arm count, universal sampling advantage, or general exploration threshold.

The evidence supports a nuanced paper about how external Bayesian calculations and interface choices alter a bounded decision model's behavior. It does not yet establish transfer to nonstationary tasks, Gaussian-process optimization, reasoning-model orchestration, or post-training optimization.

Both comparison figures can be regenerated from exported aggregate results with `python scripts/plot_followup_summary.py` using the repository environment. Their source is [plot_followup_summary.py](../scripts/plot_followup_summary.py).
