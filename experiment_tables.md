---
title: "Jev and Bayesian assistance: experiment tables"
subtitle: "Companion to the research summary"
date: "September 22, 2026"
lang: en-US
---

These tables accompany the [research summary](research_summary.md). Each describes a completed experiment on `jev-1.13.0`; the final table covers the batching validation. Quoted prompts are **excerpts from implemented question templates**, not complete API payloads. Example arm IDs instantiate template variables as `arm_00` or `arm_01`. The shared task specifies stationary binary rewards, independent Beta(1,1) priors, and that remaining choices include the current choice. Structured observations contain the policy's own evidence, never hidden success probabilities.

An *arm* is an option, such as a production process with an unknown success rate. A *fixture* is a fixed evidence state; an *online environment* supports a sequence of choices and new observations. Regret measures expected reward forgone relative to always choosing the truly best arm; lower is better. Local exact loss assesses one choice assuming optimal continuation. Probability-reference risk instead measures squared discrepancy from Bayesian event probabilities.

Intervals resample whole paired environments or fixtures, not individual calls. Unless marked otherwise, intervals are 95%; original-study and secondary comparisons are exploratory. E8 and E10 use 97.5% individual intervals for their respective two planned primary contrasts, targeting 95% coverage within each two-contrast family. Zero observed errors do not establish certainty on new tasks.

## E1 — Horizon and framing

| Field | Description |
|:--|:--|
| Experiment | **E1: horizon/framing.** 100 two-arm fixtures; 4,800 questions. |
| Question | Does Jev respond appropriately to the time available for learning, and does explaining exploration help? |
| Task | Cross 1, 2, 5, and 10 remaining choices with neutral, exploration, and regret wording; reverse labels and repeat calls. Compare choices with exact Bayesian action values. The panel includes six illustrative states and 94 randomly generated states. |
| Example prompts from the task | **Neutral:** “Choose the arm to pull now to maximize expected total reward over all remaining pulls, including this one.” **Exploration adds:** “Pulling an uncertain arm may improve later choices. Explore only when the expected future benefit justifies the immediate reward tradeoff.” Inputs include `remaining_pulls_including_this_one`, successes, and failures. |
| Result | At ten remaining choices, optimal agreement was **84.5% with exploration wording, 77.0% with neutral wording, and 78.25% with regret wording**. Only eight fixtures strictly required a lower-mean action; exploration-worded agreement there was **31.25%**. Jev missed the Beta(11,9)-versus-Beta(1,1) example where the optimal choice changes from the familiar arm at one choice to the untried arm at two. |
| Interpretation | Explaining the value of learning changes behavior, but does not reliably produce optimal exploration. Jev can favor immediate expected success even when a short-term sacrifice is worthwhile. The eight-state diagnostic is too small for a broad frequency claim. |

Sources: [protocol](first_experiment_Plan.md), [prompt templates](src/jevbandits/prompts.py), [results](reports/overnight_v2/report.md).

## E2 — Evidence and calculation assistance

| Field | Description |
|:--|:--|
| Experiment | **E2: assistance ladder.** 100 new two-arm fixtures; 6,000 questions. |
| Question | Which assistance helps: counts, explicit means, uncertainty summaries, exact action values, or a recommended action? |
| Task | Compare five input packages at 1, 2, and 10 remaining choices, with label reversals and repeated calls. Each package retains counts. Score agreement and local loss against exact planning. |
| Example prompts from the task | **Counts/means/uncertainty/action values:** the E1 exploration instruction. Inputs successively add `posterior_mean`, uncertainty fields, and `expected_total_reward_if_chosen_then_optimal`. **Recommendation:** “Choose the recommended arm. The supplied recommendation is from an exact Bayesian calculation maximizing expected total reward over the remaining pulls, given these observations and priors.” |
| Result | With one choice left, means yielded **100% optimal agreement versus 91.5% for counts**. At ten choices, agreement was **93.5% with means, 92.25% with full summaries, and 89.25% with the recommendation package**. Exact action values yielded **zero observed local loss** across tested horizons. |
| Interpretation | Explicit arithmetic can fix ranking errors, and numerical planning outputs are usable. More statistics or a recommended ID do not automatically help. Fixed-state success does not by itself establish better learning over a full episode. |

Sources: [protocol](first_experiment_Plan.md), [prompt templates](src/jevbandits/prompts.py), [results](reports/overnight_v2/report.md), [follow-up motivation](docs/e8_evidence_protocol.md).

## E3 — Sequential decisions with an exact reference

| Field | Description |
|:--|:--|
| Experiment | **E3: exact online control.** 200 independent two-arm environments; 20 choices per policy; 20,000 Jev questions. |
| Question | Do useful fixed-state responses translate into good adaptive decisions when each choice changes subsequent evidence? |
| Task | Compare counts versus full summaries, direct versus sampled execution, and exact-value-assisted direct choice. Policies observe their own outcomes. The matched-prior, two-arm task permits exact dynamic programming. |
| Example prompts from the task | “Choose the arm to pull now to maximize expected total reward over all remaining pulls, including this one.” The exploration sentences follow. Counts and horizon update after each choice; exact assistance adds `expected_total_reward_if_chosen_then_optimal`. Direct and sampled policies use the same instruction; sampling changes execution outside the prompt. |
| Result | Exact assistance produced **zero local loss in 4,000 decisions**. Direct full-summary Jev had **0.511 less episode regret than Thompson sampling [0.365, 0.649]**. Sampling instead increased regret by **0.400 [0.274, 0.522]**. The summary-minus-counts regret difference, **−0.050 [−0.179, +0.065]**, was inconclusive. |
| Interpretation | Jev can execute supplied planning calculations throughout an episode. But the independent exact audit found greedy selection only **0.066 expected reward units below optimal** over 20 choices. Beating Thompson sampling here therefore does not demonstrate sophisticated exploration; randomization can consume choices without enough time to repay the information cost. |

Sources: [protocol](first_experiment_Plan.md), [prompt templates](src/jevbandits/prompts.py), [results](reports/overnight_v2/report.md), [exact policy-value audit](docs/theory_audit.md).

## E4 — Scaling options and difficulty

| Field | Description |
|:--|:--|
| Experiment | **E4: online scaling.** 600 independent environments across 15 cells; 100 choices per policy; 240,000 Jev questions. |
| Question | Do assistance and sampling remain useful as option count and difficulty change? |
| Task | Cross 2, 3, 5, 10, or 15 arms with clear-winner, close-winner, and prior-drawn families; use 40 environments per cell. Compare counts/direct, summaries/direct, counts/sampled, and summaries/sampled against classical policies. The family is hidden from the policy. |
| Example prompts from the task | “Pulling an uncertain arm may improve later choices. Explore only when the expected future benefit justifies the immediate reward tradeoff.” Inputs contain 2–15 arm records. A criterion reads, for example, “Pull arm_00 now.” Direct versus sampled execution does not change the wording. |
| Result | Averaging cells equally, full summaries reduced direct-policy regret by **2.629 [1.952, 3.341]** relative to counts. Sampling the full-summary distribution reduced regret by **1.791 [1.073, 2.525]** relative to direct execution. Direct full-summary Jev nevertheless incurred **2.463 more regret than Bayes-UCB [1.524, 3.410]**; information-directed sampling and knowledge gradient also performed better. |
| Interpretation | With many possible processes, wider exploration can be useful, and accessible summaries can improve decisions. Pooled benefits are not universal across regimes and do not establish superiority to Bayesian algorithms. E8 and E9 tested narrower questions on fresh tasks. |

Sources: [protocol](first_experiment_Plan.md), [prompt templates](src/jevbandits/prompts.py), [results](reports/overnight_v2/report.md), [behavior audit](reports/overnight_v2/behavior/report.md).

## E5 — Strategy instructions and supplied indices

| Field | Description |
|:--|:--|
| Experiment | **E5: strategy interventions.** 240 fresh environments across four cells; 100 choices per policy; 120,000 Jev questions. |
| Question | Can a named strategy or externally calculated exploration index improve decisions beyond ordinary exploration wording? |
| Task | Use 3 or 10 arms in close-winner or prior-drawn environments, with 60 environments per cell. Compare exploration, neutral, regret, and Thompson-style instructions with supplied Bayes-UCB indices; all five policies execute direct choices. |
| Example prompts from the task | **Thompson:** “Choose an arm using the principle of Thompson sampling: an arm should be selected in proportion to its posterior probability of having the highest success probability.” **Index:** “Choose the arm with the largest supplied Bayes-UCB index, breaking equal-index ties arbitrarily.” The latter supplies `bayes_ucb_index`, calculated from the recipient policy's history. |
| Result | Index adherence was approximately **99.93%**; regret fell by **1.768 [0.758, 2.850]** relative to exploration-worded summaries. Regret wording increased regret by **0.763 [0.224, 1.441]**. The Thompson-instruction difference was **+0.064 [−0.435, +0.565]**, and the neutral difference **+0.237 [−0.001, +0.640]**; neither establishes a difference or equivalence. |
| Interpretation | A numerical ranking was more effective than naming a strategy. Requesting Thompson-like behavior while executing a direct choice does not implement posterior sampling. A deterministic selector can also use the supplied indices, so adherence alone does not establish added value from Jev. |

Sources: [protocol](first_experiment_Plan.md), [prompt templates](src/jevbandits/prompts.py), [results](reports/overnight_v2/report.md).

## E6 — Binding advice to an action

| Field | Description |
|:--|:--|
| Experiment | **E6: advice presentation.** 100 random and 100 targeted two-arm fixtures; 28,800 questions. |
| Question | Can changing the presentation of a correct recommendation improve adherence? |
| Task | Cross six formats, three horizons, two label schemes, label reversals, and repeats. Targeted states require exploring the lower-mean arm at ten remaining choices. Compare nested, explicit-lookup, inline, per-option, advice-only, and exact-value formats. |
| Example prompts from the task | **Nested:** “Choose the recommended arm.” with `observation.recommended_arm`. **Explicit lookup:** “Read observation.recommended_arm. Choose the criterion whose ID exactly matches that value.” **Inline example:** “Choose EXACTLY arm_01.” Advice-only removes counts and posterior statistics. |
| Result | In targeted ten-choice states, nested and inline adherence was **0%**. Explicit lookup reached **56.75% with arm_00/arm_01 labels and 84.25% with A/B labels**. Advice-only reached **100% adherence**; exact values reached **100% optimal agreement**. With A/B labels, explicit lookup reduced mean local loss from **0.05954 to 0.00727**. |
| Interpretation | A correct recommendation may be ignored when other evidence favors an immediately attractive option. Presentation and competing content matter, but bundled changes do not isolate an internal mechanism. Selected hard states do not estimate the natural frequency of failure; exact-value agreement is distinct from following an advice pointer. |

Sources: [protocol](docs/e6_advice_protocol.md), [question builder](src/jevbandits/advice_followup.py), [results](reports/e6_advice/report.md), [interpreted contrasts](reports/followup_interpretation.md).

## E7 — Forecasting success and the best option

| Field | Description |
|:--|:--|
| Experiment | **E7: probability forecasts.** 500 fixtures, 100 at each of five arm counts; 24,000 questions. |
| Question | Do explicitly requested probabilities match Bayesian probabilities of success or of an option being best? |
| Task | Compare counts, means, and full summaries with two repeats. Ask one Noul next-reward question per arm using that arm's evidence, and one joint Choice best-arm question using all arms' evidence. Score against analytic or quadrature-derived probabilities. |
| Example prompts from the task | **Reward example:** “On the next pull of arm_00, will the reward be 1? Use the stated Beta(1,1) prior and observed outcomes.” **Best arm:** “Which arm has the highest fixed but unknown success probability?” and “This is a belief about the latent best arm, not a recommendation for the next action.” |
| Result | Full-summary next-reward forecasts had mean absolute error about **0.029**, or 2.9 percentage points. At 15 arms, best-arm Choice categorical excess Brier risk was **0.4030 [0.3594, 0.4493]** with full summaries versus **0.2863 [0.2476, 0.3296]** with counts. A repeat audit found most discrepancy persisted across two calls. |
| Interpretation | Predicting one process's next success and identifying the best process are different tasks. E7 also changes interface and evidence scope, so the contrast cannot isolate joint-inference difficulty. Persistent error does not reveal internal computation. E10 addressed part of the interface confounding. |

Sources: [protocol](docs/e7_forecast_protocol.md), [question builder](src/jevbandits/forecast_followup.py), [results](reports/forecast_v1/report.md), [repeat audit](reports/forecast_v1/repeats/report.md).

## E8 — Fresh online evidence ladder

| Field | Description |
|:--|:--|
| Experiment | **E8: online evidence ladder.** 320 fresh environments; 100 choices per policy; 96,000 Jev questions. |
| Question | Does exposing arithmetic improve sequential reward, and do uncertainty summaries add value beyond means? |
| Task | Use 3 or 10 arms, close-winner or prior-drawn families, and 80 environments per cell. Compare three direct policies receiving counts, counts plus means, or full summaries. Freeze two primary contrasts before collection. |
| Example prompts from the task | All policies receive “Choose the arm to pull now to maximize expected total reward over all remaining pulls, including this one.” followed by the exploration sentences. **Counts:** `successes`, `failures`. **Means adds:** `posterior_mean` and the summary-definition sentence. **Full adds:** `posterior_alpha`, `posterior_beta`, `posterior_sd`, `credible_interval_95`. |
| Result | Means reduced regret relative to counts by **2.784 [1.618, 4.136]**. Full summaries then increased regret relative to means by **0.833 [0.321, 1.434]**. These are **97.5% intervals** for the planned contrasts. Effects were concentrated descriptively in prior-drawn environments. |
| Interpretation | A concise display of expected success rates helped more than the richer dashboard. Counts and the prior already determine the summaries, so the intervention changes accessibility rather than initial information. It does not separate uncertainty content, wording, length, or numerical load. |

Sources: [protocol](docs/e8_evidence_protocol.md), [prompt templates](src/jevbandits/prompts.py), [results](reports/e8_evidence/report.md), [primary contrasts](reports/e8_evidence/evidence_primary.csv).

## E9 — Fresh test of when sampling helps

| Field | Description |
|:--|:--|
| Experiment | **E9: sampling-by-option-count interaction.** 200 fresh environments; 100 choices per policy; 40,000 Jev questions. |
| Question | Is sampling the answer distribution more helpful with ten options than with two? |
| Task | Use 100 matched-prior environments per arm count. Hold summaries and exploration wording fixed. Compare backend choice with a local draw from the normalized probability vector. The sole primary contrast is the ten-arm sampling effect minus the two-arm effect. |
| Example prompts from the task | Both policies use “Pulling an uncertain arm may improve later choices. Explore only when the expected future benefit justifies the immediate reward tradeoff.” There is **no sampling-specific instruction**: execution changes outside the prompt. Later observations can differ because earlier actions differ. |
| Result | The primary interaction was **−6.284 [−8.986, −3.700]** regret units. Sampling reduced ten-arm regret by **5.441 [3.113, 7.960]**; its two-arm increase was **0.843 [−0.372, 1.875]**, individually inconclusive. At ten arms, mean arm coverage rose from **2.55 to 7.14**. |
| Interpretation | Sampling can reduce narrow commitment when many alternatives remain, consistent with the coverage diagnostic. This confirms a regime interaction, not a sign reversal or threshold. Arm count also changes input size and fixed-budget difficulty. Sampling is distinct from Thompson sampling and did not beat the stronger classical controls. |

Sources: [protocol](docs/e9_sampling_protocol.md), [prompt templates](src/jevbandits/prompts.py), [primary interaction](results/sampling_v1/interaction_summary.json), [interpretation](reports/followup_interpretation.md).

## E10 — Elicitation and supplied-probability controls

| Field | Description |
|:--|:--|
| Experiment | **E10: forecast primitive bridge.** 90 fresh fixtures, 30 each at 2, 5, and 15 arms; 16,560 questions. |
| Question | Does forecast error depend on the interface, and does supplying the exact probability remove it? |
| Task | Cross counts, full summaries, summaries plus exact event probabilities, and exact probabilities alone. Compare Noul and binary Choice for identical next-reward questions; compare separate Noul best-arm events with one joint Choice distribution. Score on a common mean binary excess Brier scale. |
| Example prompts from the task | **Reward example:** “Is it true that the next pull of arm_00 yields reward 1? Report the probability that this event is true.” **Joint best arm:** “Which arm's fixed unknown success probability is largest? Report probabilities for the identities of the best arm.” **Exact-probability instruction:** “When an exact event probability is supplied, it is the known conditional probability of the named event and should be reported directly.” |
| Result | With full summaries, reward Choice risk exceeded Noul by **0.05075 [0.04595, 0.05584]**. Separate best-arm Noul risk was lower than joint Choice by **0.05721 [0.04369, 0.07213]**. Both use **97.5% intervals**. Exact probabilities did not eliminate Choice error. At 15 arms, full-summary best-arm Noul probabilities summed to **1.466 on average**. |
| Interpretation | The reward comparison isolates the request primitive while holding wording and evidence fixed. The best-arm comparison also changes joint versus marginal scope and requires one Noul question per arm instead of one Choice question. Better marginal forecasts do not guarantee coherence, cost-matched superiority, or better online reward. Missing Bayesian arithmetic cannot be the sole source of discrepancy when the probabilities are supplied. |

Sources: [protocol](docs/e10_primitive_protocol.md), [question builder](src/jevbandits/primitive_followup.py), [results](reports/primitive_v1/report.md), [primary contrasts](reports/primitive_v1/primary_effects.csv).

## Validation — Standalone versus batched questions

| Field | Description |
|:--|:--|
| Experiment | **Batching validation**, separate from E1–E10. Twenty reused E6 fixtures, ten per cohort; 1,920 questions. |
| Question | Could processing several questions in one request explain the large advice-format effects? |
| Task | Compare matched E6 questions sent alone or in mixed batches; evaluate choices and probability-vector changes. This reuses fixtures and is not fresh independent evidence about all states. |
| Example prompts from the task | Reuse E6 wording, including “Choose the recommended arm.” and “Read observation.recommended_arm. Choose the criterion whose ID exactly matches that value.” Each question's wording and evidence are matched across delivery modes. |
| Result | On targeted fixtures, nested and inline advice had **0% adherence in both modes**; advice-only had **100% adherence**, and exact values had **100% optimal agreement**, in both. Explicit lookup adherence was **78.75% alone versus 77.50% batched**. |
| Interpretation | The large format effect survives standalone delivery on this subset, making batching an unlikely explanation here. This does not establish universal invariance or independence. Fixture intervals do not capture all dependence induced by mixed requests. |

Sources: [protocol](docs/batch_control_protocol.md), [implementation](src/jevbandits/batch_control.py), [results](reports/batch_control_v1/report.md).

Together these studies account for **598,080 evaluation/control questions**. E3–E5, E8, and E9 use **1,560 independent online environments**; policies, repeated calls, and turns do not multiply that sample size. The [collection audit](docs/data_completeness_audit.md) records completeness and cost. Additional offline theory, behavior, forecast-repeat, and numerical audits refine the interpretation; they are not extra Jev collection experiments.
