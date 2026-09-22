# Selective literature review and experimental design implications

Reviewed 2026-09-20 using the user's cleaned local OCR manuscripts. This is a design review, not an experiment result. The execution manifest and revised first-experiment plan are authoritative for the final sample sizes and prompts. The five-experiment allocation below reflects the subsequently selected design.

## What the selected papers establish

### Krishnamurthy et al.: separate useful exploration from randomness

Read Sections 2, 3.1, 3.4, 4.1, and Appendix A of *Can Large Language Models Explore In-Context?*, arXiv v3, accepted at NeurIPS 2024. The paper factorizes scenario, exploration framing, raw versus summarized history, direct versus distributional output, and reasoning requests. Its successful GPT-4 configuration combines several interventions; the headline comparison does not isolate the causal contribution of summarization alone. Its main smaller GPT-4 runs use ten replicates, while the successful longer configuration receives forty. [Primary paper](https://arxiv.org/abs/2403.15371v3)

Two useful diagnostics distinguish abandoning the best arm for the remainder of an episode from continuing almost uniform allocation. Distributional output can reduce the first without solving the second. The paper also shows that greedy-looking behavior on externally supplied histories depends strongly on which policy generated those histories. Fixed-state probes therefore complement, rather than substitute for, closed-loop evaluation. Its UCB comparator uses the heuristic bonus $\sqrt{1/n_i}$, not standard UCB1; our UCB1 results should not be described as an exact replication. [Sections 3.4 and 4.1](https://arxiv.org/html/2403.15371v3)

The appendix's displayed successful prompt describes the agent as a bandit algorithm and repeats a reasoning instruction. It is not identical to the explicit exploration-benefit sentence proposed for Jev. We should preserve our own exact text and avoid calling it the paper's prompt. Jev does not expose a generative reasoning channel, so this study tests interventions available through its own interface.

**Design implications:** report distributions and late allocation alongside regret; include both useful-exploration and unnecessary-exploration states; separate framing, summary access, and host sampling; and avoid attributing a mechanism from only one offline history distribution.

### Sun et al.: model prediction and exploration control are different roles

Read Sections 2–3, 5.1, 5.3, and Appendix A of *Large Language Model-Enhanced Multi-Armed Bandits*, arXiv v1. TS-LLM predicts a stochastic reward separately for each arm, selects the largest prediction, and uses a decaying generation temperature. RO-LLM predicts losses and applies a separate SquareCB-style sampling rule. The synthetic experiments use numeric arm features and noisy function values, so the task is distinct from independent anonymous Bernoulli arms. [Primary paper](https://arxiv.org/abs/2502.01118v1)

The synthetic stochastic experiments use sixteen arms, four-dimensional features, one hundred decisions, and ten repetitions. The temperature schedule is chosen through an empirical ablation; its tuning budget and transfer should be separated from untouched evaluation when designing a new study. The displayed main comparison is principally against direct LLM selection and random search, not evidence that the proposed hybrids beat exact Bayesian planning. [Algorithms and experimental appendix](https://arxiv.org/html/2502.01118v1)

**Design implications:** sampling Jev's answer probabilities is not TS-LLM, and neither is automatically posterior sampling. An external Bayesian recommendation supplied to Jev tests whether it can use advice; it does not show that Jev independently inferred or computed the recommendation. Conversely, successful advice use is a practically meaningful outcome worth measuring, not an invalid result merely because external computation helped.

## Recommended five complementary experiments

The following decomposition retains the original plan's core while separating the scientific questions. The final manifest may redistribute replication after measuring API throughput, token use, and pilot variability.

| Experiment | Question and intervention | Main comparison and unit |
| --- | --- | --- |
| 1. Exact horizon and framing | On paired two-arm fixtures, vary remaining horizon and neutral, exploration-benefit, or regret wording. Include the one-to-two-pull crossover, equal means with unequal precision, and a clear incumbent. | Exact Bellman action-value loss and tie-aware agreement; posterior fixture is the independent unit, repeated API calls measure service variation. |
| 2. Bayesian information ladder | On a preregistered random posterior-state bank, show counts; counts plus posterior means; counts plus means and uncertainty; exact action values; or an exact action recommendation. Cross short horizons and label reversals. | Within-state changes in exact decision loss; distinguish arithmetic/representation assistance from policy advice. |
| 3. Exact closed-loop control | Run two-arm, short-horizon prior-drawn episodes using direct and sampled Jev with counts and Bayesian summaries, plus a direct policy supplied exact action values. | Exact DP, TS, Bayes-UCB, greedy; independent paired episodes. Measure regret and local Bellman loss. |
| 4. Online scaling | Use $K\in\{2,3,5,10,15\}$, fixed horizon, and clear-winner, close-winner, and prior-drawn families. Compare direct and sampled Jev with the two evidence formats. | TS and Bayes-UCB primary references; paired episode pseudo-regret. Preserve every cell rather than relying on a pooled score. |
| 5. Frozen intervention validation | On fresh seeds, compare Bayesian-summary direct choice with neutral wording, regret wording, a Thompson-strategy instruction, and supplied Bayes-UCB assistance. Include Bayesian-only execution. | Paired changes in regret, advice adherence, and deviations that help or hurt. Freeze any selection rule and prompt after development, before opening these results. |

Recommendation assistance in Experiment 2 can use exact DP because horizons are short. Experiment 5 must identify advice as **Bayes-UCB advice**, not optimal advice: the large-arm finite-horizon optimum is not available. Supply the recommendation calculated from the recipient policy's own history. A separately running Bayes-UCB trajectory is not the same-history counterfactual.

For a controlled online intervention experiment, select the framing in advance or use only development diagnostics to select it. If selected after inspecting online evaluation outcomes, the validation requires wholly new tasks, and that selection must be disclosed. Do not repeatedly add prompts to the held-out experiment until an improvement appears.

The selected allocation is 4,800 decisions for 100 horizon/framing fixtures, 6,000 for 100 disjoint assistance fixtures, 20,000 for the exact episode panel, 240,000 for the core matrix, and 120,000 for fresh-seed strategy validation: **390,800 Jev decisions**, plus the development pilot. These are decisions, not necessarily HTTP requests if the API safely supports independent-question batching. At an illustrative 2,000 billed tokens per decision and $0.042 per million input tokens, this would be approximately **$32.83** before retries; shorter measured prompts can reduce this considerably. Actual API accounting must replace this illustrative estimate. Batching must preserve complete per-question context and independence; do not share another episode's observations as evidence. More replication should target uncertain scientifically relevant contrasts rather than indiscriminately expanding every cell.

## Inference and controls needed for a credible paper

For independent Beta posteriors with mean $m_i$, the exact finite-horizon comparison is

$$
Q_h(s,i)=m_i+m_iV_{h-1}(s^{i,+})+(1-m_i)V_{h-1}(s^{i,-}),
\qquad V_h(s)=\max_iQ_h(s,i),\qquad V_0(s)=0.
$$

Report $V_h(s)-Q_h(s,A)$ as a local decision loss, not regret against a clairvoyant arm. Tied optimal choices incur zero loss. To score a sampled action distribution offline without adding Monte Carlo noise, report $V_h(s)-\sum_i p_iQ_h(s,i)$ using the same declared normalization as the online sampler. This is an expected one-decision loss, not simulated closed-loop policy performance.

Counts under a known prior already determine the full posterior. The information ladder changes numerical accessibility, not observed evidence. Keep counts in every format, so the means-only condition does not accidentally remove sample size. A useful description of an uncertainty benefit is improved control after exposing a redundant but easier-to-use representation.

The primary episode endpoint remains pseudo-regret,

$$
\bar R_T=\sum_{t=1}^T(\theta^*-\theta_{A_t}).
$$

Expected reward maximization and expected regret minimization have the same optimizer here. Reward-threshold probability or best-arm identification is a different objective and should not be mixed into a wording experiment.

Use common arm-indexed potential reward streams, separate action RNGs, and paired episode differences. Bootstrap independent episodes for online comparisons and whole posterior states for offline comparisons; keep their repeated horizons, labelings, and requests together. A thousand decisions from ten trajectories are ten independent trajectories. Report failed/incomplete episodes by policy and reason; do not silently replace malformed output with a classical policy.

Prespecify a small primary contrast family: summary benefit, sampling benefit, explicit-framing benefit, and recommendation benefit. Report effect sizes and intervals throughout; if hypothesis tests support headline claims, apply Holm correction within that family. Interactions by arm count or task difficulty are exploratory unless prespecified with adequate replication. Nonsignificance does not establish equivalence.

Late abandonment and minimum allocation are descriptive mechanisms, not universal failure criteria. At a short horizon or tiny gap, uncertain arms can rationally remain unresolved. Similarly, a Bayesian learner can rationally stop sampling an arm that is actually best but looks poor under its observations. Judge those decisions against exact values where possible, and compare aggregate consequences elsewhere.

## Promising scientific narrative and limits

The useful question is where performance changes across **evidence representation, horizon, instruction, and allocation of computation between model and algorithm**. A mixed result can be informative: for example, summaries could improve final-pull arithmetic while leaving exploration-value judgments unchanged, or external advice could improve reward while sampling merely reduces abandonment at the expense of continued unnecessary exploration.

The present anonymous-arm setting deliberately excludes Jev's semantic knowledge advantage. It is a controlled study of numerical evidence use and sequential control, not an overall product evaluation. Positive or negative results should name the pinned model, prompt, horizon, and environment family. Future semantic-prior and reasoning-model harness studies need their own controls and held-out task families; they are not necessary to finish this first five-experiment series.

## September 22 update: allocating computation under response deadlines

This section extends the historical review above. No new experiment results are reported. Active scope, budgets, and scheduling are consolidated in the private [research plan](../../aistats2027-manuscript/research_plan.md), with a separate [timing design](../../aistats2027-manuscript/design/anytime_decisions.md) and [document map](../../aistats2027-manuscript/TABLE_OF_CONTENTS.md).

### The relevant gap is narrower than combining tools

The interesting question is when a decision system should spend additional computation on beliefs or actions, given a deadline and a usable current policy. Bayesian metareasoning, contract algorithms, learned model routing, program execution, and LLM-assisted Bayesian optimization already address substantial parts of this question. Adding Jev to a reasoning model and a Bayesian library does not by itself establish a new method. A defensible contribution would measure or improve the conversion of heterogeneous computation into sequential decision utility, with explicit timing semantics, matched information, strong classical controls, and protected evaluation.

The prior Bernoulli experiments are useful controls because exact updates and many strong policies are inexpensive. Jev's API latency is not evidence of an advantage over local Bayesian computation. A hybrid should earn its complexity through measured benefit, including routing and tool overhead. A negative result in this setting can establish a boundary; it does not settle semantic or expensive-inference tasks.

### What the selected rereadings change

**Callaway et al., Learning to Select Computations (2018):** reread the metalevel MDP and value-of-computation construction (Sections 2–3), plus the experimental discussion of metareasoning overhead (Section 5). Computation has a cost, and stopping executes an object-level policy. Their selection setting rewards the final selected action; our bandit trajectories also earn rewards and acquire observations along the way. Transferring a stopping rule requires matching that objective. BMPS offers an approximation using different value-of-information features, not a universal optimal scheduler. Its offline tuning and online routing costs must be accounted for separately. See the [local paper](../../aistats-2027-literature/papers/callaway_2018_learning_select_computations/).

**Hay et al., Selecting Computations (2012):** reread the computation-selection setup, non-indexability counterexample, and blinkered approximation (Sections 2–4). Which computation is worthwhile can depend on competing alternatives, so independent per-tool indices are not generally optimal. Blinkered decomposition requires independence assumptions that a shared LLM call informing several actions may violate. Include a simple myopic computation selector as a baseline and name its limitations; do not call it Bayesian-optimal merely because its features are Bayesian. See the [literature collection](../../aistats-2027-literature/searches/2026-09-21/README.md).

**LLAMBO:** reread the candidate-generation/evaluation discussion and Appendix E.5 timing account. Its useful lesson is to separate candidate generation, surrogate estimation, and point selection. Its reported optimization overhead excludes objective-evaluation time; this distinction matters when expensive evaluations make substantial controller overhead acceptable. That is not evidence of meeting hard per-request deadlines. Benchmark end-to-end execution under our own objective and clock.

**Nafar et al., Reasoning with Uncertain Text (2025):** read the problem definition, Bayesian-network construction, symbolic mappings, and evaluation discussion from raw OCR. BLInD maps textual probability descriptions to queries with exact reference answers, using a restricted graph family. The paper compares direct answers with Python, Monte Carlo, and ProbLog mappings. It is close prior art for delegating probability calculation while retaining language-based extraction and orchestration. The authors state that underperforming combinations are omitted from one results table; our evaluation should retain every frozen condition. This is a probability-query benchmark, not sequential regret or deadline-aware control. See the [local paper](../../aistats-2027-literature/papers/nafar_2025_uncertain_text_llms/).

**PAL (Gao et al., 2023):** reread the analysis comparing actual interpreter execution with Python-shaped prompts lacking execution. This motivates an execution control: supplying structured Bayesian output and asking a model to use it is distinct from giving it an executable tool. Valid syntax alone does not establish correct inference or good control. Record extraction, tool execution, answer integration, and action selection errors separately. See the [local paper](../../aistats-2027-literature/papers/gao_2023_pal/).

**Zhu et al., Learning to Act While Waiting (2026):** read the asynchronous architecture, intermediate-state augmentation, informal delayed-oracle result, and limitations from raw OCR. This gives a close fast/slow control analogy. Its architecture requires access to specific intermediate components of the pretrained policy; a remote Jev/reasoning API may expose no equivalent. Its guarantee is conditional on the stated delayed-state environment assumptions and does not certify our proposed bandit wrapper. The transferable lesson is to record which observation version each computation used and account for committed actions during inference. See the [local paper](../../aistats-2027-literature/papers/zhu_2026_learning_act_waiting/).

These are selected readings, not a claim that every paper or proof was checked. The contract-scheduling full reviews and the researcher-supplied Hansen–Zilberstein monitoring paper are being integrated through the [September 22 collection](../../aistats-2027-literature/searches/2026-09-22/README.md); raw OCR must not be treated as verified mathematics.

### Arrival uncertainty and response budgets are different variables

A homogeneous Poisson process has exponentially distributed inter-arrival gaps. Drawing integer Poisson gap lengths instead defines a different renewal process. Both can be studied, but the exact law should be specified rather than described ambiguously as “Poisson intervals.” Reveal a request's allotted budget at its arrival, and keep future requests hidden.

A known current deadline permits a contract computation; an unknown next interruption motivates an interruptible schedule that keeps a completed result available. A streaming API is not automatically an anytime algorithm. A valid fallback, result validation, stale-answer handling, and out-of-order deadlines are part of the policy definition.

The probability of no arrival during a computation is not the probability of meeting a separate request deadline. Queueing, worker limits, overlapping requests, and cancellation charges must be explicit. Reward should follow the action actually executed at the defined response event. A late answer cannot replace it in retrospective scoring. The initial design fixes feedback release at each request's deadline to avoid silently giving fast policies extra early observations; early-return feedback can be a separately declared treatment.

### Baseline ladder and study sequence

Compare classical-only policies and a development-selected fixed policy before adding a reasoning model. For the reasoning-model tool-access factorial, add a cheap router and a sequential cascade. Keep coding permissions identical: unrestricted Python allows a nominal “no Bayesian tool” model to implement Bayesian calculations itself.

Start with event simulation, measured development latencies, and numerical bandits. Freeze the deadline grid and primary contrast family before comparative collection. Add costly posterior inference or contextual decisions only when a real computation tradeoff exists. Report trace-level uncertainty, reward/regret, deadline misses, fallback rates, information staleness, and full online/offline costs. Repeated polls within one trace do not create independent replicates. Abstract responsiveness experiments can motivate robotic systems without claiming robot safety or deployment performance.
