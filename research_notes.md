# Research notes: Jev, Bayesian decisions, and sequential experiments

Investigated 2026-09-20. These notes distinguish official product documentation, observed API behavior, published research, and proposed experiments. The actionable first study is in [first_experiment_Plan.md](first_experiment_Plan.md).

## 1. What Jev offers

Jev is TypeSafe AI's hosted System One decision model. It evaluates typed questions against supplied state and returns structured answers rather than generated explanations. Its three primitives are Choice, Score, and Noul. Multiple questions share state but are evaluated independently; questions in one request cannot consume one another's answers. Persistent bandit memory must therefore be maintained by our program and supplied on subsequent calls. [Introduction](https://docs.typesafe.ai/introduction), [quick start](https://docs.typesafe.ai/introduction/quickstart)

| Primitive | Documented meaning | Relevance here |
|---|---|---|
| Choice | A listed option, probabilities over options, and confidence | Select an arm or a pre-generated candidate point |
| Score | A probability-weighted position across an ordered rubric | Coarse semantic assessments, not arbitrary precise numeric generation |
| Noul | A value in $[0,1]$ for a yes/no proposition | Separate reward-event or comparison-probability diagnostics |

Choice accepts up to 255 options. Score accepts up to ten levels and computes its score from their probability-weighted indices. These constraints make 2–15-arm selection straightforward, but suggest a candidate-selection interface for future continuous optimization. [Choice](https://docs.typesafe.ai/primitives/choice), [Score](https://docs.typesafe.ai/primitives/score)

The current documented version is `jev-1.13.0`; aliases include `jev-latest` and `jev-preview`. Pin and log the version for reproducibility. Text/JSON input is supported. Documented limits are 64k total input tokens and 32k for state plus the longest question; advertised limits are 1,200 requests/minute and 250,000 tokens/second, subject to change. Direct pricing is $0.042 per million input tokens and free output. [Models](https://docs.typesafe.ai/models)

The official launch announcement dates to September 15, 2026 and describes Reinforcement Learning for Calibrated Decisions. Its workflow evaluations compare against reference-model judgments; they are not direct evidence of optimal Bayesian sequential behavior. Type guarantees also do not establish factual correctness or good decisions. [Launch and evaluation discussion](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

### Important limitations for this research

TypeSafe explicitly identifies numerical precision, counting, indirection, and overly large irrelevant state as weaknesses. It recommends putting arithmetic in code and warns against interpreting Score interpolation as precise numerical reconstruction. This motivates separate tests of statistical computation and decision-making given computed statistics. Anonymous numerical bandits deliberately test a difficult capability boundary. A failure would not establish poor performance on semantic decision tasks. [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)

`confidence` is computed from the returned distribution, not an additional independent source of evidence. The reviewed confidence documentation does not supply a formula that should be reverse-engineered from a few examples. In particular, it is not a Bayesian credible interval for an arm's success rate. Preserve the distribution and evaluate task-specific calibration separately. [Confidence](https://docs.typesafe.ai/confidence)

The direct endpoint is `POST https://api.typesafe.ai/v1/systemone`, with bearer authentication and fields `state`, `model`, and `questions`. `GET /v1/models` lists available names. A question's ID is routing metadata, not an instruction supplied to the model: put the objective in `instructions`, not merely in a key called `maximize_reward`. The reviewed API has no documented temperature or seed control. [API reference](https://docs.typesafe.ai/api)

Search results contained several independently operated Jev-branded sites. Product/API claims here use official `typesafe.ai` documentation; the key was sent only to `api.typesafe.ai`.

## 2. Actual API checks and environment

Used Python 3.12.9 and HTTPX, with `dotenv_values` reading the repo's lowercase `jev_key`. Neither the key nor authorization headers were printed or saved. No conversational/generative API was required.

The models endpoint returned `jev-latest` and `jev-preview`. All four inference responses reported `jev-1.13.0`.

| Check | Input tokens | End-to-end seconds | Observed answer |
|---|---:|---:|---|
| Two arms, one pull remaining | 476 | 0.156 | A: 0.96; B: 0.04; choose A |
| Same evidence, 100 pulls remaining | 478 | 0.161 | A: 0.83; B: 0.17; choose A |
| Identical repeat of previous request | 478 | 0.124 | A: 0.82; B: 0.18; choose A |
| Fifteen arms, computed posterior summaries, 100 pulls | 2,248 | 0.121 | Choose arm_09; reported mass summed to 0.99 |

In the two-arm checks, A had six successes and four failures, B was unobserved, and both started from Beta(1,1). Thus the posterior means were $7/12$ and $1/2$. Both horizons used the same instruction to maximize expected total reward and consider learning's future value. The shift toward B shows response sensitivity to horizon in this example, but the selected action did not change. No claim of optimal exploration follows from these four calls.

Observed probabilities are not bit-stable across repeated calls. The 15-arm response looked quantized to two decimal places (with ordinary floating-point artifacts). An initially strict sum-to-one assertion failed after all four HTTP requests had succeeded. Offline revalidation accepted the small discrepancy and confirmed valid IDs/ranges/argmax behavior. This was a validation finding, not authentication failure. The future sampled policy must explicitly normalize acceptable discrepancies while retaining raw probabilities.

Total: 3,680 reported input tokens, or **$0.00015456** at the published rate. This is an estimate from usage and pricing, not a verified account-balance deduction. Latencies are four local measurements, not a benchmark distribution or SLA. The request/response log and preparation script are in ignored `artifacts/`.

The installed research environment has NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.6, Matplotlib 3.11.2, HTTPX 0.28.1, python-dotenv 1.2.3, Pydantic 2.13.5, pytest 9.1.1, and Ruff 0.16.8. `uv.lock` is the reproducibility record. Scientific imports and an analytic Beta mean check passed. No experiment implementation has been started.

## 3. The distinctions the study must preserve

### Action probability, reward probability, and probability of being best

These three questions are different:

1. “Which arm should I choose to maximize remaining total reward?” returns Jev's distribution over answers to a decision problem.
2. “Will the next pull of arm $i$ succeed?” concerns a posterior predictive reward probability.
3. “Which arm has the highest unknown success probability?” concerns posterior probability of being best.

An action distribution from the first question cannot automatically be used as the answer to either of the others. Likewise, distribution concentration is not proof of correct Bayesian inference.

Choosing the modal answer to a long-term decision question can implement directed exploration. Conversely, sampling any distribution can create random action diversity without valuing information. The plan makes direct choice primary and treats categorical sampling as a separate policy.

### Expected reward versus probability of success

For fixed horizon and fixed unknown means, maximizing expected total reward is equivalent to minimizing expected regret against the best fixed arm. Maximizing the probability of exceeding a target reward, winning a tournament, or identifying the best arm is not generally equivalent. The first experiment specifies expected cumulative reward; it should not alternate casually between these objectives.

### Inference versus control

For independent stationary Bernoulli arms, counts and a specified prior are sufficient to calculate the posterior. Giving Jev posterior summaries does not give it extra observations. Improvement would suggest that numerical computation or representation was a bottleneck. Failure despite correct summaries would shift attention toward comparison, objective understanding, or sequential planning.

Supplying TS samples, UCB indices, or exact optimal-action values is a stronger intervention: the classical method has already made much of the decision. Such conditions can test whether Jev follows an externally computed recommendation, but should not be described as evidence that Jev independently discovered Bayesian exploration.

## 4. Mathematical reference

### Beta-Bernoulli learning

Assume independent unknown arm means:

$$
\theta_i\sim\operatorname{Beta}(\alpha_{i,0},\beta_{i,0}),
\qquad Y_t\mid A_t=i,\theta_i\sim\operatorname{Bernoulli}(\theta_i).
$$

After observing $s_i$ successes and $f_i$ failures:

$$
\alpha_i=\alpha_{i,0}+s_i,\qquad
\beta_i=\beta_{i,0}+f_i,\qquad
\theta_i\mid D_t\sim\operatorname{Beta}(\alpha_i,\beta_i).
$$

$$
m_i=\mathbb E[\theta_i\mid D_t]=\frac{\alpha_i}{\alpha_i+\beta_i},
\qquad
v_i=\operatorname{Var}(\theta_i\mid D_t)
=\frac{\alpha_i\beta_i}{(\alpha_i+\beta_i)^2(\alpha_i+\beta_i+1)}.
$$

The predictive probability of success on the next pull is $m_i$. The predictive variance of that binary outcome is $m_i(1-m_i)$, which is not $v_i$. Epistemic uncertainty about an arm can shrink while its outcomes remain noisy.

A 95% equal-tailed credible interval is

$$
[F_i^{-1}(0.025),F_i^{-1}(0.975)],
$$

where $F_i$ is the posterior Beta CDF. With independent continuous posteriors, probability of being best is

$$
w_i=\Pr(\theta_i>\max_{j\ne i}\theta_j\mid D_t)
=\int_0^1 f_i(x)\prod_{j\ne i}F_j(x)\,dx.
$$

Numerical quadrature or seeded Monte Carlo can supply a reference for a later inference-only calibration test. This probability is not identical to the arm's posterior mean or its optimal finite-horizon action probability.

### Objectives and regret

For a policy $\pi$ and fixed means $\theta$:

$$
J_T(\pi;\theta)=\mathbb E_\pi\left[\sum_{t=1}^T Y_t\mid\theta\right],
\qquad
R_T(\pi;\theta)=T\theta^*-J_T(\pi;\theta).
$$

The realized action sequence has nonnegative pseudo-regret

$$
\bar R_T=\sum_{t=1}^T(\theta^*-\theta_{A_t}).
$$

Its expectation equals expected regret. In contrast, $T\theta^*-\sum_tY_t$ includes outcome noise and may be negative. Prefer pseudo-regret for comparing action quality while also reporting actual rewards.

Bayesian regret averages expected regret over a specified prior:

$$
\operatorname{BR}_T(\pi)=\mathbb E_{\theta\sim p_0}[R_T(\pi;\theta)].
$$

Do not label a uniform average over hand-picked fixed-mean examples “Bayesian regret” unless that task distribution has explicitly been declared the prior.

For a best-arm identification task, a final recommendation $\hat a_T$ instead incurs simple regret

$$
r_T=\theta^*-\theta_{\hat a_T}.
$$

An algorithm can collect less reward during learning yet give a better final recommendation. That is a different experiment.

### Exact finite-horizon Bayesian control

Let $s$ contain all Beta posterior parameters and let $h$ count remaining pulls including the current one. Write $s^{i,+}$ for the state after success on arm $i$, and $s^{i,-}$ after failure.

$$
V_0(s)=0,
$$

$$
Q_h(s,i)=m_i+m_iV_{h-1}(s^{i,+})+(1-m_i)V_{h-1}(s^{i,-}),
\qquad
V_h(s)=\max_i Q_h(s,i).
$$

This Bellman recursion is the exact Bayesian objective under the model assumptions. With one pull left, choose the largest posterior mean. With longer horizons the second and third terms can favor an uncertain arm. The state space grows rapidly, motivating a small two-arm exact panel rather than pretending a 15-arm solver is cheap. A primary applied treatment explicitly solves the finite-horizon two-arm problem with dynamic programming. [Villar et al., Bayesian adaptive design](https://pmc.ncbi.nlm.nih.gov/articles/PMC5473477/)

At a visited state, $V_h(s)-Q_h(s,A_t)$ measures the loss from the chosen action followed by optimal continuation. It isolates the current decision; it is not the same as the complete policy's loss or regret against known true means.

### Thompson sampling and Bayes-UCB

Thompson sampling draws independently from the arm posteriors and maximizes the draw:

$$
\tilde\theta_i\sim\operatorname{Beta}(\alpha_i,\beta_i),
\qquad A_t=\arg\max_i\tilde\theta_i.
$$

Consequently it selects arm $i$ with probability $w_i$. It is computationally simple and has strong regret results, but is generally not exact finite-horizon Bayesian planning. It can still explore on the final pull. [Agrawal and Goyal](https://arxiv.org/abs/1111.1797), [Russo et al. tutorial](https://arxiv.org/abs/1707.02038)

Bayes-UCB chooses an optimistic posterior quantile. It gives a complementary Bayesian baseline: optimism instead of probability matching. The first experiment fixes its schedule explicitly; schedule variants must be named and not tuned on evaluation results. [Kaufmann, Cappe, and Garivier](https://proceedings.mlr.press/v22/kaufmann12.html)

## 5. Which additional Bayesian methods are most interesting?

| Method | Question it helps answer | Priority and caveat |
|---|---|---|
| Exact finite-horizon DP | Does Jev recognize genuinely valuable exploration? | First study, restricted to small two-arm settings |
| Thompson sampling | Can Jev match a simple strong Bayesian learner? | First study, all arm counts |
| Bayes-UCB | Does uncertainty-seeking resemble optimism or probability matching? | First study, all arm counts |
| Posterior-mean greedy | Is apparent sophistication just mean maximization? | First study control |
| Information-directed sampling (IDS) | Does Jev seek information that matters for decisions? | Follow-up, especially informative/correlated actions |
| Knowledge gradient (KG) | Can it value an observation by improvement in the best posterior decision? | Follow-up, especially best-arm identification and BO |
| Gittins / finite-horizon index approximations | Can a cheap horizon-aware index approximate exact planning? | Follow-up with objective matched carefully |
| Hierarchical/contextual Thompson sampling | Can semantic priors help across related arms/tasks? | Follow-up where Jev has actual domain information |

IDS explicitly balances expected one-step regret with information about the optimal action. For an action distribution $p$:

$$
\Psi(p)=\frac{\left(\sum_a p_a\Delta_a\right)^2}{\sum_a p_ag_a},
\quad
\Delta_a=\mathbb E[\theta_{A^*}-\theta_a\mid D_t],
\quad
g_a=I(A^*;Y_t\mid A_t=a,D_t).
$$

It is appealing for testing purposeful information acquisition, particularly when one observation informs several choices. In independent-arm Bernoulli tests, the added implementation burden may buy less diagnostic value than the exact small panel. [Russo and Van Roy, IDS](https://arxiv.org/abs/1403.5556)

A basic knowledge-gradient quantity is

$$
\operatorname{KG}(a)=\mathbb E[\max_i m_i^{\mathrm{after\ observation\ at}\ a}\mid D_t]-\max_i m_i.
$$

Its conversion into an online reward policy needs a horizon/objective convention. A heuristic such as $m_a+(h-1)\operatorname{KG}(a)$ should be labeled a one-step approximation, not the exact Bellman solution. [Frazier, Powell, and Dayanik](https://people.orie.cornell.edu/pfrazier/pub/CorrelatedKG.pdf)

The classic Gittins optimality result concerns an infinite discounted setting under particular independence assumptions. Do not present a discounted index as the exact solution to our undiscounted finite-horizon task. Finite-horizon index policies have separate analyses, including Gaussian settings. [Lattimore, finite-horizon Gittins](https://proceedings.mlr.press/v49/lattimore16.html)

## 6. Prior art and its implications

### Directly related LLM bandit studies

- **Krishnamurthy et al. (2024), Can large language models explore in-context?** Tests untrained-for-the-task LLM agents in bandit environments. Its comparisons of interaction histories, summarized evidence, and decision representations are directly relevant. Our study should separate sufficient-statistic inputs from memory/counting burdens and measure late-episode failure to recover from early mistakes. [Paper](https://arxiv.org/abs/2403.15371)
- **Sun et al. (2025 preprint; ACL 2026), Large Language Model-Enhanced Multi-Armed Bandits.** Places LLM reward prediction inside classical exploration frameworks, including TS-style sampling and a regression-oracle approach. This is motivation to test hybrid inference/control systems, not evidence that model output variation automatically constitutes a correct Bayesian posterior. It considers both semantic and synthetic arm settings. [Preprint](https://arxiv.org/abs/2502.01118), [ACL paper](https://aclanthology.org/2026.acl-long.368.pdf)
- **TextBandit (2025).** Uses linguistic feedback and compares language models with standard bandit algorithms. It motivates a later representation study, but its task differs from our deliberately numerical sufficient-statistic test. [ACL Anthology](https://aclanthology.org/2025.ethicalllms-1.1/)
- **Sasso et al. (2025), Exploration with Foundation Models.** Studies exploration across bandits and other environments and discusses hybrid guidance. Useful context for separating semantic competence from reliable action selection. [Paper](https://arxiv.org/abs/2509.19924)

### Jev-specific prior art found

The independent **jev-news-cold-start** repository uses Jev's semantic news features to inform a Thompson-sampling prior. Its reported click improvements come from a semi-synthetic evaluation using logged article CTRs, with calibration on training data. This is evidence of an existing Jev-plus-Bayesian approach to investigate, not a replication or an established result of the present project. It differs from asking Jev itself to choose each anonymous arm. [Repository and caveats](https://github.com/zhuyansen/jev-news-cold-start)

TypeSafe's **Autoresearch feature discovery** cookbook uses a generative model to propose semantic questions, Jev to answer them, and a downstream supervised model to evaluate the resulting features. This is close to the user's longer-term orchestration idea, but it is feature/harness optimization rather than a demonstration that Jev changes its own weights or recursively improves general intelligence. [Official cookbook](https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery)

The searches found these useful precedents, but did not establish a direct published Jev benchmark matching our proposed finite-horizon anonymous Bernoulli tasks. That is a search finding, not a claim of exhaustive novelty.

### Local papers and access status

Two primary papers were downloaded under ignored `pdfs/`: Krishnamurthy et al. and Sun et al. The delegated paper review used the [Mistral OCR skill](/Users/tom/.agents/skills/mistralocr/SKILL.md) for the first. The completed `--full` workflow processed 33 pages, annotated all 19 figures, reviewed all 23 caption crops and 53 inline-math candidates, and corrected OCR errors including lost marks that distinguish prompt variants. This involved one paid Mistral OCR operation; subsequent reviews used the cached response and local images, with no additional OCR calls. Its cost is separate from the Jev token estimate and was not verified against account billing. The second paper was read through accessible primary HTML as well as retained as a PDF.

Detailed local notes are in [literature paper notes](../aistats-2027-literature/notes/paper_notes.md). The original Krishnamurthy source has several caption/plot inconsistencies, retained and flagged in those notes. The OCR markdown is a reading aid, not a replacement for original figures/data when reproducing results.

No identified priority paper currently requires library access. This does not mean every bibliography entry was downloaded or fully reviewed. Most additional sources above were inspected through primary abstracts, documentation, or accessible full text as relevant. If later reading encounters a paywall, add the exact title/DOI and missing material here rather than relying on a secondary summary.

## 7. Follow-up experiments worth prioritizing

### A. Can Jev use the supplied statistics correctly?

Expand the assisted-input ladder: counts; counts plus means; means plus credible intervals; full posterior parameters; and finally posterior probability of being best. Vary sample size while holding means approximately fixed. Add deliberately stale or contradictory summaries as a separate robustness study. Do not mix those with the primary clean-data comparison.

Ask inference-only questions separately from action choice. Compare predicted reward-event probabilities to analytic posterior predictive means using Brier score/log loss, and best-arm distributions to $w_i$ with proper distributional scores. For empirical calibration, use independently generated or held-out tasks with a defined target event. Log loss needs a declared numerical clipping rule; do not tune it to flattering results.

This can disentangle “does not calculate Bayes,” “does not understand uncertainty,” and “understands the posterior but plans poorly.”

### B. Different sequential objectives and environments

After stationary cumulative reward: best-arm identification, unequal evaluation costs, rare rewards, drifting means, delayed outcomes, and correlated/contextual arms. Each needs matching baselines and a separately stated objective. For example, nonstationary tasks need discounted/sliding-window or change-point-aware learners; ordinary stationary TS is insufficient as the only comparator.

Semantic arms are especially valuable for distinguishing domain knowledge from exploration control. Use descriptive arm features related to rewards, shuffled descriptions as a negative control, and a classical posterior learner that receives the same covariates or learned prior. Hold out task families, not just random rows. A Jev-informed prior should expose and vary its effective prior sample size so a confident wrong prior cannot silently dominate evidence.

### C. Numeric optimization against a GP surrogate

For a black-box function on a bounded domain, minimize $f(x)$ given a limited number of evaluations:

$$
y_t=f(x_t)+\epsilon_t,
\qquad
f\mid D_t\sim\operatorname{GP}(m_t,k_t).
$$

The natural main metric is terminal recommendation/simple regret, or best evaluated true value for noiseless functions, rather than cumulative bandit reward. In noisy tests, do not treat the smallest noisy observation as the true optimum; evaluate final recommendations independently or against simulator truth.

Proposed progression: dimensions 2, 5, 10, then 20; fixed evaluation budgets; shared initial designs; smooth synthetic functions with known minima; and later noisy or mixed-variable tasks. Compare random/Sobol designs, GP expected improvement (using a numerically stable log-EI implementation), GP confidence-bound selection, posterior sampling, and TuRBO when dimension increases. GP-UCB connects confidence-bound exploration to regret; TuRBO addresses weaknesses of a single global surrogate in more difficult optimization. [GP-UCB](https://arxiv.org/abs/0912.3995), [TuRBO](https://arxiv.org/abs/1910.01739), [BoTorch acquisition documentation](https://botorch.readthedocs.io/en/stable/acquisition.html)

Because Jev selects predefined answers, use a controlled candidate pool, initially 128 points per step, rather than asking Score to manufacture precise coordinates. Compare:

1. Jev choosing from candidates using observation history and coordinates.
2. Jev choosing from those same candidates with GP predictive means/uncertainty.
3. Classical acquisitions choosing from the same candidate pool.
4. A separately labeled unconstrained continuous BO reference.

Candidate generation is part of the algorithm and can dominate the result. Use the same generator and initial design across candidate-restricted policies, refresh candidates with the same reproducible rule, and keep candidate IDs neutral. A GP-assisted Jev result tests acquisition choice given a surrogate, not independent GP inference. Delay installing PyTorch/GPyTorch/BoTorch until this study is specified.

**LLAMBO** is relevant prior art for LLM participation in Bayesian optimization, especially semantic priors, initialization, and surrogate/acquisition components. Its interfaces are not directly interchangeable with Jev's non-generative outputs. [LLAMBO](https://arxiv.org/abs/2402.03921)

### D. Reasoning model + Jev + Bayesian model

A useful later architecture is: a reasoning model proposes hypotheses, features, or experimental strategies; Jev evaluates bounded semantic questions/candidates; a Bayesian learner maintains uncertainty and allocates evaluations; a held-out evaluator decides whether the whole system improved.

Measure improvement in sample efficiency, final performance, compute, and total dollars. Include Bayesian-only, Jev-only, reasoning-only, fixed-hybrid, and adaptive-hybrid controls with matched resource budgets. Separate the task-level learning loop from the slower loop that changes prompts or policies. Freeze a final task-family test set that neither loop can inspect.

Call this automated research or policy/harness improvement until evidence supports a stronger claim. Repeatedly tuning prompts on the same benchmark can improve its score without improving general decision-making. The first bandit study establishes a much cleaner baseline for that later work.

## 8. Recommended reading order

1. Official Jev introduction, API, confidence, and numerical-limitations pages linked above.
2. Krishnamurthy et al. for direct exploration tests and prompt/representation pitfalls.
3. The Thompson sampling tutorial and Bayes-UCB paper for the primary baselines.
4. Sun et al. for hybrid model/classical exploration.
5. IDS and knowledge gradient for explicit information value.
6. GP-UCB, TuRBO, and LLAMBO before designing the numeric optimization study.

The immediate recommendation remains the small exact diagnostic plus the replicated Bernoulli comparison in the plan. Its conclusion should be about the tested version, evidence representation, objective, and task family, rather than a universal yes/no answer to whether Jev “does Bayes.”
