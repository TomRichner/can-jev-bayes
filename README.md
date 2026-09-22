# can-jev-bayes
Jev Bayes, No?

How well can Jev make sequential decisions under uncertainty, and how can Bayesian methods help it learn and act more effectively?

## Research question and motivation

[Jev](https://docs.typesafe.ai/introduction) is TypeSafe AI's model for structured judgments and decisions. This project studies it on classic problems with well-understood Bayesian methods and, where computationally feasible, Bayes-optimal strategies. We begin with multi-armed bandits containing 2, 3, 5, 10, or 15 arms: repeatedly choose an option, observe its reward, and try to maximize total reward within a finite number of turns. In this setting, maximizing expected reward and minimizing expected cumulative regret have the same optimal strategy.

The central difficulty is exploration versus exploitation. An option with a lower estimated reward can be worth trying because the information it provides improves later choices. That information becomes less valuable as the remaining budget shrinks. Bandits let us measure whether a model responds to this tradeoff, distinguish purposeful exploration from randomness, and compare decisions against explicit mathematical references.

Our central hypothesis is that **Bayesian-informed instructions and externally computed statistics can improve Jev's sequential decisions, with benefits that depend on the task, representation, and execution rule**. We test three related predictions:

- Supplying posterior expected rewards and uncertainty estimates can improve decisions by making the implications of observed outcomes easier to use.
- Instructions that explain the future value of exploration, and sampling from Jev's answer distribution, can change how it balances learning and immediate reward.
- Supplying Bayesian action values or recommendations can improve performance beyond statistical summaries alone, provided Jev can reliably use that advice.

These hypotheses motivate separate tests of probability estimation, action selection, and the use of external computation. Comparators include Thompson sampling, Bayes-UCB, knowledge gradient, information-directed sampling, finite-horizon index approximations, and exact two-arm dynamic programming, alongside greedy and random policies. The [protocol](first_experiment_Plan.md), [research notes](research_notes.md), and [selected literature review](docs/literature_design_review.md) describe the methods and experimental rationale.

## Main results so far

Ten experiments and a batching validation are complete for `jev-1.13.0` on stationary Bernoulli bandits: **598,080 evaluation/control questions across fixed-state probes and 1,560 independent online environments**. The [research report](research_report.md) provides effect sizes, uncertainty intervals, figures, and interpretation; the [data audit](docs/data_completeness_audit.md) verifies completeness and provenance.

- **Posterior means helped, but richer summaries were not uniformly better.** In a fresh 100-pull comparison, adding posterior means reduced average regret by 2.78 reward units relative to counts alone. Adding Beta parameters, standard deviations, and credible intervals then increased regret by 0.83 relative to means alone. This tests a representation package; it does not isolate uncertainty information from wording or length.
- **Jev could use external decision calculations effectively.** With exact action values supplied, it incurred zero observed local decision loss in 4,000 sequential decisions. It followed supplied Bayes-UCB indices about 99.93% of the time. A recommendation alone was much more sensitive to its format and surrounding evidence. These results demonstrate use of supplied computation; a deterministic selector can also execute those values or indices.
- **Exploration instructions influenced behavior without reliably producing optimal planning.** Explicitly describing the benefit of exploration improved some choices, but Jev still missed states where trying a lower-mean arm was optimal. Merely naming a Thompson-sampling strategy did not reproduce that algorithm.
- **Sampling had task-dependent benefits.** On fresh ten-arm, 100-pull tasks drawn from the stated prior, sampling Jev's answer distribution reduced regret by 5.44 units relative to its direct choice. The corresponding two-arm effect was individually inconclusive; a prespecified comparison confirmed that sampling was more favorable at ten arms than two.
- **Probability elicitation mattered.** Noul forecasts were closer to Bayesian event probabilities than Choice outputs in controlled comparisons. Separate Noul best-arm forecasts did not consistently sum to one, and required one question per arm instead of a single joint Choice question. Better marginal forecasts therefore leave questions about coherence, cost, and downstream control.
- **The Bayesian comparator changed the conclusion.** Jev outperformed Thompson sampling in some settings, while Bayes-UCB, information-directed sampling, and knowledge gradient achieved lower regret than direct Jev with Bayesian summaries in the broader scaling benchmark. Exact calculations also showed that greedy was already close to optimal in the short two-arm task, making an advantage over Thompson sampling insufficient evidence of sophisticated exploration.

## Continued and next directions

The next phase will replicate the strongest findings on fresh tasks and service dates, isolate which uncertainty fields and advice formats change behavior, and test whether more faithful probability forecasts improve sequential reward. The [continuation plan](docs/next_experiments.md) specifies the proposed comparisons, precision targets, and budget estimates.

We then plan to move into continuous parameter spaces and compare Jev with Bayesian optimization using a Gaussian-process surrogate. Given a limited number of objective evaluations, can Jev find a better minimum when supplied predicted values, uncertainty, or acquisition scores? Candidate generation, selection, and evaluation budgets will be controlled so the contribution of each component can be measured.

A longer-term direction is a reasoning-model harness with access to Jev, Bayesian algorithms, both tools, or neither. This would test whether their combination improves sequential decisions and eventually supports iterative model improvement, including post-training hyperparameter optimization. Reasoning-model and model-training experiments remain future work.

## Reproduce

Install the pinned Python environment and run the checks:

```bash
uv sync --frozen --python 3.12
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

The [recorded datasets](results/) and aggregate reports are committed. Rebuild the summary figures without API calls:

```bash
.venv/bin/python scripts/plot_followup_summary.py
```

For an example collection run, add `jev_key = "your_key"` to the ignored root `.env`, then run:

```bash
PYTHONPATH=src .venv/bin/python -m jevbandits preflight --run-dir artifacts/reproduction_v1 --cap 19
PYTHONPATH=src .venv/bin/python -m jevbandits diagnostics --experiment e1_horizon --run-dir artifacts/reproduction_v1 --cap 19
PYTHONPATH=src .venv/bin/python -m jevbandits online --experiment pilot --run-dir artifacts/reproduction_v1 --cap 19
```

Run the corresponding classical pilot and generate a report from the collected records without further API calls:

```bash
PYTHONPATH=src .venv/bin/python -m jevbandits online --experiment pilot --only baselines --run-dir artifacts/reproduction_v1
PYTHONPATH=src .venv/bin/python -m jevbandits report --run-dir artifacts/reproduction_v1 --output reports/reproduction
```

Live commands spend API credits. `--cap` sets the project-wide spending ceiling, including existing sibling run ledgers. Use one live API writer across the project; classical simulations can run separately. Raw records and checkpoints are stored under ignored `artifacts/`; report generation requires those local records. The linked protocols document the full experiment matrix and follow-up commands. The completed study passed 319 tests, and its estimated Jev cost was $18.10 including conservative retry reserves.

## License

Copyright 2026 Thomas J Richner. Licensed under the [Apache License 2.0](LICENSE); see [NOTICE](NOTICE).
