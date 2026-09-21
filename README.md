# can-jev-bayes
Investigating Jev on sequential decisions: how objectives, Bayesian summaries, external computation, and action selection affect learning and reward.

See [the protocol](first_experiment_Plan.md), [research notes](research_notes.md), and [selected literature review](docs/literature_design_review.md). The study does not presume a positive or negative result.

## Reproduce

```bash
uv sync --frozen --python 3.12
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m jevbandits preflight
PYTHONPATH=src .venv/bin/python -m jevbandits diagnostics --experiment e1_horizon
PYTHONPATH=src .venv/bin/python -m jevbandits online --experiment pilot
PYTHONPATH=src .venv/bin/python -m jevbandits online --experiment e4_scaling --only baselines
PYTHONPATH=src .venv/bin/python -m jevbandits report --output reports/overnight_v2
```

Live commands read `jev_key` from ignored `.env` and spend API credits. Raw records and checkpoints are under ignored `artifacts/`. Offline reports need no API key. Use only one live API writer per ledger; baseline simulation can run separately.

For this repo, my research question is how Jev does on classic problems where there are Bayesian-optimal strategies or at least well-known Bayesian methods.  For example, can we test Jev on a multi-arm bandit problem with 2, 3, 5, 10, 15 arms?  Can we ask it to minimize regret, or just maximize reward over a finite number of turns?  How does it compare to different classic methods like thompson sampling?  What can we say about its exploration-exploitation strategies?  Additionally, we could test it in a numeric parameter space against Bayesian optimization with a Gaussian process surrogate.  Maybe it needs to try to find the best minimum it can in a high dimensional space within a certain number of turns.  In general, I'm interested in sequetial tasks where there are multiple options.  Additionally, I'd be interested to see how well it can do if we give it some Bayesian stats about expected value and confidence.  Can it make use of this?  Further down the road, can test having a generative (reasoning) AI model make use of both Jev and Bayesian model to do RSI?

## License

Copyright 2026 Thomas J Richner. Licensed under the [Apache License 2.0](LICENSE); see [NOTICE](NOTICE).
