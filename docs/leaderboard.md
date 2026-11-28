# HandsOn leaderboard

[Interactive leaderboard](https://llltttwww.github.io/HandsOn-website/leaderboard.html) | [Machine-readable table](results/leaderboard.json)

The single leaderboard reports native task success on Core-50, Transfer-16 and all 66 tasks. Overall is the task-macro mean across the full catalog: `(50 x Core + 16 x Transfer) / 66`. Each model has equal planned trials per task within its own evaluation, so this also equals its pooled success rate where all outcomes are available. Higher is better.

<!-- leaderboard-table:begin -->
| Rank | Model | Core-50 | Transfer-16 | Overall | Evaluation |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | OpenWAM | 29.1% | 0.4% | 22.1% | HandsOn team |
| 2 | FastWAM | 26.4% | 0.0% | 20.0% | HandsOn team |
| 3 | pi0.5 | 24.9% | 0.1% | 18.9% | HandsOn team |
| 4 | DM0.5 | 21.4% | 0.9% | 16.4% | HandsOn team |
| 5 | LingBot-VLA | 19.7% | 0.0% | 14.9% | HandsOn team |
| 6 | HY-Embodied | 18.9% | 0.0% | 14.3% | HandsOn team |
| 7 | Motus | 18.1% | 0.1% | 13.7% | HandsOn team |
| 8 | Spatial Forcing | 12.4% | 0.1% | 9.4% | HandsOn team |
| 9 | [GPT-6.2 Astra (`move_eef`)](gpt62-astra-results.md) | 9.2% | 3.1% | 7.7% | HandsOn team |
| 10= | [Fable 5.2 (`move_eef`)](fable52-results.md) | 5.2% | 1.9% | 4.4% | HandsOn team |
| 10= | [GPT-6 (`move_eef`)](gpt6-results.md) | 5.2% | 1.9% | 4.4% | HandsOn team |
| - | DP (single-task) | 9.7% | not evaluated | not ranked | HandsOn team |
<!-- leaderboard-table:end -->

The eight trained-model Overall entries are **derived from the manuscript's suite rates rounded to one decimal place**; they are not an additional 66-task evaluation. Core uses 100 planned trials per task and Transfer uses 50, subject to the manuscript's valid-reset caveats. DP is a separate Core-only reference and has no 66-task rank.

GPT-6, GPT-6.2 Astra, and Fable each cover ten fixed trials per task. GPT-6.2 Astra uses a 100-call limit for every trial. GPT-6 retains historical mixed context and call-budget settings. **Fable's displayed result is a post-hoc 100-response cutoff of trajectories originally allowed up to 500 model calls**, not an independent 100-call run. In the current Fable snapshot, 656 of 660 original trials passed independent native audit; the four continuing trajectories had already crossed response 100. A replaced trajectory requires a refreshed score. Observation, control interface, interaction budgets and sample counts differ across methods, so ranks describe published outcomes under each declared protocol rather than a controlled head-to-head comparison.

Sources: [paper](https://arxiv.org/abs/2610.26765), [GPT-6 trial results](results/gpt6-results-20261019.json), [GPT-6.2 Astra trial results](results/gpt62-astra-call100.json), [Fable 100-call trial outcomes](results/fable52-call100.json). The machine-readable table records the per-model protocol and the Fable snapshot audit count.

## Submit results

The leaderboard accepts submissions from other teams. Run the frozen
`handson-eval-v1` protocol with `python -m handson.evaluation.suite`, then open
a pull request that adds `leaderboard/submissions/<id>/`. CI checks every
planned trial, the recomputed scores and the evaluator commit.

Entries are labelled by who produced the scores:
- **HandsOn team**: evaluated by the maintainers.
- **Verified**: reproduced by the maintainers on hidden seeds.
- **Self-reported**: checked by CI only. These are listed but not ranked.

See the [submission guide](../leaderboard/README.md) for tracks, rules and commands.
