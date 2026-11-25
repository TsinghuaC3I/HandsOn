# Transfer-16 ordered progress for HandsOn Table 6

This standalone Python 3.8+ scorer reads native control-step logs for one checkpoint or policy. The task map and aggregation follow the archived builder for arXiv v1, Table 6. All percentages in the output use the 0–100 scale. Individual episode progress uses the 0–1 scale. [Table 6](https://arxiv.org/html/2610.26765v1#S5.T6)

## Inputs and fixture

Run these commands from the repository root.

```sh
cd tools/transfer16_progress
python3 score.py fixture/manifest.jsonl > scores.json
diff -u fixture/expected.json scores.json
```

For your policy, supply one JSON Lines manifest per checkpoint, with one row per planned task/trial slot. For a scored trial point `episode_dir` at the folder containing that attempt's `steps.jsonl` and `RESULT.json`. Paths are relative to the manifest. Use the accepted attempt for that slot. The current public leaderboard records its selected attempt in `scored_attempt`. A manifest row looks like this.

```json
{"task":"Flex02","trial":0,"seed":100,"status":"scored","episode_dir":"flex02-0"}
```

Use the short paper ID, public task ID, or historical runtime ID from `stages.tsv`. `trial` is a nonnegative slot index. The optional `seed` is checked against the receipt. Record an invalid or missing slot with `status` set to `invalid` or `missing` and omit `episode_dir`. Include valid early failures and timeouts as scored trials. Each task/trial pair can appear once. A missing stage field that prevents a predicate from being evaluated raises a scoring error. Investigate such errors before comparing the resulting rates.

Each trace starts at the first executed action (`step=1`). Each row contains the control-step number and the native `info` dictionary at `native.info`. Keep the full dictionary, including nested metrics, through the end of the episode. The receipt supplies `task`, `steps`, `success`, and `complete_evaluation_episode`. Keep task/trial identity, seed, and the disposition of invalid or missing slots separately if they are stored outside that receipt. Custom loggers can convert to this layout offline. RGB, actions, and model outputs are unnecessary for this calculation. [Public rollout logger](https://github.com/TsinghuaC3I/HandsOn/blob/cdf5ba2df823626af4144d1cd1ad21957cde7881/handson/handson/evaluation/rollout.py#L105-L165) · [Public attempt selection](https://github.com/TsinghuaC3I/HandsOn/blob/cdf5ba2df823626af4144d1cd1ad21957cde7881/handson/handson/evaluation/leaderboard.py#L273-L313)

## How an episode earns credit

Read `stages.tsv` from top to bottom within each task. Each row gives the stage name, exact predicate on native `info`, and inclusive time limit in 30 Hz control steps. Dots select nested fields. Within one step, `!` negates a boolean, `&` joins conditions that must hold together, and `|` allows alternative conditions. The threshold `0.599999999` for salt and pepper is 0.60 seconds with the original numerical tolerance.

1. For each predicate, record whether it was true on any step from 1 through the limit or earlier episode end. The operands of a geometric fit must coincide on one step. A trace recorded on a longer budget may contain a native success after the cutoff. That success earns zero SR within the paper budget.
2. Count consecutive reached stages from the beginning of the list and divide by the number of stages, including the final native success stage. A later isolated hit earns nothing beyond the first gap. Hits persist when geometry is lost or a native phase regresses within that episode. The offline builder does not compare the first-hit timestamps of successive stages or clear hits on an internal retry. Task-specific latches and phase predicates provide the temporal requirements. A new environment reset starts a separate episode with empty history.
3. For Comb06 the first two predicates are parallel. Either seasoning reaching its threshold earns 1/5. Both earn 2/5, and unlock the remaining prefix. For Flex02 the four predicates are `first_fold_latched`, `second_fold_latched`, `release_latched`, and `succeed`. An initially flat towel alone earns 0/4. For Gen02 the `succeed` row is SR-only because the paper traces lack a graded hinge signal.
4. Compute SR independently from native `succeed` within the limit, and progress from the prefix. A task mean uses its valid trials. Each group mean gives equal weight to each included task. The overall result is also a task mean. A group score is null if any task required for that metric has no valid trials.

| Metric | Gen | Comb | Flex | Overall |
|---|---:|---:|---:|---:|
| Tasks in SR | 7 | 6 | 3 | 16 |
| Tasks in Prog. | 6 | 6 | 3 | 15 |

## Reproducing the paper denominators

The eight trained multi-task checkpoints each had 50 planned slots per task. The original table used 6,232 valid episodes out of 6,400 planned, including 5,832 with graded progress. Gen06 had 43 valid trials and Gen07 had 37 for every checkpoint. The remaining exceptions were Gen05 for FastWAM (48), Comb04 for FastWAM (49), and Comb02 for Motus (48) and HY-Embodied, Spatial Forcing, and FastWAM (49 each). Every other cell had 50. These historical counts describe the paper runs. Use the valid counts from your own receipts for a new policy and report coverage alongside the score.

`paper_task_results.tsv` gives the original builder's unrounded per-task progress, success counts, and actual denominators for all eight checkpoints. For example, OpenWAM's overall progress rounds to 12.9%. The file contains aggregate counts. It does not identify individual excluded seeds.

The fixture is synthetic. Its Flex02 trials earn 0/4 for an initially flat towel, 1/4 for a first fold later lost, and 0/4 for an isolated release. Two additional planned slots are marked invalid and missing. Its Flex02 task mean is 8.3333%, and its Flex group macro is 30.3968%.

## Validation against the archived calculation

The portable scorer was checked on the scoring fields from 22 original control-step traces spanning all 16 tasks, after checking each source trace's checksum. A separate check using saved per-episode predicate summaries reproduced all 64 printed SR and Prog. cells and the original per-task valid counts. The historical summary archive contains two additional Comb02 records relative to the original table population. The exact two omitted trial identities could not be established from the summary archive. The cross-check used score-equivalent exclusions for FastWAM and Motus and matched the original per-task aggregates. The scorer uses the supplied manifest to select your policy's episodes.
