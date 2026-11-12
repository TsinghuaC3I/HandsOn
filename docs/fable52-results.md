# Fable 5.2 with `move_eef`

This release records the first 100 completed model responses in each existing Fable 5.2 trajectory. The original evaluation allows **up to 500 model calls per episode**. The figures below are a **post-hoc 100-response truncation**, not results from a separate 100-call rerun. A model response, rather than a tool action or network retry, consumes one unit of this cutoff.

| Suite | Success by response 100 | Native failure by response 100 | Still active at response 100 | Fixed trials | Success |
| --- | ---: | ---: | ---: | ---: | ---: |
| Core-50 | 26 | 205 | 269 | 500 | 5.2% |
| Transfer-16 | 3 | 62 | 95 | 160 | 1.9% |
| All 66 | **29** | **267** | **364** | **660** | **4.4%** |

All 660 fixed slots have an observed 100-response classification. At the snapshot, **656 original trials passed independent native audit** (649 native-complete, seven stopped at the original call budget); four original trajectories were still running, each past response 100. The score is settled for the current accepted trajectories. An infrastructure restart that replaces any of those four trajectories requires regenerating and verifying this record. Do not present this snapshot as final audit of all original 500-call episodes.

One original success falls outside the cutoff: Core task Lift Carton, trial 009, succeeded at response 179. Rotate Propeller, trial 000, succeeded exactly at response 100 and is included. A trajectory that continued to response 101 counts as unsuccessful at the 100-response cutoff, regardless of any later result.

[Download the 660 trial classifications and 66 task rows](results/fable52-call100.json). That file includes the original manifest, queue and audit SHA-256 digests and the SHA-256 of the source snapshot, while omitting private host paths and worker details. [View the unified leaderboard](leaderboard.md).

The prompted controller used three RGB views, measured state, task instruction, execution feedback and the `move_eef` palm/finger interface. Equalizing the call-count cutoff alone does not make it a matched GPT-6 comparison: saved context, control implementation and trial histories may differ. The [GPT-6 release](gpt6-results.md) also retains mixed historical call budgets rather than a uniform 100-call ablation.
