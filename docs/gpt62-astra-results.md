# GPT-6.2 Astra with `move_eef`

HandsOn evaluated `gpt-6.2-astra` at `medium` reasoning effort without benchmark demonstrations or finetuning. Each of the 66 tasks has ten fixed trials.

| Suite | Successes / trials | Success | Stopped at 100 calls |
| --- | ---: | ---: | ---: |
| Core-50 | 46 / 500 | 9.20% | 211 |
| Transfer-16 | 5 / 160 | 3.13% | 69 |
| All 66 | **51 / 660** | **7.73%** | **280** |

The model receives three RGB views, measured robot state, the task instruction, static task guidance, and execution feedback. The controller retains compact feedback and the latest two image turns. The `move_eef` function specifies palm poses and up to 40 finger-joint targets. The native executor runs at 30 Hz and pauses simulation while awaiting the API.

Every trial has a 100-call budget. The controller retains the budget charge for an interrupted request with uncertain billing and charges its replay to the same trial. Budget stops count as unsuccessful. Open Latched Door uses the declared instruction/source waiver. Lift Carton uses native support, lift, and hold checks and records the force magnitude without applying the 12 N threshold.

Transfer-16 successes occur on Orient Bottle Upright (3/10), Open Trash Bin (1/10), and Open Laptop and Click Mouse (1/10). The audit verified all 660 outcomes and all 1,980 videos. The trial file includes hashes for each result and excludes server paths. Raw rollouts and videos are stored on the evaluation server. [Download the task and trial results](results/gpt62-astra-call100.json) · [Unified leaderboard](leaderboard.md).
