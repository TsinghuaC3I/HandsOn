# Task reference

HandsOn registers **66 tasks** through `handson.tasks.CONFIG_MAPPING`.
Every task has a directory under `handson/handson/tasks/<name>/` and an
evaluation config at `configs/multi_task/<name>.yaml`.  The tables below give
one line per task; the full instruction, camera mapping and time limit are in
that task's config.

## Core-50 (50 tasks)

Numbered as in the manuscript.  Each has demonstrations and an execution test.

| Task | Environment | What it tests |
|---|---|---|
| 1 | `t01_align_blocks_in_midair` | Lift both assigned blocks with their assigned hands, then mate their marked faces in midair and hold the… |
| 2 | `t02_connect_charger_and_cable` | Insert the charger plug into the power strip first, then insert the charging cable connector into the charger… |
| 3 | `t03_cover_box` | Align the box lid and place it onto the clear box. |
| 4 | `t04_cover_pitcher` | Align the pitcher lid and place it onto the pitcher body. |
| 5 | `t05_fill_cup` | Hold the white ceramic mug under the outlet with its local +Z axis within 15 degrees, inclusive, of world +Z. |
| 6 | `t06_fit_device_cover` | At each reset device_body and device_cover receive independently randomized tabletop positions and yaw… |
| 7 | `t07_flip_sponge` | Stabilize the YCB skillet with the left hand. |
| 8 | `t08_hammer_nail` | Pick up the wood block with the left hand, keep the vertical nail inserted in the block, then use the right… |
| 9 | `t09_insert_peg` | Insert the round peg into the socket tray hole. |
| 10 | `t10_lift_carton` | Support opposite lower carton sides with both hands, lift the entire carton clear of the table without either… |
| 11 | `t11_line_up_blocks` | Use only the set square tool to push three scattered wood blocks onto the baseline and align them side by… |
| 12 | `t12_load_and_start_microwave` | Open the microwave door, carry the plate with its food box into the microwave, close the door, then press the… |
| 13 | `t13_load_cans` | Pick up the left and right tomato-soup cans with their assigned hands, cross-load each can into the… |
| 14 | `t14_move_and_click_mouse` | Move the mouse onto the mouse pad and click the left mouse button. |
| 15 | `t15_open_drawers` | Use each assigned hand to pull the highlighted drawer on its side beyond 60 percent travel; |
| 16 | `t16_open_laptop` | Use the right hand to lift the initially closed display from its front edge and open it to the round target… |
| 17 | `t17_open_latched_door` | Brace and push the door panel with the left hand. |
| 18 | `t18_open_micropipette_box` | The micropipette box is fixed after reset at a randomized tabletop pose and yaw. |
| 19 | `t19_orient_cup_upright` | Pick up the cup and place it upright on the table with its bottom facing down. |
| 20 | `t20_pass_and_orient_pen` | Touch the pen anywhere with the right hand only, then with both hands at the same time, then with the left… |
| 21 | `t21_pass_block` | Pick up the left block with the left hand, transfer it to the right hand in the middle, then place it stably… |
| 22 | `t22_pass_screwdriver` | Touch the screwdriver with the right hand first, then touch it with the left hand while it is airborne, and… |
| 23 | `t23_photograph_logo` | Pick up the camera, aim it at the logo, and press the shutter button. |
| 24 | `t24_place_ball_in_bowl` | Pick the specified ball and place it into the bowl. |
| 25 | `t25_place_cup_on_plate` | Pick the brown cup and place it at the center of the plate with its bottom facing down. |
| 26 | `t26_place_fruit_in_bucket` | Pick the specified fruit from the mixed fruits and place it into the bucket. |
| 27 | `t27_place_largest_fitting_ball` | Find the target C, G, or J YCB cup in the shuffled cup row. |
| 28 | `t28_place_tableware` | Follow the sampled single-step table-setting instruction. |
| 29 | `t29_place_wine_glass` | Place the YCB plastic wine glass upright at the center of the YCB plate, release it, and leave it stable. |
| 30 | `t30_pour_seasonings` | The salt jar and cyan target are randomized within the left-hand comfort zone; |
| 31 | `t31_prepare_coffee` | Open either cabinet door, remove the mug, then close both doors. |
| 32 | `t32_press_soap_pump` | Place the dish sponge under the dish-soap bottle, hold the bottle with the right hand, align its outlet over… |
| 33 | `t33_reposition_marker_grasp` | Pick up the capped YCB marker with the right hand. |
| 34 | `t34_roll_ball_through_gate` | The tennis ball may start on either side of the centred cracker and sugar boxes. |
| 35 | `t35_rotate_propeller` | Rotate the YCB airplane propeller through 1080 degrees of accumulated absolute angular travel. |
| 36 | `t36_scoop_marbles` | Use the right hand to control the spoon: keep it level or slightly nose-down, scoop the marble, carry it over… |
| 37 | `t37_scrub_bottle` | Retrieve and stabilize the labeled water bottle with the left hand, retrieve the cleaning brush with the… |
| 38 | `t38_set_dual_knobs` | Use the left hand on the left knob and the right hand on the right knob, start both rotations together, set… |
| 39 | `t39_set_timer` | Stabilize the YCB timer with the left hand, program the prompted minutes and seconds with a right fingertip,… |
| 40 | `t40_slide_knife_blade` | Pick up the utility knife with the right hand. |
| 41 | `t41_sort_nested_cups` | Separate the three nested YCB cups, then place them upright in a straight row from left to right in the main… |
| 42 | `t42_sort_objects` | Find six targets among the clutter and sort boxes, cans, and dishes into their marked baskets. |
| 43 | `t43_squeeze_scissors` | Lift the scissors clear of the table and make three complete cuts in the air. |
| 44 | `t44_stack_blocks` | Arrange the four blocks into four distinct height layers with the large wood block on the bottom layer, then… |
| 45 | `t45_stir_pot` | Hold the Dutch oven fixed with the left hand while the right hand moves the spoon through six ordered… |
| 46 | `t46_store_charger` | Hold the drawer open with the left hand, place the charger fully inside with the right hand, release the… |
| 47 | `t47_trigger_drill_on_block` | Bring the unpowered YCB drill into physical contact with the YCB wood block, then press the physical trigger… |
| 48 | `t48_unfold_fan` | Use both hands to grasp the two outermost fan ribs and unfold the fan symmetrically to the episode target… |
| 49 | `t49_unfold_ruler` | Pull both outer ruler segments through their real hinges until all three segments are straight. |
| 50 | `t50_water_plant` | Grasp and aim the spray bottle before pressing; |

## Transfer-16 (16 tasks)

Evaluation-only: no target-task demonstrations.  These reuse Core-50 assets
and skills in combinations or with shifted language.

### Compositional (6)

| Task | Environment | What it tests |
|---|---|---|
| `eval_comb01_cover_pitcher_and_box` | `eval_comb01_cover_pitcher_and_box` | First align and seat the pitcher lid, then release it and let it settle. |
| `eval_comb02_open_laptop_and_click_mouse` | `eval_comb02_open_laptop_and_click_mouse` | Open the initially closed laptop to the requested angle with the right hand and hold it steadily. |
| `eval_comb03_pass_block_and_close_box` | `eval_comb03_pass_block_and_close_box` | Pick up the left block with the left hand, hand it to the right hand in the middle, and release it stably… |
| `eval_comb04_place_fruit_and_photograph_logo` | `eval_comb04_place_fruit_and_photograph_logo` | Pick the specified fruit, place it in the bucket and release it. |
| `eval_comb05_scrub_bottle_then_fill_cup` | `eval_comb05_scrub_bottle_then_fill_cup` |  |
| `eval_comb06_season_and_stir` | `eval_comb06_season_and_stir` | Use the left hand for salt and the right hand for pepper. |

### Generalization (7)

| Task | Environment | What it tests |
|---|---|---|
| `eval_gen01_cover_lunchbox` | `eval_gen01_cover_lunchbox` | Align the lunchbox lid and place it onto the lunchbox. |
| `eval_gen02_open_trash_bin` | `eval_gen02_open_trash_bin` | The trash bin is fixed after reset at a randomized tabletop pose and yaw. |
| `eval_gen03_orient_bottle_upright` | `eval_gen03_orient_bottle_upright` | Pick up the yellow mustard bottle and place it upright on the table with its bottom facing down. |
| `eval_gen04_pass_glue_stick` | `eval_gen04_pass_glue_stick` | Pick up the glue stick with the left hand, transfer it to the right hand in the middle, then place it stably… |
| `eval_gen05_set_mixer_knobs` | `eval_gen05_set_mixer_knobs` | Use your left hand to turn the left knob counterclockwise and your right hand to turn the right knob… |
| `eval_gen06_spray_shoe` | `eval_gen06_spray_shoe` | Grasp and aim the spray bottle at the shoe upper before squeezing the trigger. |
| `eval_gen07_stack_boxes` | `eval_gen07_stack_boxes` | Arrange the four package boxes into four distinct height layers with the large package box on the bottom… |

### Deformable (3)

| Task | Environment | What it tests |
|---|---|---|
| `eval_flex01_compress_sponge_into_mug` | `eval_flex01_compress_sponge_into_mug` | Use the left hand to stabilize the YCB mug. |
| `eval_flex02_fold_towel_twice` | `eval_flex02_fold_towel_twice` | Using both hands, fold the flat towel in half once. |
| `eval_flex03_pack_pillow` | `eval_flex03_pack_pillow` | Use both hands to lift the foam pillow clear of the table, place it fully inside the open carton, release… |

## Regenerating this file

The tables are derived from each task's `configs/multi_task/<name>.yaml`
`prompt` field.  If a task's prompt changes, update the corresponding row.
