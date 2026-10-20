# HandsOn Newton 任务迁移模板

这份模板用于把已有 Isaac/PhysX 数采任务增量迁移到 NVIDIA Newton。
迁移只能改新任务分支和共用 Newton 基础设施，不能改原任务的
Isaac/PhysX 路径，也不能改已经验收的 A01 Newton 行为。

## 1. 迁移规格

先复制下面的规格，每一项都必须有任务特定值：

```text
task_name: <task_name>
reference_isaac_task: <same task on isaacsim>
newton_backend_name: isaacsim_newton
active_arm_sides: <right/left/both>
task_collider_groups:
  <role>: <unique USD path token>
minimum_geometric_clearance_m: <smallest intended clearance>
task_contact_gap_m: <per role; pair sum must be below intended clearance>
hand_contact_gap_m: <large enough for speculative grasp contact>
success_condition: <reuse the same task success implementation>
recording_schema_name: handson.<task>_newton_ground_truth
scene_lifecycle: one demo / one fresh process / destroy scene in finally
determinism_profile: cross_gpu_v1
cross_gpu_certification: 3 fresh processes on a distinct GPU model
```

## 2. 代码接入点

1. 在任务 `config.py` 中只新增 `isaacsim_newton` 和 Newton 别名，路由到
   `HandsOnABCIsaacLabEnv(..., physics_backend="newton")`。保留 `isaacsim`
   的 PhysX 路由不变。
2. 把新任务加入 `abc_collection_env.py` 的 Newton 白名单，数采 UI
   和 `record_demos_zarr.py` 的 Newton 白名单。
3. 重用 A01 的 Newton solver、Wuji 完整手部 collider、Ideal-PD 驱动和
   机械臂重力补偿。任务自己只定义物体接触包络和重置。
4. 不复制 A01 的接触间隙数值。根据新任务的真实最小装配间隙设置，
   必须满足：

   ```text
   mating_gap_a + mating_gap_b < minimum_geometric_clearance
   ```

   所有任务 collider 组和 52 个 Wuji 手部 collider 都要在启动时点数；
   缺任意一个必须拒绝启动，不能降级继续。
5. 直接复用 Isaac 任务的 reset、observation、success 和 tracker action
   语义；不在 Newton 分支里写第二套任务逻辑。

## 3. Tracker 和机械臂不变式

- 位姿动作为全零时，`*_pose_commanded` 必须为 false，不得运行 IK，
  机械臂驱动目标保持当前值。
- tracker 没有产生新位姿时，不得用 TCP 读回、物体位姿、重置值或
  拟合轨迹反向驱动机械臂。
- 验收至少连续运行 120 个控制帧，记录关节目标、关节读回和
  `pose_commanded`；不能只测 Python 属性。

## 4. 抓取和防穿模验收

- 使用新任务的 Newton 实时 tracker 操作抓取，不得使用旧 Isaac
  动作流代替。
- 手部闭合期间必须有 Newton 手-物体接触，抓起后物体要有明确离桌
  高度和随手位移。
- 用 Newton 接触点、法向、shape margin 计算最大实体穿透深度；
  仅看渲染视频或仅看物体被抬起都不够。
- 装配任务还必须完成原 Isaac 的 success 条件，包括原有稳定保持时间。

## 5. 录制和物理复现

1. 录制时保存每个 Newton physics substep 的完整机械臂和手部 drive
   target、joint readback、TCP 和任务刚体读回。
2. 复现只能从记录的 collection/reset seed 创建全新 scene，并在每个
   physics substep 只写 drive target。动作开始后写 joint state、TCP state
   或 task rigid-body state 均为失败。
3. 每条 demo 独占一个新进程。不论成功、失败或异常，都必须在
   `finally` 调用 `handson.sim.isaaclab.newton_replay.destroy_newton_scene`。
   必须确认 simulation 已停止、Newton manager 已清空、
   `SimulationContext` 已清空，然后才能发布复现报告。
4. 物理复现入口必须拒绝下列输入：临时合成 zarr、旧 Isaac 动作流、
   非 success 数据、非 Newton 录制、缺少子步 target 或 metadata 步数不匹配。
5. 跨 GPU 发布必须开启 Newton deterministic collision ordering，
   并在另一 GPU 型号上进行三次语义容差认证。Python、依赖、系统、
   NVIDIA driver 和 CUDA toolkit 必须与源端一致；同型号不同 UUID
   不能充当跨 GPU 型号证据；未实测通过时不得声明 bitwise 或跨型号可复现。

## 6. 发布门禁

只有以下证据全部存在时才能说“迁移成功”或“物理复现通过”：

- 新任务的 Newton UI 启动和完整 collider 启动报告。
- tracker 静止实时读回证据。
- 新任务 Newton 实际抓取、无实体穿透、原 success 条件通过。
- 由该成功轨迹正式录制的 zarr 和 metadata。
- 另一种 GPU 型号上的 3 个新进程均从全新 scene 完整物理复现，
  读回误差在 schema 阈值内，且每份报告均为 `scene_destroyed=true`。
- 原 Isaac/PhysX 任务回归测试通过，A01 任务文件无 diff。

当前 B03 可复用的代码底座是
`handson/handson/sim/isaaclab/newton_replay.py`；B03 任务特定的严格复现入口是
`scripts/replay_b03_newton_physics.py`。迁移下一个任务时复制任务特定入口，
但继续复用通用 drive-target 和 scene-destroy 实现。
B03 的跨型号运行手册见 `docs/b03_newton_cross_gpu_replay.md`。
