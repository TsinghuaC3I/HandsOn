# B03 Newton 跨 GPU 物理复现

## 边界

B03 的正式 Newton 数据与 Isaac Sim 任务共用同一套 scene、seeded
reset、observation 和“插入后稳定 0.5 s”success 逻辑。右侧 ROKAE + Wuji
执行任务，左侧保持初始驱动目标。延后机械臂模式只影响实时显示和
后处理渲染，物理 scene 内始终是有限质量的完整机械臂和手。

正式路径禁止每帧写 root pose、joint state 或物体 state。采集端记录每个
Newton physics substep 的完整 position-drive target；复现端从相同 seed
创建全新 scene，只写这些 drive target。

## 正式 runtime

`cross_gpu_v1` 会硬性检查：

- Newton `>=1.4.0`；
- Isaac Lab Newton `>=0.13.6`、Newton Actuators `>=0.1.0`；
- Warp `>=1.15.0`；
- MuJoCo `>=3.10.0`、MuJoCo-Warp `>=3.10.0.2`；
- Warp 全局模式和 `SolverMuJoCo` 实例模式均为
  `DeterministicMode.GPU_TO_GPU`；
- Newton `CollisionPipeline(deterministic=True, broad_phase="explicit")`；
- `njmax` / `nconmax` 由 Newton 1.4 从最终模型估算，禁止沿用旧后端的
  `8192 / 4096` 过度预分配（会使确定性散射记录超过 int32 上限）；
- 禁用不参与任务状态读回的 MuJoCo 只读 sensor pass；训练状态继续来自
  Newton/Isaac Lab tensor。MJWarp 3.10 的 tactile sensor kernel 会在同一输出
  上混用 `max/add` 原子操作，Warp 的 `GPU_TO_GPU` 模式会按设计拒绝它；
- 采集与复现使用完全相同的 Python、依赖版本、操作系统、
  NVIDIA driver 和 CUDA toolkit；认证时只允许 GPU 型号变化。
- B03 物理、录制、回放代码和关键 MJCF 资产的 SHA-256 完全一致。

Isaac Sim 6 官方环境仍锁定旧版 Newton/Warp。数采工作台更新后会在后台
自动安装固定版本的 Isaac Sim 6.0.1、Isaac Lab Newton 和 GPU_TO_GPU
overlay；也可以手动执行完整引导器：

```bash
./scripts/install_isaacsim_newton_runtime.sh
```

默认安装日志位于
`../handson-local-state/runtime/newton-runtime-install.log`。设置
`HANDSON_AUTO_INSTALL_NEWTON=0` 可以禁用工作台启动时的自动下载。

`run_collection_ui.sh` 仅会把该 overlay 注入 `isaacsim_newton` 子进程；
数采 UI、普通 MuJoCo 和 Isaac Sim/PhysX 进程都不会加载它，避免
Python 3.11/3.12 ABI 串环境。

项目隔离运行时锁定 Newton `1.4.0`、Warp `1.15.0`、MuJoCo
`3.10.0` 和 MuJoCo-Warp `3.10.0.3`。采集机和认证机都必须用上面的
安装脚本创建同一组 overlay，不能只满足最低版本。

录制 schema 和每份复现报告都会同时封存 Warp 全局模式、
`SolverMuJoCo` 实例模式、确定性碰撞管线及代码/资产指纹；缺少任一项即
拒绝认证。

这是“实际跨型号、语义容差内复现”，不是整个求解器的 bitwise
声明。`GPU_TO_GPU` 保证受支持的 Warp 原子操作跨 GPU 可确定；
Newton 1.4 还会根据最终模型自动计算 MuJoCo-Warp 生成内核的记录上限，
不再使用 Newton 1.3 时会溢出的固定上限。发布资格仍由另一 GPU
型号上的三次完整轨迹实测授予。

不满足时会拒绝开始正式采集，不会自动降级。`local_debug_v1` 也使用
`GPU_TO_GPU`，但仅供本机 preview/烟测，生成的报告不能解锁上传。

每次更换工具链或修改物理代码后，先运行两个本机独立进程的语义容差
烟测：

```bash
./scripts/run_newton_gpu_to_gpu.sh \
  scripts/smoke_b03_newton_fresh_process.py
```

它会比较 joint/TCP/任务刚体的位置、姿态和速度，并检查零轨迹状态
写入与 scene 完整销毁。它只能证明本机基线，不能替代下面的跨 GPU
认证。

## 采集与认证

1. 在数采 UI 选择 `t09_insert_peg` / `isaacsim_newton`。Preview 使用
   `local_debug_v1`，正式采集自动使用 `cross_gpu_v1`。
2. 将产生的 `*__newton_v2` demo 复制到另一种 GPU 型号的机器。GPU UUID
   不同但型号相同不算跨型号。
3. 在第二台机器的同版本工具链中执行：

   ```bash
   ./scripts/run_newton_gpu_to_gpu.sh \
     scripts/certify_b03_newton_cross_gpu.py \
     /absolute/path/to/b03_demo__newton_v2
   ```

命令会顺序启动 3 个独立 Python 进程。每次必须完整复现 success、接触可读、
手-柱销实体穿透不超过 1 mm、轨迹状态写入计数为 0，并成功销毁 scene。
通过后会生成：

```text
newton_cross_gpu_certification_runs/run_001.json
newton_cross_gpu_certification_runs/run_002.json
newton_cross_gpu_certification_runs/run_003.json
newton_cross_gpu_acceptance.json
```

上传门禁会重新校验 demo 输入 hash、三份报告 hash、软件身份、GPU 型号、误差、
接触、success 和 scene lifecycle。任意输入或报告在认证后发生变化，上传会重新
锁定。

本机仅做非发布烟测时：

```bash
./scripts/run_newton_gpu_to_gpu.sh \
  scripts/certify_b03_newton_cross_gpu.py \
  /absolute/path/to/b03_demo__newton_v2 \
  --same-gpu-smoke
```
