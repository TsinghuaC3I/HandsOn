# 历史 MuJoCo 原型说明

> **历史文档，不适用于当前发布运行时。** 下文记录早期 Panda/Allegro + MuJoCo 原型；其中的旧任务环境、硬件桥接和私有采集工具并未完整随本仓库发布。缺失文件仅作为历史路径保留，不是可运行的安装入口。
> 当前 HandsOn 论文评测采用 Isaac Sim 5.1 / Isaac Lab 2.3.2、PhysX、ROKAE/Wuji 和实际 30 Hz 动作时钟。请先阅读[安装指南](installation.md)、[公共策略接口](custom_policy_integration.md)及[运行验证范围](release-validation.md)。
> 以下关于 MuJoCo 引擎、控制器、数据通路和频率的描述只针对历史原型。

> 本文解释 HandsOn 仿真系统是如何搭建并运转的，重点回答五个问题：
> 1. 怎样设置场景（scene）
> 2. 怎样往场景里加入可交互的 3D asset
> 3. 遥操（teleoperation）输入的信号是什么格式
> 4. simulator 如何用这些信号驱动机械臂位姿与灵巧手形态
> 5. 输出数据怎样保存
>
> 全系统基于 **MuJoCo** 物理引擎 + **Gymnasium** 环境接口，机器人为
> **Franka Panda 7-DoF 机械臂 + Wonik Allegro 16-DoF 灵巧手**。

---

## 0. 总体架构

数据从硬件到磁盘的完整链路：

```
Vive Tracker ──┐                                    ┌─► MuJoCo 物理步进 (mj_step)
(手腕6D位姿)    │  UDP:5012  ┌──────────────────────┐ │
               ├──────────► │  Teleop Wrapper       │ │   opspace 算力矩 → 驱动 Panda
Rokoko 手套 ──► │            │ (ActionWrapper)       │─┼─► position 执行器 → 驱动 Allegro
  │ GeoRT retarget          │  组装 action 向量      │ │
  │  UDP:5014(右/单) 5016(左)└──────────────────────┘ │
  └──────────────────────────────────────────────────┘
                                                       ▼
                                          观测(图像+state) ──► Zarr + MP4 落盘
```

代码骨架对应关系：

| 层 | 文件 |
|---|---|
| MuJoCo Gym 基类 | `handson/sim/mujoco_gym_env.py` |
| 具体任务环境（场景+逻辑） | `handson/sim/envs/panda_water_plant_env.py` 等 11 个 |
| 场景 XML（MuJoCo 模型） | [`handson/sim/envs/xmls/`](../handson/handson/sim/envs/xmls/) |
| 机械臂操作空间控制器 | `handson/sim/controllers/opspace.py` |
| 遥操输入 wrapper | [`handson/tasks/sim_teleop.py`](../handson/handson/tasks/sim_teleop.py) |
| 任务配置（装配 wrapper） | [`handson/tasks/*/config.py`](../handson/handson/tasks/) |
| 录制脚本（落盘） | [`scripts/record_demos_zarr.py`](../scripts/record_demos_zarr.py) |
| Zarr 存储 | [`handson/data/episode_store.py`](../handson/handson/data/episode_store.py) |
| 硬件→UDP 的桥接 | `teleoperation/vive_bridge/`, `teleoperation/rokoko/`, `teleoperation/GeoRT/` |

---

## 1. 怎样设置场景（Scene Setup）

场景完全由 **MuJoCo XML** 描述，再由 Python 环境类加载。

### 1.1 XML 通过 `<include>` 组合

以浇花任务为例，主场景文件
[`xmls/arena_arm_hand_plant.xml`](../handson/handson/sim/envs/xmls/arena_arm_hand_plant.xml)
本身只负责"房间/桌子/相机/灯光"，机器人和物体都是 `include` 进来的：

```xml
<mujoco model="Arena_Allegro_Plant">
  <include file="panda_allegro_copy.xml" />  <!-- 机械臂 + 灵巧手 -->
  <include file="plant.xml" />               <!-- 可交互物体：植物 -->
  <include file="spray.xml" />               <!-- 可交互物体：喷壶 -->

  <option timestep=".002" noslip_iterations="5" .../>   <!-- 物理求解参数 -->
  ...
</mujoco>
```

这种"主场景 = 房间模板 + N 个 include 资产"的写法，使得**换任务只需换 include 的物体 XML**，房间/桌子/机器人模板可以复用。

### 1.2 静态环境（房间 + 桌子 + 相机 + 灯）

在主 XML 的 `<worldbody>` 里直接写死：

- **地面 / 墙**：`type="plane"` 的 floor + 6 个 `type="box"` 的 wall，墙只做视觉（`contype="0" conaffinity="0"`，不参与碰撞）。
- **桌子**：一个 `body name="table"`，里面包含
  - `table_collision`（碰撞 box，`group="3"`，给定 friction）
  - `table_visual`（视觉 box，贴材质）
  - 4 条 `table_leg_*`（圆柱腿）
- **相机**：多个 `<camera>`，关键的是 `front`（第三人称）和 `handcam_rgb`（腕部相机，定义在机器人 XML 里）。
- **灯光**：`<light>` + `<visual><headlight>`。

`<asset>` 段集中声明所有 **texture / material**（地板、桌面、墙面，以及一组用于域随机化的桌面纹理 `table_bamboo`、`table_metal`…）。

### 1.3 Python 侧加载场景

`MujocoGymEnv`（gym.Env 基类）完成模型加载与时间步设置：

```python
self._model = mujoco.MjModel.from_xml_path(xml_path.as_posix())  # 编译 XML
self._data  = mujoco.MjData(self._model)                          # 运行时状态
self._model.opt.timestep = physics_dt          # 物理步长 0.002s (500Hz)
self._n_substeps = int(control_dt // physics_dt)  # 0.02/0.002 = 每个控制步走10个物理子步
```

具体任务类（如 `PandaWaterPlantGymEnv`）在 `__init__` 里**缓存各种 id**（关节、执行器、传感器、相机、site 的索引），并定义 `observation_space` / `action_space`。

### 1.4 场景的"初始化与随机化"在 `reset()` 里完成

每个 episode 开始时 `reset()` 做的事：

1. `mujoco.mj_resetData()` 复位物理状态。
2. **桌面高度随机**：`delta_h = uniform(0, 0.05)`，整体抬高桌子 body 并同步加长桌腿。
3. **物体位姿随机**：在预设范围内采样喷壶/植物的 xy 位置写回模型。
4. **机械臂复位**：把 Panda 7 关节写成 `_PANDA_HOME`，Allegro 16 关节写成 `_ALLEGRO_HOME`。
5. `mujoco.mj_forward()` 前向运算，把 mocap 目标体对齐到当前末端位姿。
6. 若 `randomize=True`，再做**视觉域随机化**：
   - `randomize_lighting()`：灯位/灯向/漫反射/headlight 抖动
   - `randomize_camera()`：从 `replay_cameras.npy` 里随机挑一个第三人称机位
   - `randomize_desktop_texture()`：随机换桌面材质
   - （`randomize_dynamics=True` 时还会随机喷壶的摩擦/刚度/质量）

> 这就是 README 里 `rand_obj`（只随机物体+桌高）和 `rand_full`（再加相机/灯光/纹理）两档随机化的实现处。

---

## 2. 怎样加入可交互的 3D Asset

每个可交互物体是一个**独立的 XML 文件**，被主场景 `include` 进来。它的"可交互性"取决于两点：**有没有自由度（joint）** 和 **碰撞 geom 怎么配**。

### 2.1 视觉 / 碰撞分离（两套 geom）

以 [`spray.xml`](../handson/handson/sim/envs/xmls/spray.xml)（喷壶）为例，用 `<default class>` 区分：

```xml
<default class="spray/visual">
  <geom condim="1" contype="0" conaffinity="0" group="2"/>   <!-- 只渲染，不碰撞 -->
</default>
<default class="spray/collision">
  <geom condim="4" contype="1" conaffinity="14" friction="1 0.005 0.0001"
        solref="0.001 2" group="3" density="100"/>            <!-- 参与物理碰撞 -->
</default>
```

- **视觉 geom**：直接挂高精度三角网格 `mesh`（`.obj`），贴 material，`contype=0` 不参与碰撞 → 好看但不影响物理。
- **碰撞 geom**：用同样或简化的几何，`contype/conaffinity` 打开，给 friction/density → 真正决定抓握手感。

网格资产在 `<asset>` 里声明，例如：
```xml
<mesh file="spray/textured_objs_thickness/original-15/original-15.obj"
      name="original-15" scale="0.2 0.2 0.2"/>
```
（`textured_objs/` 这些目录就是从真实扫描/CAD 切出来的网格件。）

### 2.2 用关节赋予"可交互的自由度"

物体能不能被操纵，取决于它身上挂了什么 joint：

- **整体可被抓取移动** → `<freejoint>`（6 DoF 自由漂浮）。
  喷壶根体 `link_2` 就挂了 `<freejoint name="spray_root"/>`，所以可以被手抓起来、自由移动。
- **铰接可动部件** → `<hinge>` joint + 执行器。
  喷壶的扳机 `link_0` 上有：
  ```xml
  <joint name="joint_0" type="hinge" range="0 0.45" stiffness="1" damping="2"/>
  ...
  <actuator>
    <motor name="joint_0" joint="joint_0" ctrlrange="-1 1" gear="16"/>
  </actuator>
  ```
  外加一个传感器 `<jointpos name="spray_joint_0_pos" joint="joint_0"/>` 读扳机角度。
  仿真逻辑里读这个角度判断"扳机有没有被扣下"，进而控制喷雾锥 `cone_visual` 的可见性（见 `step()` 中 `_TRIGGER_PULL_THRESHOLD`）。
- **纯刚体障碍/容器** → 不挂 joint（如 [`plant.xml`](../handson/handson/sim/envs/xmls/plant.xml) 的植物用固定 body）。

### 2.3 隐藏的判定辅助元素：site 与 group

- **`<site>`**：不可见的参考点，用于成功判定。例如植物里的 site 和喷壶的 `ref_point`，`step()` 里用它们算"喷头是否对准植物半径 0.2m 的圆柱区域内"来判成功。
- **geom group 技巧**：植物的判定圆柱用 `group="5"` 标记，平时不渲染；human 渲染时临时把 group5 切到 group0 显示，渲染完再切回（`_temporarily_show_group5_in_gui` / `_restore_group5`），既能在调试时看到判定区又不污染采集图像。

### 2.4 在 reset 里把物体"摆"到随机位置

加入物体后，环境在 `reset()` 里通过改 `model.body_pos` 或写 `data.jnt(...).qpos` 来设定物体初始位姿（见 1.4 第 3 步）。freejoint 物体改 `qpos[:3]`，固定 body 改 `body_pos`。

> **小结：加一个可交互物体的标准流程** = 写物体 XML（视觉 mesh + 碰撞 geom + 合适的 joint/actuator + 判定 site）→ 在主场景 `include` → 在环境类 `__init__` 缓存其 id → 在 `reset()` 里随机摆放 → 在 `step()`/`_compute_success()` 里写交互与判定逻辑。

---

## 3. 遥操输入信号的格式（含硬件接入）

遥操有两路独立信号——**手腕位姿**和**手指关节**，最终都以 **UDP 裸字节（`numpy.tobytes()`）** 进入仿真侧的 [`sim_teleop.py`](../handson/handson/tasks/sim_teleop.py)。但两路的硬件来源和中间处理完全不同：手腕走 **Vive Tracker → OpenVR**，手指走 **Rokoko 手套 → Rokoko Studio → GeoRT 重定向**。

### 3.0 从硬件到 wrapper 的完整链路

```
[Vive Tracker]                                   ┌─► UDP 5012 ──────────────┐
   └─ OpenVR/SteamVR ── send_vive_pose.py ───────┘  (3×4 位姿矩阵, float64)  │
                                                                            ▼
[Rokoko 手套]                                                        ┌──────────────┐
   └─ Rokoko Studio                                                  │ sim_teleop.py│
       └─ UDP 14044 (JSON 场景, 21 关键点)                            │  后台线程接收 │
           └─ rokoko_mocap.py (解析+规范化)                           └──────────────┘
               └─ UDP 5013(右)/5015(左) (21×3 canonical keypoints)         ▲
                   └─ GeoRT rokoko_retarget_send_*.py (神经重定向)          │
                       └─ UDP 5014(右/单手)/5016(左) (16 维 qpos) ──────────┘
```

仿真侧 wrapper 实际监听的端口：

| 端口 | 内容 | 直接发送方 |
|---|---|---|
| `5012` | 手腕 6D 位姿（Vive Tracker，单臂 12 / 双臂 24 个 double） | `vive_bridge/send_vive_pose.py` |
| `5014` | 右手 / 单手 16 维 Allegro 关节角 `qpos` | GeoRT `rokoko_retarget_send_right.py` |
| `5016` | 左手 16 维 `qpos`（双臂任务） | GeoRT `rokoko_retarget_send_left.py` |

> 中间端口 `14044`（Rokoko Studio → rokoko_mocap）、`5013/5015`（rokoko_mocap → GeoRT）只在桥接进程之间用，**不进仿真**。

### 3.1 手腕：Vive Tracker 怎么连、原始格式是什么（端口 5012）

**硬件连接**（`send_vive_pose.py`）：
- 依赖 **`openvr`（SteamVR）**：`openvr.init(VRApplication_Other)` → `vr_system = openvr.VRSystem()`。
- `discover_devices()` 枚举所有 OpenVR 设备，`select_device()` 按 `--device-index` / `--serial-contains` / 设备类（默认挑第一个 `tracker`）选定要用的 tracker。
- 每帧 `getDeviceToAbsoluteTrackingPose(universe, 0, maxCount)` 读位姿；取 `pose.mDeviceToAbsoluteTracking`（一个 **OpenVR `HmdMatrix34_t`**），并检查 `bPoseIsValid`。默认 90 Hz。

**硬件原始格式**：OpenVR 的 `HmdMatrix34_t` = **3×4 行优先矩阵 `[R(3×3) | t(3×1)]`**，是 tracker 在 absolute tracking 坐标系下的位姿；`_matrix34_to_numpy()` 把它转成 `np.float64` 的 `(3,4)`。

**发出格式**：`pose.astype(np.float64).tobytes()` → **12 个 double = 96 字节**。双臂（`--two-trackers`）把主、副 tracker 各 12 个拼成 **24 个 double = 192 字节**一个包。

**仿真侧解析**（`_recv_vive_loop`）：
```python
pose = np.frombuffer(data, dtype=np.float64, count=12).reshape(3, 4)
transform = np.eye(4); transform[:3, :] = pose      # 补成 4×4 齐次矩阵
self.latest_tracker = transform
```
双臂 `DualArm...` 按字节长度判断是 24 还是 12 个 double，拆成 `latest_tracker_right/left`。

> **要换掉 Vive**：只要让新设备(任意 6-DoF 跟踪器/IMU/光学动捕)也按"3×4 位姿矩阵 → 12 个 float64 → UDP 5012"的约定发包即可，仿真侧 `_recv_vive_loop` 完全不用改。即重写一个等价于 `send_vive_pose.py` 的发送脚本，把你设备的位姿整理成同样的 3×4（行优先 R|t）字节流。

### 3.2 手指：Rokoko 手套怎么连、原始格式是什么（端口 5014 / 5016）

这一路有 **3 段**，"人手→机器手"的形态映射（retargeting）在 GeoRT 完成，**仿真侧收到的已经是 Allegro 关节角**。

**① Rokoko Studio → UDP 14044（原始手套数据）**
- Rokoko 手套数据进 **Rokoko Studio** 软件，由它对外 UDP 推流（默认端口 14044）。
- `rokoko/common.py` 的 `RokokoReceiver` 接收并 `parse_rokoko_packet`：**包体是 JSON（可选 LZ4 压缩）**，结构 `msg["scene"]["actors"][0]["body"]` → 各关节名 → `position:{x,y,z}`。
- `extract_hand_positions()` 按 21 个手部关节名（`*Hand` + 拇指/食指/中指/无名指/小指各 4 节 = **21 个关键点**）取出 → `(21, 3)` 世界坐标数组（**这就是 Rokoko 的硬件原始格式**）。
- `hand_to_canonical()` 把 21 点变换到以腕为原点、由手指几何定轴的**规范手系**，输出 `(21,3)`。

**② rokoko_mocap.py → UDP 5013(右)/5015(左)**
`rokoko_mocap.py` 把 canonical `(21,3)` 关键点 `tobytes()`（float32）转发到 5013/5015。

**③ GeoRT 重定向 → UDP 5014/5016（16 维 qpos）**
`rokoko_retarget_send_right.py`：
- 收 `(21,3)` 关键点；
- `model = load_model(ckpt_tag)`（默认 `handson_right_default`）—— **GeoRT 几何重定向模型**（每根手指一组 FK/IK MLP，按手的关节分组，如 Allegro 的 `[[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]`）；
- `qpos = model.forward(keypoints)` → **机器手关节角**；
- `qpos.astype(np.float64).tobytes()` 发到 5014/5016。

**仿真侧解析**（`_recv_hand_loop`），按 16 维截断/补零：
```python
hand_angles = np.frombuffer(data, dtype=np.float64)
parsed = np.zeros(16); parsed[:min(16, hand_angles.size)] = hand_angles[:16]
self.latest_allegro_angles = parsed
```

> **要换掉 Rokoko / 换手**：仿真侧只认"16 维 qpos → UDP 5014/5016"。两种改法：
> 1. 换数据手套但仍用 GeoRT：让新手套也吐 `(21,3)` 关键点到 5013/5015 即可（重写 ① ②）。
> 2. 换灵巧手（关节数变了）：需要**重新训练/导出 GeoRT 模型**（新手的 `config.json` + 关节分组 + checkpoint，见 GeoRT README 的采集→训练流程），让 `model.forward` 输出新手的关节维度；同时仿真侧 `parsed = np.zeros(16)` 的 16 要改成新手关节数（见 §7）。

### 3.3 接收方式：后台线程 + 锁

wrapper 在构造时各起一个 daemon 线程 `_recv_vive_loop` / `_recv_hand_loop` 持续 `recvfrom`，把最新值存进 `latest_tracker` / `latest_allegro_angles`，主控制循环再加锁取用 —— 网络接收与仿真步进解耦，避免阻塞物理。单臂开 2 个 socket（5012、5014）；双臂开 3 个（5012、5014、5016），其中**两只手腕共用 5012**（一个包里拼 24 个 double），**两只手分用 5014/5016**。

### 3.4 键盘控制（GLFW 回调）

**怎么挂上的**：`_ensure_key_callback()` 在每次 `reset/step` 时调用 `_maybe_set_key_callback`，从 MuJoCo viewer 取 GLFW `window` 并 `glfw.set_key_callback(window, self.glfw_on_key)`。**只有 human 渲染（有 GLFW 窗口）时才生效**；headless（`rgb_array`/策略模式）下没有窗口，键盘回调不挂。

**两个按键**（`glfw_on_key`）：

- **`;`（分号）= 开/关遥操介入（intervention）**：这是"踩离合"式的总开关。
  - 按下→开启：先 `sleep 2 秒`（`INTERVENTION_START_DELAY_SECONDS`，给操作者把手摆成与仿真手对齐的姿势），然后**快照基准**：`tracker_start_world = latest_tracker`（当前 Vive 位姿作为相对运动原点）、`ee_start = get_end_effector_pose_matrix()`（当前末端位姿）。之后 `_vive_action()` 才用 `inv(start)@now` 算相对增量驱动机械臂。
  - 再按→关闭：清空 `tracker_start_world/ee_start`，机械臂转为 `_hold_pose_action()` 原地保持，tracker 怎么动都不影响仿真。
  - **用途**：分段采集——只在"按下 `;`"期间记录有效遥操；可随时松开重新对齐手部基准，避免漂移/误操作进入数据。
- **`r` = 丢弃当前轨迹并重置**：置 `reset_trigger=True` → 下一次 `step()` 在 `info["manual_reset"]=True` → 录制脚本 [`record_demos_zarr.py`](../scripts/record_demos_zarr.py) 看到后**丢弃当前 trajectory 并 `env.reset()`**（采集到一半发现做坏了就按 `r` 重来）。
  - 录制脚本还另挂了一个 OpenCV 手腕预览窗的 `r` 键（`cv2.waitKey`）做同样的重置。

> **可扩展**：要加"标记成功/保存/切相机/微调 pose_scale"等热键，只需在 `glfw_on_key` 里加 `key == glfw.KEY_*` 分支并置对应标志位，主循环/录制脚本读取该标志即可——回调本身只负责"置状态位"，不直接操作仿真。

---

## 4. Simulator 如何用信号驱动机械臂位姿与手部形态

核心在 [`SingleArmViveHandTeleopWrapper`](../handson/handson/tasks/sim_teleop.py)（一个 `gym.ActionWrapper`）。它把外部 UDP 信号转换成一个统一的 **action 向量**，再交给底层环境 `step()` 执行。

单臂 action 布局（23 维）：

```
[ pos(3), quat_wxyz(4), allegro_joints(16) ]
   └── 机械臂末端目标位姿 ──┘   └── 灵巧手 16 关节目标 ──┘
```

### 4.1 机械臂位姿：相对增量 → mocap 目标 → 操作空间力矩控制

**这是关键设计：tracker 不是绝对映射，而是"相对增量"映射。**

1. 按下 `;` 时记录基准：tracker 起点 `tracker_start_world` 和末端起点 `ee_start`（4×4 矩阵）。
2. 每步计算 tracker 相对位移：
   ```python
   tracker_delta = inv(tracker_start_world) @ tracker_now   # 自基准以来的相对变换
   tracker_delta[:3, 3] *= pose_scale                       # 平移放大 (water_plant 用 1.5)
   target_pose = ee_start @ tracker_delta                   # 叠加到末端起点上 = 目标位姿
   ```
3. 转成 `[pos(3), quat_wxyz(4)]` 作为 action 前 7 维。
   - 未介入时则用 `_hold_pose_action()` 让末端"保持当前位姿"不动。

底层 `PandaWaterPlantGymEnv.step()` 拿到这 7 维后：
```python
self._data.mocap_pos[0]  = xyz          # 写入 mocap 目标体的位置
self._data.mocap_quat[0] = wxyz_quat    # 和姿态
```
然后在每个物理子步用**操作空间控制器 `opspace`** 把"当前末端"驱动向"mocap 目标"：

```python
tau = opspace(model, data, site_id=attachment_site,
              dof_ids=panda_dof_ids,
              pos=mocap_pos, ori=mocap_quat,
              joint=_PANDA_HOME,          # 零空间偏好姿态
              pos_gains=(400,400,400), damping_ratio=4, gravity_comp=True)
self._data.ctrl[panda_ctrl_ids] = tau   # 力矩驱动 7 个关节
mujoco.mj_step(model, data)
```

`opspace` 内部做的是标准**操作空间(任务空间)逆动力学**：
- 用雅可比 `mj_jacSite` 把末端位置/姿态误差经 PD 转成任务空间加速度；
- 用任务空间惯量 `Mx = (J M⁻¹ Jᵀ)⁻¹` 投影成关节力矩 `tau = Jᵀ Mx ddx`；
- 在**零空间**里附加一个把关节拉向 `_PANDA_HOME` 的次级目标（避免奇异/漂移）；
- 加重力补偿 `qfrc_bias`。

> 所以机械臂是**力矩控制**（动力学真实），mocap body 只是给控制器当"目标锚点"，并不直接 weld 到手上。

### 4.2 手部形态：关节角直接作为位置执行器目标

手的处理简单直接 —— **收到的 16 维 Allegro 关节角就是位置执行器的目标**：

```python
def _hand_action(self, action):
    if self.intervened:
        return self.latest_allegro_angles.copy()   # 直接用 UDP 收到的 16 维 qpos
    ...
```
拼进 action 的第 7~23 维，底层 `step()`：
```python
allegro_angles = action[7:7+16]
self._data.ctrl[self._allegro_ctrl_ids] = allegro_angles   # 16 个 position 执行器
```
Allegro 的执行器在机器人 XML 里是 `<position kp="2">` 类型，带各指关节的 `ctrlrange`（如 `tha0` 的 `0.263~1.396`）。MuJoCo 的位置伺服会自动把每个关节驱动到目标角并受限位约束。

**因此"信号→手形态"的对应关系是：** 人手关键点 →(GeoRT 重定向)→ Allegro 16 关节角 →(UDP)→ wrapper →(action[7:23])→ position 执行器目标 → MuJoCo 伺服驱动手指到该形态。人手与机器手的运动学差异在 **GeoRT 重定向**阶段被吸收，仿真侧只是忠实地伺服跟踪。

### 4.3 双臂

`DualArmViveHandTeleopWrapper` 逻辑相同，只是返回 `{"right": [...], "left": [...]}` 两套 `[pose7, hand16]`，分别用各自的基准做相对增量、各自伺服。

### 4.4 一个时间步的完整节奏

`step()` 里：写 mocap → 循环 `_n_substeps`(=10) 次 `{opspace 算力矩 + 写手指 ctrl + mj_step}` → 更新交互逻辑（扳机/喷雾锥）→ 计算观测 → 判定成功 → `time.sleep` 把节奏卡到 `hz`(=30Hz)。

---

## 5. 怎样保存输出

录制由 [`scripts/record_demos_zarr.py`](../scripts/record_demos_zarr.py) 驱动，落盘由
[`ZarrEpisodeStore`](../handson/handson/data/episode_store.py) +
[`Mp4VideoWriter`](../handson/handson/data/video_writer.py) 完成。

### 5.1 录制循环

```python
config = CONFIG_MAPPING[task_id]()
env = config.get_environment(render_mode=..., randomize=...)   # 内含 teleop wrapper
obs, info = env.reset()
while success_count < success_needed:
    next_obs, rew, done, _, info = env.step(np.zeros(action_space))  # 动作由 wrapper 从UDP注入
    actions = info["intervene_action"]                  # 真正执行的遥操动作
    trajectory.append({observations: obs, actions: ..., infos: info})
    if done and info["succeed"]:                        # 只保留成功 episode
        _write_demo_zarr_and_videos(trajectory, ...)
    obs = next_obs
```
注意：传给 `env.step` 的是占位零动作，真正的动作是 teleop wrapper 在内部从 UDP 信号组装并通过 `info["intervene_action"]` 回传记录的 —— **只有成功的轨迹才会被写盘**。

### 5.2 目录结构

每条成功 demo 生成一个带时间戳的目录：

```
<out_dir>/<exp_name>_demo_<index>_<timestamp>/
  replay.zarr/                 # 低维数据（action / state / timestamp / action_rotvec）
  videos/<camera_key>.mp4      # 每路 RGB 相机一个视频（如 wrist.mp4, front/random_camera.mp4）
  videos/<camera_key>_depth.npz / _depth.mp4   # --save_depth 时额外保存深度
```

### 5.3 Zarr 低维数据（time-major）

`ZarrEpisodeStore` 是 append-only、时间为主维（time-major）的存储。一个 episode 写入的字段：

| 字段 | 含义 |
|---|---|
| `action` | 实际执行的动作（四元数姿态表示）。单臂 23 维，双臂 46 维 |
| `action_rotvec` | 把姿态四元数转成旋转向量后的动作。单臂 22 维，双臂 44 维（OpenPI 训练用的目标） |
| `state` | 来自 `obs["state"]` 的本体感知 + 特权状态（TCP 位姿、手关节、物体位姿、桌高…） |
| `timestamp` | 按 `data_fps` 生成的每步时间戳 |

实现要点（[`episode_store.py`](../handson/handson/data/episode_store.py)）：
- `append_episode()` 校验所有字段时间长度一致；
- 用 `meta/episode_ends` 记录每个 episode 的结束下标（多 episode 拼成一条长流）；
- 自动按 ~2MB 选 chunk，磁盘压缩用 `Blosc(zstd, bitshuffle)`。

> action 的两种布局与转换：录制时双臂记成 `[r_pose7, r_hand16, l_pose7, l_hand16]`；策略模式环境期望 `[r_pose7, l_pose7, r_hand16, l_hand16]`，由 OpenPI client 自动重排。`convert_action_quat_to_rotvec()` 负责 quat(wxyz)→rotvec 的转换以产出 `action_rotvec`。

### 5.4 视频

对轨迹里发现的每个图像键（`obs` 中 `ndim>=3 且最后一维=3` 的项），用 `Mp4VideoWriter.create_h264()` 逐帧写 H.264 MP4（输入 RGB，CRF 21）。图像本身由 MuJoCo 离屏渲染器 `MujocoRenderer` 渲染 `wrist`（腕部相机）和第三人称相机得到。

---

## 6. 灵巧手 Allegro：在 XML 与代码里如何初始化，以及如何替换

机械臂(Panda)和灵巧手(Allegro)合在 [`xmls/panda_allegro_copy.xml`](../handson/handson/sim/envs/xmls/panda_allegro_copy.xml)（被各任务场景 `include`）。要换手，需要同时改 **XML 模型** 和 **环境代码里写死的名字/维度**两处。

### 6.1 XML 里 Allegro 做了什么（4 块）

1. **`<default class="allegro_right">`（默认/限位）**：定义所有手指 geom 的 visual/collision 类、`<position kp="2">` 伺服增益，以及 8 个关节类的**关节限位与执行器 ctrlrange**：`base(-0.47~0.47)`、`proximal`、`medial`、`distal`、`thumb_base/proximal/medial/distal`。换手时这些限位要换成新手的。
2. **`<asset>` 网格**：`<mesh file="wonik_allegro/assets/base_link.stl"/>` 等 11 个 STL（手掌 + 各指节）。网格文件在 [`xmls/wonik_allegro/assets/`](../handson/handson/sim/envs/xmls/wonik_allegro)。
3. **`<worldbody>` 手体树**：手挂在机械臂法兰的 `attachment_site` 下——`allegro_attachment > allegro_palm(pos/quat 对齐到法兰) > ff_base > ff_proximal > ... > th_distal`。每个指节一个 `<body>` + 一个 `<joint name="ffj0/.../thj3">`（共 **16 个关节**）+ visual/collision geom。
4. **`<actuator>` 与 `<sensor>`**：16 个 `<position name="ffa0..tha3" joint="ffj0..thj3">` 位置伺服执行器；16 个 `<jointpos name="allegro_right/ffj0_pos..thj3_pos">` 关节角传感器。

> 命名约定：关节 `ffj0-3 / mfj0-3 / rfj0-3 / thj0-3`（食/中/无名/拇指 × 4 节），执行器 `ffa0-3...tha3`，传感器 `allegro_right/<joint>_pos`。环境代码就是靠这些名字定位的。

### 6.2 代码里 Allegro 做了什么（以 `panda_water_plant_env.py` 为例）

环境类把上面的名字硬编码成几组常量与缓存：
- 常量：`_ALLEGRO_JOINT_NAMES`(16)、`_ALLEGRO_ACTUATOR_NAMES`(16)、`_ALLEGRO_SENSOR_NAMES`(16)、`_ALLEGRO_HOME`(16 维初始关节角)、`_N_ALLEGRO = 16`。
- `__init__` 缓存 id：`_allegro_dof_ids`（各关节 qpos 地址）、`_allegro_ctrl_ids`（各执行器 id）。
- `action_space` = `7 + _N_ALLEGRO`（臂 7 + 手 16）。
- `reset()`：`data.qpos[_allegro_dof_ids] = _ALLEGRO_HOME` 复位手。
- `step()`：`allegro_angles = action[7:7+_N_ALLEGRO]`，`data.ctrl[_allegro_ctrl_ids] = allegro_angles` —— **把 16 维关节角直接写进位置执行器**（见 §4.2）。
- `_compute_observation()`：`gripper_pose = [data.sensor(name).data for name in _ALLEGRO_SENSOR_NAMES]`（16 维手关节观测）。

遥操侧 [`sim_teleop.py`](../handson/handson/tasks/sim_teleop.py) 也有 `np.zeros(16)` 这类**写死 16** 的地方（`latest_allegro_angles`）。

### 6.3 怎么替换成另一款灵巧手

按"模型 → 名字 → 维度 → 重定向"四步：

1. **换 XML 模型**：用新手的 MuJoCo 模型替换 `panda_allegro_copy.xml` 里的 allegro 部分——
   - 替换 `<asset>` 网格、`<default>` 关节限位、`<worldbody>` 手体树（把手 palm 挂到 `attachment_site` 下并调 `pos/quat` 对齐）、`<actuator>`（每关节一个 position 执行器）、`<sensor>`（每关节一个 jointpos）。
   - 若新手关节数 ≠ 16，关节/执行器/传感器数量随之变化。
2. **改代码里的名字常量**：把 `_ALLEGRO_JOINT_NAMES / _ALLEGRO_ACTUATOR_NAMES / _ALLEGRO_SENSOR_NAMES / _ALLEGRO_HOME` 换成新手的关节/执行器/传感器名与初始姿态（**这些字符串必须和新 XML 完全一致**），否则 `model.joint(name)` / `data.sensor(name)` 会找不到而报错。
3. **改维度 `_N_ALLEGRO`**：设为新手关节数 N。这会自动联动 `action_space=7+N`、`step()` 切片 `action[7:7+N]`、观测维度等。同时把 `sim_teleop.py` 里写死的 `16`/`np.zeros(16)` 改成 N（单臂 state 23=7+16、双臂 46 等下游维度，以及 OpenPI/训练 config 的 action_dim/state_dim 也要相应改）。
4. **重做手部重定向（GeoRT）**：GeoRT 模型是**针对具体手训练**的（`config.json` 关节限位 + 关节分组 + checkpoint）。换手要按 GeoRT README 流程：用 Rokoko 采人手数据 → 配新手的 config（关节分组/限位）→ 训练/导出新 checkpoint → `rokoko_retarget_send_*.py --ckpt_tag <新手>`，让 `model.forward` 输出新手维度的 qpos 发到 5014/5016。

> 一句话：**XML 决定"手长什么样、有哪些关节/执行器/传感器"，代码常量决定"用哪些名字、多少维"，GeoRT 决定"人手关键点怎么映射到新手关节角"**——三者的关节命名与维度必须对齐。最省事的替换路径是先拿到新手的官方 MuJoCo XML，统一好关节命名，再在环境常量里照抄这些名字并改 `_N_ALLEGRO`，最后重训 GeoRT。

---

## 7. 一句话总结

HandsOn 用 **MuJoCo XML 模板化地搭场景**（房间模板 + `include` 资产），**靠 joint/碰撞 geom 让物体可交互**；遥操硬件把 **手腕 6D 位姿（5012，3×4 float64 矩阵）** 与 **GeoRT 重定向后的 16 维手关节角（5014/5016 float64）** 经 UDP 送入；wrapper 把手腕做**相对增量 → mocap 目标 → opspace 操作空间力矩控制**驱动机械臂，把手关节角**直接作为 position 执行器目标**伺服灵巧手；最后只把**成功轨迹**以 **Zarr（低维 action/state/timestamp）+ MP4（多相机）** 的形式落盘，供 OpenPI π0.5 训练与回放。
