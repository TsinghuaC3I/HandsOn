# EVAL-Flex03 枕头紧实度复测

上一版能抱起，但用户仍报告松垮。本轮保持几何、重量、摩擦和控制不变，材料保持 0.4 MPa、泊松比 .33、密度 55 kg/m³。提高 G01-5 的软体求解请求与场景上限，从 128 次增至 255 次，减少离散求解造成的额外软化。材料数值是仿真标定值，不是厂家实测参数。

曾测试仅把材料加硬至 1.2 MPa：按压拉伸峰值 1.770，抱起全过程出现 3 个翻转单元，整体悬持变形几乎不变，因此拒绝发布该候选。当前结果没有依赖继续加大材料模量。

同一原生双手动作、首次初始化、三个种子各 10 秒静置、按压与向外拉伸、悬持 3 秒、放回松手均在独占 GPU 的 Isaac Sim 5.1 / IsaacLab 2.3.2 中完成。测试读取实际资产材料，无临时参数覆盖。

| 求解次数 | 按压／拉伸时局部最大伸长比 | 抱起时局部最大伸长比 | 悬持末帧整体形变 RMS | 悬持最低离桌 |
| --- | ---: | ---: | ---: | ---: |
| 128 次 | 1.619 | 1.725 | 60.6 mm | 5.8 cm |
| 255 次 | 1.498 | 1.530 | 17.3 mm | 14.6 cm |

伸长比是局部四面体应变，不是枕头整体长度；整体形变 RMS 去除了刚体平移和旋转，包含弯曲。按压仍产生 10.9 mm 局部凹陷并触发增摩，原生双手承重通过，全程 0 个翻转单元。松手 3 秒后整体残余形变 RMS 为 2.77 mm。

38 项相关代码检查及 10 项注册检查通过。本轮诊断记录的控制循环中位数从 6.9 Hz 降至 4.0 Hz；这些是同类诊断程序的耗时读数，不代表已验收正式遥操作吞吐。 更高求解精度增加了计算耗时；记录中的 120 Hz 是仿真时间步，不能宣称达到实时 120 Hz。完整装箱、任意抓姿和正式物理回放未验证；人工手感仍需操作者复核。

羽毛填充并不能自动解决拉长：乳胶更偏响应式支撑，羽毛／羽绒更偏蓬松柔软，见 [Saatva 官方对比](https://www.saatva.com/blog/product-comparisons/latex-vs-down-chamber/)。当前继续使用单体体积 FEM 的等效枕头，没有新增羽毛颗粒或独立布套。材料标定和网格收敛依据见 [NVIDIA 软体文档](https://nvidia-omniverse.github.io/PhysX/physx/5.3.1/docs/SoftBodies.html)。

USD SHA-256：`33bc3580d714acb131e65ad2fc2279d06bba2a5a5ac1a140928bf1eacfff5646`。

---

# Historical 400 kPa / 128-iteration calibration

# G01-5 latex-foam reference and physical validation — 2026-09-16

The reported problem was localized stretching resembling a loose fabric bag.
The pillow remains one FEM volume with an embedded visual cover; it has no separate freely stretching cloth, rigid core, attachment or interaction-time node clamp.

## Reference and selected equivalent material

- [Talatex latex pillow](https://www.talatex.com/products/premium-latex-pillow?variant=40930325758049): densities of 45, 55 and 65 kg/m³; its ILD listing lacks a complete protocol and was not converted to a modulus.
- [Dorsal Vitalay Tranquillo](https://www.dorsal.it/en/pillows/vitalay-tranquillo): density above 60 kg/m³ for a 65 × 40 × 13 cm pillow. Our larger 80 × 42.22 × 22 cm geometry is retained.
- [Roberts and Garboczi, 2002, Fig. 12](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=860256): open-cell foam comparison including rubber-latex Poisson-ratio measurements. ν=.33 is a compressible-foam approximation, not a commercial-pillow fit.
- [Chen et al., 2022](https://journals.sagepub.com/doi/10.1177/00405175221086896): latex compression, shear and tension tested separately. Its unavailable full parameter table is not claimed as the source of our modulus.
- [NVIDIA solver convergence](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/deformables/deformable_bodies.html#deformable-body-component-parameters): insufficient iterations can cause artificial softness. G01-5 now honors its asset's 128-iteration request instead of the former scene cap of 64.
- [NVIDIA 107.3 material limitations](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/deformables_beta/deformable_beta.html#material-limitations): static friction is unsupported for deformables; live dynamic-friction readback is the grip evidence.

Selected parameters: E=400 kPa, ν=.33, density 55 kg/m³, elasticity damping 0.18.
E is simulator calibration of an equivalent core-and-cover continuum, not a measured commercial-latex modulus or an ILD conversion. Mass is derived from the actual 0.0510563 m³ physical mesh: 2.80810 kg (formerly 0.8 kg, about 15.67 kg/m³). Geometry remains 22 cm thick.
The 6 mm local indentation trigger engages dynamic friction 1.45; unloading restores 0.65. Static friction remains authored but is not supported by this runtime.

The heavier, density-corrected pillow also exposed the 35 mm measured-palm target-offset cap: under load the hands lagged while the pillow stayed on the table. G01-5 alone now allows 100 mm of drive position error. This is a bounded force-preload allowance, not a speed multiplier or a pose setter. Shared joint slew, target-jump, workspace and orientation limits remain active. G01-3/G01-4 retain 35 mm.

## Exact authored-asset verification

Isaac Sim 5.1 / IsaacLab 2.3.2 ran alone on the RTX 4090, with no material or controller overrides. First reset passed. Actual robot base, q/qdot and palm states were read back; seeds 0, 1, 2 each passed 1200 zero-action physics steps (10 simulated seconds at 120 Hz). Robot/pillow states and thresholds are recorded in `validation_summary.json`.

Both native Wuji hands pressed 40 mm beyond contact, pulled outward 60 mm each, returned, released and recovered for 3 s.

| Configuration | Peak local principal stretch | Peak inverted cells | Residual maximum stretch |
| --- | ---: | ---: | ---: |
| Original | 2.32994 | 3 | 1.07464 |
| Solver cap only | 2.05921 | 0 | 1.04747 |
| 200kPa authored candidate | 1.77415 | 0 | 1.02354 |
| 400kPa authored asset + loaded control | 1.61943 | 0 | 1.02362 |

These are local tetrahedron deformation-gradient singular values, not whole-pillow length changes. The final peak 95th percentile is 1.05046. Material parameters and control changed together; the table does not isolate each contribution. The old 200 kPa asset row used rendering; other rows can differ in contact timing.

Native carry: palms inward/fingers forward, palm height 0.965 m, squeeze half-span 0.38 m, per-finger targets [.65, 0, .7, .5], lift target +320 mm at 36.7 mm/s, hold 3 s, lower at 110 mm/s and release. No pillow state writes or attachments. Minimum held collision-surface gap=0.057667m; minimum centroid rise=0.129003m. Peak inverted cells=0; unexpected terminal=False. A 22 cm lift-target travel had left the sagging middle on the table and was correctly recorded as a failure; thresholds were not relaxed.

Earlier 200 kPa upper-side lifts slipped and inverted cells. 400 kPa alone did not solve this. Lower side contact avoided inversions but the 35 mm servo offset still failed to carry; the final combination is the verified result. Interrupted/concurrent/asserting runs are excluded.

Asset SHA-256: `61d82cc610073da6238852c52fb8bf6ad95c3f683e177241d3dc8f138d6b7a2e`.
Physics 120 Hz/control 30 Hz are simulated rates, not achieved wall-clock throughput. 41 focused checks pass; two unchanged UI catalog tests still expect removed fields/the old label. Shared-control regression:322 passed,27 failures reproduced on the unmodified baseline,10 skipped; no new failing cases. Registration audit:10/10. Full carton-placement recording, all possible grasps and formal deformable physical replay are not verified.
