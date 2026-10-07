# SkyHAD 3.0.0 验收记录

本页记录 2026-10-08 已执行的源码验收，并区分 2026-09-07 独立拆分时的历史结果。当前实现与验收位于隔离工作区 `E:/Code/.had-backups/skyhad-implementation-20261007/checkout`；机器可读报告保存在同级备份目录，不进入源码包。版本名称为 SkyHAD，Python 分发名为 `had-env`，导入名为 `had_env`。目录更名、最终安装包与发布操作仍需最后验收，本页没有把这些操作记为已完成。

## 来源、基线与历史产物

原项目基线为提交 `cf26fd4`，标记 `baseline-20261007`。开始改造时完整回归为 **76 passed、33 failed、2 skipped，23.82 秒**。失败包括过期物理 fixture、旧 7 列观测断言，以及已经移除的全局随机状态快照断言。该结果是改造前事实，不能作为当前源码的通过数。

回归参考的来源分别为：

- `tests/data/physics_reference.json` 的当前参考由**未修改的基线 `cf26fd4`**生成，协议标记为 `rebuild-calibrated-v3-r7-target-initialization`，显式使用 3D；旧拆分参考保留在 `historical_v1` 中。
- `tests/data/grouping_extraction_validation.json` 保留 2026-09-07 的 32 回合迁移比对记录和 4 条 `historical_golden_cases`。当前 4 条 `golden_cases` 依据**未修改的基线 `cf26fd4`、显式二维 grouping**更新；历史比对与当前 fixture 的来源有独立字段。统一工厂如今默认 3D，不改变这组显式二维回归的含义。
- `tests/data/open_score_reference.json` 是上游 Open-SCORE 原模块的字面输出，来源修订为 **`6f119e9904b9dfed9cb08b660aff078a2d82247d`**。采集时通过包桩排除无关训练器，物理部分来自与上游相同的原生 HAD 基线；预期值没有由新增兼容层生成。覆盖实体默认值、规模采样、观测与掩码、任务奖励、友军清空后的折叠、轨迹和 nv1 覆盖策略。

旧 `dist/`、`outputs/`、生成的 `build/` 已迁往 `E:/Code/.had-backups/skyhad-history-20261007`，旧发布包和实验结果在源码目录外保留。源目录的 9 个 `__pycache__`（排除 `.venv`、`.git`）也移入该备份下的 `bytecode/`。工作虚拟环境与 editable 元数据保留到最终安装验收。源码包版本统一来自 `had_env.__version__`，物理协议仍单独标记。

## 当前源码回归

使用 `E:/Code/Open_Score_HAD_Workbench/.venv/Scripts/python.exe`，在隔离工作区执行 pytest，并关闭字节码与 pytest 缓存。已有两轮完整记录：

| 记录 | passed | skipped | warnings | 用时 | 报告 |
|---|---:|---:|---:|---:|---|
| b1 完整回归 | 178 | 2 | 2 | 36.58 秒 | `b1-final.txt`、`b1-final.xml` |
| rc1 最终源码回归 | 182 | 2 | 2 | 36.83 秒 | `rc1-full.txt`、`rc1-full.xml` |

两项 skipped 都因为当前解释器没有安装可选依赖 Torch：`tests/test_workbench_rng.py` 的模块级导入，以及 `tests/test_workbench_integration.py` 的可选 Torch 集成测试。两条 warnings 是 PettingZoo 旧环境创建 API 的弃用提示。这里没有将未执行的 Torch 测试记为通过。

本轮覆盖原生 Parallel/MPE 接口、配置优先级、任务模式、分组、Open-SCORE 兼容、刚体动力学、工作台与记录分支。离散和连续动作、死亡智能体的中性动作、完整校验、终止/截断语义、随机流恢复和独立随机性均由当前回归测试检查。Open-SCORE 兼容仅支持二维粒子、离散 acceleration、damage 任务；UAV、3D 与连续模式明确拒绝。

另有独立接口验收 `api-acceptance.py` / `api-acceptance.json`：两种 UAV × acceleration/actuator/position 三种控制 × damage/survival 两种任务，共 **12 个组合**。在 2 红、2 蓝、1 目标、上限 10 步设置下，官方 `parallel_api_test(num_cycles=8)` 与 `parallel_seed_test(num_cycles=8)` 各 **12/12 通过**。MPE 的 57 维扁平观测符合其空间定义，且与相同 seed 的 Parallel 逐步观测、奖励及 state 精确一致，**12/12 通过**；该独立验收 **0 warnings，用时 75.312 秒**。57 维对应本次 5 个实体的 4×11 相对实体特征及 13 维自身刚体状态，不是所有规模的固定维数。

## 三维回合与定向接近

`scenario_validation.py` / `scenario_validation.json` 保存 **20 条实测记录**：三种模型 × 两种任务 × seeds 3/17/42 的 18 个完整 3D 回合，加两项 UAV 定向接近。完整回合采用 `KnownOpponentEnv`，4 红、4 蓝、2 目标、`max_steps=100`、reactive 对手、acceleration 控制、显式 `spatial_dim=3`；红方使用 `RulePolicy("rule", seed+1)`。

下表的 `steps` 为实际物理步数，`firsttarget` 为第一次目标受攻击的物理步，`D` 为目标累计损伤，`boundaryclips` 为每步各智能体被边界投影的坐标轴次数之和；它不是发生边界接触的回合数。`—` 表示该回合没有目标攻击。

| 模型 | 任务 | seed | steps | firsttarget | D | boundaryclips | 结束原因 |
|---|---|---:|---:|---:|---:|---:|---|
| particle | damage | 3 | 46 | 42 | 2 | 10 | blue_attackers_destroyed |
| particle | damage | 17 | 21 | — | 0 | 10 | blue_attackers_destroyed |
| particle | damage | 42 | 30 | 30 | 1 | 2 | blue_attackers_destroyed |
| particle | survival | 3 | 46 | 42 | 2 | 10 | target_destroyed |
| particle | survival | 17 | 21 | — | 0 | 10 | blue_attackers_destroyed |
| particle | survival | 42 | 30 | 30 | 1 | 2 | blue_attackers_destroyed |
| UAV_fixedwing | damage | 3 | 61 | 58 | 2 | 2 | blue_attackers_destroyed |
| UAV_fixedwing | damage | 17 | 100 | 32 | 1 | 43 | horizon |
| UAV_fixedwing | damage | 42 | 93 | 42 | 3 | 70 | blue_attackers_destroyed |
| UAV_fixedwing | survival | 3 | 61 | 58 | 2 | 2 | target_destroyed |
| UAV_fixedwing | survival | 17 | 100 | 32 | 1 | 43 | horizon |
| UAV_fixedwing | survival | 42 | 52 | 42 | 2 | 12 | target_destroyed |
| UAV_quadrotor | damage | 3 | 38 | 38 | 1 | 0 | blue_attackers_destroyed |
| UAV_quadrotor | damage | 17 | 21 | — | 0 | 0 | blue_attackers_destroyed |
| UAV_quadrotor | damage | 42 | 30 | 28 | 1 | 0 | blue_attackers_destroyed |
| UAV_quadrotor | survival | 3 | 38 | 38 | 1 | 0 | blue_attackers_destroyed |
| UAV_quadrotor | survival | 17 | 21 | — | 0 | 0 | blue_attackers_destroyed |
| UAV_quadrotor | survival | 42 | 30 | 28 | 1 | 0 | blue_attackers_destroyed |
| UAV_fixedwing（定向接近） | damage | 17 | 26 | 26 | 1 | 0 | blue_attackers_destroyed |
| UAV_quadrotor（定向接近） | damage | 17 | 18 | 18 | 1 | 0 | blue_attackers_destroyed |

**20/20 记录的 `finite=true`，且空间维数均为 3。**检查包含每个实体的位置、速度、HP，以及 UAV 完整刚体状态。14 条 UAV 记录的四元数最大范数误差不超过 **2.220446049250313×10⁻¹⁶**；particle 没有姿态四元数，记录中的零值只是该指标的默认值。

两个定向接近场景使用 1 红、1 蓝、1 目标，红方置为死亡，蓝方被明确指派到目标并采用 rush。固定翼初距 **840 m**、射程 **120 m**，以水平 **−30 m/s** 初速朝目标；四旋翼初距 **336 m**、射程 **48 m**，初速为零。分别在 **26/18 个物理步**造成 D=1，边界投影都为零。这两项验证受控初态下可以由控制与积分进入射程，不表示随机回合保证攻击成功。

damage 的目标 HP 不随攻击下降，D 独立累计；survival 的目标 HP 会下降。因此，表中同 seed 的两种任务可能在不同时间结束。固定翼 seed17 的 damage 到 100 步为采样截断，survival 到 100 步为任务规定的守方存活终止；不能仅凭 `end_reason=horizon` 判断 `terminated/truncated`。各接口的奖励、折叠与 bootstrap 字段见 [API.md](API.md) 与 [MODELING.md](MODELING.md)。

## 动力学专项验收与限制

UAV 使用 Newton–Euler 刚体状态与 RK4，默认积分子步 **0.01 s**。当前固定翼控制器按期望空速调度配平，重新取相应 trim 的姿态与舵面/油门基准；**18、24、30 m/s** 的水平平衡均由测试验证平动与角加速度范数小于 10⁻⁶。默认 30 m/s 配平连续积分 10 秒的位置为 `[300,0,100]`，测试容差 10⁻⁵ m。

固定翼位置控制从 `[0,0,100]`、初速 `[30,0,0]` 向 `[160,50,130]` 积分 **10 s**，比较 0.01/0.005 s 子步的实测末态差为：位置 **0.0011978 m**、速度 **0.0003365 m/s**、姿态角 **0.000634°**。当前测试允许的对应上界为 0.1 m、0.05 m/s、0.5°；这里区分实测差值与断言阈值。

四旋翼 velocity/acceleration 控制的速度参考上限为 **20 m/s**，position 外环速度参考上限为 **12 m/s**，二者分别配置。16 m/s 的水平零加速度回归在 2 s 后仍为 16 m/s、位移 32 m，没有被 position 上限截断。悬停输入每旋翼 .4，10 秒无人工运动测试通过；零推力自由落体 1 秒得到 z=95.095 m、vz=−9.81 m/s。还检查了 FLU 倾斜方向、差动旋翼力矩、总推力保留、控制器无状态重放，以及极端输入下的有限状态和四元数归一化。

控制器限制的是参考指令或执行器输入，不直接夹紧 UAV 的真实速度。极端满推力四旋翼可超过 20 m/s，已有回归明确检查这一点。场景边界则会投影位置并处理向外速度，是环境约束；固定翼完整回合的较高 `boundaryclips` 提醒使用者：这些轨迹包含边界投影，不能当作无边界的真实飞行轨迹。固定翼 position 通过航向、航迹角、姿态和推进控制飞向并可能穿越目标，不能悬停；四旋翼模型采用理想推力，没有阻力或电机滞后。本轮验收证明实现的数值与接口性质，没有证明真实机体参数辨识、气动包线或算法优劣。

## 2026-09-07 独立拆分的历史事实

独立项目最初从工作台提交 **`786d554fa4215ae6e7be9481770408b3f3328748`**提取必要环境代码，建立独立 Git 历史。原工作台为 103 个 Python 文件、21781 行；当时独立环境为 48 个 Python 文件、约 7260 行。研究训练器、冻结模型、检查点、实验账本与旧输出没有进入新历史。拆分前完整历史保存在 `E:/Code/.had-backups/workbench-before-standalone-20260907/workbench.bundle`，旧子目录与文档也在该备份位置。

当时原研究目录只读复核的 HEAD 为 `45b627627323a7b567a7331f2f4dcf76e07c8dfc`。该 HEAD 是历史拆分时的记录；当前 Open-SCORE 兼容基准使用上文的 `6f119e…` 修订。

当时保存了 3 个场景、17 个真实物理步的独立 fixture，以及 4v4/8v8/16v8/32v32 × rule/grand/static_rule/random × seeds3/72 的 **32 个完整分组回合**逐步精确比对。历史干净环境回归为 **109 passed、2 skipped，16.27 秒**；另外在备份隔离解释器执行的可选 Torch 测试 **9 项通过**。这些数值属于 2026-09-07，不能替代本轮 Torch 跳过项或源码验收。

历史验收还包括 Qt 离屏真实控件、事件循环与 spawn 工作进程，以及约 114 KiB wheel 的独立目录安装、Python `-I` 跨工作目录调用、精确分支、PNG/SVG/PDF/MP4、Pygame RGB 与 Qt 窗口。旧报告随 `outputs/` 迁入 history 备份；这里保留其来源事实，不宣称 SkyHAD 3.0.0 新 wheel 已通过相同安装流程。

## 复验与最终结果待填

在待验收源码目录使用所选解释器执行：

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
$env:SDL_VIDEODRIVER = 'dummy'
$env:SDL_AUDIODRIVER = 'dummy'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& 'E:/Code/Open_Score_HAD_Workbench/.venv/Scripts/python.exe' -B -m pytest -q -p no:cacheprovider
```

正常使用窗口时移除 Qt/SDL 离屏环境变量。场景与独立接口脚本及 JSON 位于 `E:/Code/.had-backups/skyhad-implementation-20261007/`，可核对表格与接口结果。

**最终验证待填写：**目录更名后的完整复测、SkyHAD 3.0.0 wheel 构建与独立安装、实际导入路径和入口命令验证、最终提交与推送结果，由主执行者在这些操作完成后填写。当前已确认的是 rc1 源码回归、20 条场景记录和 12 组合接口验收。
