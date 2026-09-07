# HAD-Env

独立的异构攻防博弈仿真环境。根目录 `make_env.py` 是统一入口，支持 MPE 风格的场景组织、
PettingZoo 同时动作接口，以及固定列表兼容接口。物理核心保留 HAD 的运动、碰撞、规则开火、伤害与自毁时序。

本项目不依赖 Open-SCORE、研究训练脚本或冻结模型。默认安装不需要 Torch、Pygame 或 Qt。
分组实验、完整回放和 Qt 科研工作台保留在独立子模块中。

## 安装

Python 3.10+：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
# 按需安装：Pygame 渲染，或完整科研界面/导出/测试
.\.venv\Scripts\python.exe -m pip install -e ".[render]"
.\.venv\Scripts\python.exe -m pip install -e ".[viewer,test]"
```

本机专用解释器为 `.venv\Scripts\python.exe`，与原研究环境分开。
也可以从其他项目安装本仓库：`python -m pip install -e E:\Code\Open_Score_HAD_Workbench`。

## 统一接口

在项目根目录或者安装后：

```python
from make_env import make_env  # 也支持 from had_env import make_env

env = make_env("defense", red_count=8, blue_count=8, max_cycles=100)
obs, infos = env.reset(seed=42)
try:
    while env.agents:
        actions = {name: env.action_space(name).sample() for name in env.agents}
        obs, rewards, terminated, truncated, infos = env.step(actions)
finally:
    env.close()
```

双方按同一时刻的观测提交一个动作字典，每次 `step` 推进一个原生物理步。公开实体名为
`red_0...`、`blue_0...`，`possible_agents` 固定；死亡实体返回最后一次转换后移出 `agents`。
`state()` 返回集中训练可用的全局状态，动作/观测空间在一个环境实例内固定。

| 参数/返回值 | 语义 |
|---|---|
| `red_count`、`blue_count` | 各队总人数，包含所有异构角色；默认各 4 |
| `red_scouts`、`red_disturbers` 等 | 从总人数中划分角色，剩余为 Attack；蓝方至少一名 Attack |
| `target_count` | 保护目标数，默认 2 |
| `continuous=False` | `Discrete(27)`：零加速度和 26 个方向；索引顺序保留原控制接口 |
| `continuous=True` | 每实体有限 `[-1,1]^3` 加速度向量，再乘原生加速度上限 |
| `render_mode` | `None`、`human`、`rgb_array`；图形依赖按需导入 |
| `max_cycles` | 原生飞行采样上限，默认 100；到时仅截断，不额外判胜 |
| 观测 | `(总实体数-1, 7)` 的原生相对位置/速度/alive 行；固定世界顺序去自身 |
| `infos[a]["observation_entities"]` | 观测每行对应的实体名，死亡槽位仍保留 |
| `rewards[a]` | 默认该队原生 `RealReward`；四维 `LatentReward` 同时保存在 info |
| `reward_weights` | 可显式设置四维潜在奖励权重；episode 分量已含 RealReward，不应再重复相加 |

全态势观测和规则开火保持原样，侦察角色不会在这个接口中自动引入新的隐藏观测。
精确的奖励、终局及物理协议说明见 [docs/API.md](docs/API.md)。

旧 MPE 的固定列表调用可以使用：

```python
env = make_env("defense", api="mpe", red_count=4, blue_count=4)
obs_n = env.reset(seed=42)
obs_n, reward_n, done_n, info = env.step([0] * env.n)
env.close()
```

此接口提供 `.n`、列表 `.action_space` / `.observation_space` 和展平观测；
离散动作接受整数或严格的 27 维 one-hot。已死亡的固定槽位在最后一次转换后返回零观测、零奖励和 `done=True`。
这是一层调用约定适配，不保证所有旧 MADDPG 代码无需调整即可运行。

## 科研工作台

```powershell
.\run_workbench.ps1
# 或
.\.venv\Scripts\python.exe -m had_env.workbench live
```

默认打开独立、暂停的 8v8 分组调试会话。支持播放/单物理步/决策边界/伤亡跳转、XY 战场与 XZ 高度、
缩放选择、编组和任务关系、真实事件、决策详情、两策略比较及分支。关闭窗口只停止它拥有的调试进程。

```powershell
had-workbench record --policy rule --output outputs/rule.json.gz
had-workbench record --policy grand --output outputs/grand.json.gz
had-workbench view outputs/rule.json.gz --compare outputs/grand.json.gz
had-workbench branch outputs/rule.json.gz --step 5 --policy grand --seed 99 --output outputs/branch.json.gz
had-workbench export outputs/rule.json.gz outputs/trajectory.svg --step 10
had-workbench export outputs/rule.json.gz outputs/replay.mp4
had-workbench evaluate examples/evaluation.json --output outputs/evaluation
```

决策步会受伤亡影响，用 `had-workbench inspect` 确认分支位置。新录制含真实物理帧和受约束的 JSON 快照；
旧 v4/v5 稀疏日志仍能只读打开。不同源码的录制可查看，但离线续演严格要求行为指纹匹配。
迁移前的旧录制不会被冒充为当前版本的可恢复快照。

PNG/SVG/PDF/MP4 由 CLI 从记录生成，并保存来源 sidecar；GUI 的“导出画面”保存当前战场布局。
分支和播放不会修改原轨迹。关闭实时工作进程有 2 秒保存退出期限，长策略/慢磁盘时末次 partial 可能未写完；
正式评估使用 `record` / `evaluate` 并等待正常完成。

算法通过标准 Parallel API 接入；上层分组算法也可使用
`had_env.workbench.session.SimulationSession` 和 `register_policy(name, factory)`。
学习模型由调用方提供，不随环境捆绑。使用 PyTorch 策略时，先导入 Torch；GPU 模型和 CUDA 上下文应在
创建 `PolicyAdapter` 前准备好，避免在第一次动作内初始化新的随机源。

## 精简后的结构

```text
make_env.py             统一入口
had_env/
  factory.py            场景与 API 选择
  environment.py        Parallel / MPE 调用适配
  scenarios/defense.py   场景构建、重置、观测和奖励回调
  core/                 HAD 原生物理与可选 Pygame 渲染
  grouping/             分组状态、规则对手、控制执行及快照
  workbench/            记录、评估、分支、Qt 与导出
examples/               最小调用与评估配置
tests/                  独立测试及旧物理基准
docs/                   API 与验收说明
```

输出、虚拟环境、缓存、训练检查点均不进 Git。原研究项目的训练方案、实验记录、模型和运行器已经移出本仓库。

## 验证与来源

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

测试包括 PettingZoo 官方 `parallel_api_test` / `parallel_seed_test`、原始物理黄金轨迹、
分组规则迁移对照、随机隔离、Qt 控件、真实 spawn 进程、记录分支和导出。
具体结果与版本来源见 [docs/VALIDATION.md](docs/VALIDATION.md)。

组织方式参考 [原 MPE 的 make_env/Scenario](https://github.com/openai/multiagent-particle-envs/blob/master/make_env.py)，
同时动作接口依据 [PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/)。
参考的是模块与接口组织，HAD 的攻防动力学仍来自本项目原实现。
