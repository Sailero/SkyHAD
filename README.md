# SkyHAD

异构空中攻防环境，支持 particle、六自由度固定翼和四旋翼。红方保护多个目标，蓝方按原有自动打击规则进攻；提供生存防御、累计损伤两个任务，以及训练、分组决策和桌面调试接口。

新名称为 **SkyHAD**，版本 **3.0.0**，仓库 [Sailero/SkyHAD](https://github.com/Sailero/SkyHAD)。继续保留 Python 包 `had_env`、发行包 `had-env` 和原命令 `had-workbench`；新命令为 `skyhad`。

## 安装与使用

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[viewer,test]"
.\run_workbench.ps1
```

纯训练只需 `pip install -e .`，入口不会加载 Qt、Pygame 或 Torch。

```python
from make_env import make_env

env = make_env(config={"env_agent_type": "UAV_quadrotor",
                       "env_agent_action_type": "acceleration",
                       "task_mode": "damage"})
observations, infos = env.reset(seed=42)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    observations, rewards, terminated, truncated, infos = env.step(actions)
env.close()
```

| 选项 | 含义 |
|---|---|
| `env_agent_type` | `particle`（默认）、`UAV_fixedwing`、`UAV_quadrotor` |
| `env_agent_action_type` | `acceleration`（默认）、`actuator`、`position`；particle 无 actuator |
| `api` | `parallel`（默认）、`mpe`、`grouping`、`open_score` |
| `spatial_dim` | 统一工厂默认 3；显式 2 保留 particle 平面功能；UAV 只允许 3 |
| `task_mode` | `survival` 有限目标 HP；`damage` 保持目标 HP，按新增原始伤害计分 |
| `config` | 字典、JSON 文件或不可变 `EnvConfig`；显式关键字优先 |

三个运动模型共用碰撞、自动开火、打击后自毁、同时伤害和终止规则。位置命令提供制导，不瞬移；固定翼不会悬停。UAV 任务场景按参考速度缩放，质量、惯量和飞机尺寸不缩放。

```python
# Open_Score 原始 particle 二维 damage、十列实体特征和 ALMA 掩码。
env = make_env(api="open_score", profile="training", scale="mixed_le10", seed=0)
entities, masks = env.reset()
reward, done, info = env.step([0] * env.n_agents)
env.close()
```

直接调用 `KnownOpponentEnv` 保持 particle 二维默认。分组接口使用分配动作和加速度意图；actuator/position 可通过 Parallel、MPE 或 `FlightSession` 使用。

## 工作台

保留实时调试、暂停/单步、策略替换、决策树、XY/XZ 视图、目标初始化、记录、对比、分支、PNG/SVG/PDF/MP4 导出和配对评估。UAV 使用姿态驱动的飞机/四旋翼图标及四元数插值。历史记录可查看；继续执行要求记录的行为来源与当前版本匹配。

```powershell
skyhad record --policy rule --output outputs/rule.json.gz
skyhad view outputs/rule.json.gz
skyhad branch outputs/rule.json.gz --step 5 --policy grand --seed 99 --output outputs/branch.json.gz
skyhad export outputs/rule.json.gz outputs/trajectory.svg --step 10
skyhad evaluate examples/evaluation.json --output outputs/evaluation
```

模型、控制、空间维度及配置文件选项见 `skyhad live --help` 和 [API 文档](docs/API.md)。

## 完整文件结构与必要性

以下列出全部版本管理文件及各自作用。可选显示模块由无头入口延迟导入，源码不包含训练器、第三方项目副本、构建产物或实验输出。

```text
SkyHAD/
  .gitattributes  # 统一文本换行和跨平台来源指纹
  .gitignore  # 排除环境、缓存和实验输出
  README.md  # 项目说明、完整文件结构及恢复节点
  docs/
    API.md  # 工厂、动作、观测、兼容层与工作台协议
    MODELING.md  # 两个任务的 POMDP、动力学及控制模型
    VALIDATION.md  # 测试结果、历史基线和清理证据
  examples/
    evaluation.json  # 相同开局的配对评估
    grouping.py  # 分组记录示例
    quickstart.py  # 最小 Parallel 交互示例
  had_env/
    __init__.py  # 明确包边界和稳定的公开导入
    actions.py  # 合法动作与固定编号
    config.py  # 默认参数及不可变实例配置
    core/
      __init__.py  # 明确包边界和稳定的公开导入
      agents/
        __init__.py  # 明确包边界和稳定的公开导入
        attack.py  # 自动打击、自毁和几何奖励
        base.py  # 实体健康、原运动与状态同步
        disturb.py  # 扇区干扰与软杀伤
        scout.py  # 侦察扇区及辅助奖励
      config.py  # 默认参数及不可变实例配置
      dynamics/
        __init__.py  # 明确包边界和稳定的公开导入
        control.py  # 四元数、旋转和控制数学工具
        fixedwing.py  # 气动、螺旋桨、配平和固定翼制导
        quadrotor.py  # 旋翼力矩、控制及分配
        rigid_body.py  # Newton–Euler 方程和 RK4 积分
      env/
        __init__.py  # 明确包边界和稳定的公开导入
        env.py  # 出生、动作转换、观测与边界
        world.py  # 同步碰撞和同时伤害结算
      function/
        Function.py  # 距离、衰减与结算快照
        __init__.py  # 明确包边界和稳定的公开导入
      make_env.py  # 稳定的公开环境入口
      render/
        __init__.py  # 明确包边界和稳定的公开导入
        glyphs.py  # 姿态驱动的飞机与四旋翼矢量图标
        render.py  # 可选 Pygame 显示
      resources/
        target.png  # 存活目标图像
        target_dead.png  # 损毁目标图像
      version.py  # 物理协议及版本标识
    environment.py  # 环境协议、观测及生命周期
    factory.py  # 统一工厂与参数优先级
    grouping/
      __init__.py  # 明确包边界和稳定的公开导入
      actions.py  # 合法动作与固定编号
      adapter.py  # 共享物理世界、分配及精确分支
      domain.py  # 不可变实体与分组决策状态
      environment.py  # 环境协议、观测及生命周期
      opponents.py  # 三个已知蓝方分配规则
      policies.py  # 可替换的上层策略示例
      protocol.py  # 原评估协议、种子及回合定义
      rules.py  # 各接口对应的覆盖/导航执行规则
    open_score_compat/
      __init__.py  # 明确包边界和稳定的公开导入
      entity_env.py  # ALMA 实体环境完整接口
      features.py  # Open_Score 十列特征与掩码
      rules.py  # 各接口对应的覆盖/导航执行规则
      scales.py  # 上游规模池和独立回合种子
      wrapper.py  # 分配时钟、奖励、折叠和统计
    scenarios/
      __init__.py  # 明确包边界和稳定的公开导入
      defense.py  # 场景、初始化和奖励回调
      presets.py  # 按参考速度配置任务几何
    workbench/
      __init__.py  # 明确包边界和稳定的公开导入
      __main__.py  # 支持 python -m 启动
      agent_session.py  # 原生飞行策略接入与记录
      cli.py  # 记录、查看、分支、导出和评估命令
      evaluation.py  # 相同开局的配对评估
      export.py  # 科学图像、矢量图与视频导出
      gui.py  # Qt 控件与桌面交互
      identity.py  # 行为和显示来源指纹
      live.py  # 独立进程中的实时调试
      playback.py  # 回放采样及姿态 SLERP
      protocols.py  # 可序列化场景及接口语义标识
      recording.py  # JSON/GZ 保存及历史读入
      rng.py  # 策略随机状态隔离
      session.py  # 分组会话、策略注册和运行分支
      snapshot.py  # 严格验证的可移植分支快照
      views.py  # 战场、曲线、树及状态详情
  make_env.py  # 稳定的公开环境入口
  pyproject.toml  # 依赖、可选界面、打包与命令定义
  run_workbench.ps1  # 本地虚拟环境启动入口
  tests/
    data/
      grouping_extraction_validation.json  # 升级前完整分组回合和历史验证摘要
      open_score_reference.json  # 独立采集的上游兼容接口基准
      physics_reference.json  # 升级前 particle 轨迹及历史归档
    test_core_physics.py  # 粒子基线、伤害边界、同时结算及非法动作原子性
    test_core_pygame.py  # 显示缓存、资源生命周期、渲染隔离及 UAV 图标
    test_dynamics.py  # 六轴配平、悬停、自由落体、控制与数值收敛
    test_environment.py  # 官方 Parallel/seed 合约、固定槽及 MPE 语义
    test_grouping_standalone.py  # 完整粒子回合基线、分组执行及快照恢复
    test_open_score_compat.py  # 独立上游基准、十列实体、mask、shaping 和折叠
    test_scenarios.py  # 配置优先级、场景缩放、位置控制及交战可达性
    test_workbench_evaluation.py  # 配对开局、跨种子统计和 OOD 协议
    test_workbench_gui.py  # 真实 Qt 选择、对比、导出及模型相机范围
    test_workbench_integration.py  # CLI、各控制模式、录制分支和工作进程
    test_workbench_live_branch_gui.py  # 实时保存分支且不推进父会话
    test_workbench_live_gui.py  # 实时暂停、单步与进程退出
    test_workbench_optional.py  # 无头导入、可选 Torch 和策略随机状态隔离
    test_workbench_recording.py  # 新记录完整往返及旧记录真实信息导入
    test_workbench_rng.py  # 安装 Torch 时检查 CPU/CUDA 随机流恢复
    test_workbench_session.py  # 策略会话、独立规划分支与宏步暂停
    test_workbench_snapshot.py  # 可移植快照校验及死亡后刚体精确恢复
```

忽略的 `.venv/` 提供运行依赖。旧发行包、构建和实验输出归档于项目外 `E:/Code/.had-backups/skyhad-history-20261007/`。临时实现工作区在完成后清除。

## 验证和版本恢复

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q
```

详细数学模型见 [MODELING](docs/MODELING.md)，验证结果和历史基线差异见 [VALIDATION](docs/VALIDATION.md)。版本仅由 `had_env.__version__` 定义；物理协议标识独立管理。

| 标签 | 恢复节点 |
|---|---|
| `baseline-20261007` | 升级前基线，保留本地及远端历史 |
| `v3.0.0a1` | 文件清理和单一版本来源 |
| `v3.0.0a2` | Open_Score 兼容及公开配置接口 |
| `v3.0.0a3` | UAV 动力学及三种控制 |
| `v3.0.0b1` | 工作台、图标、回放和快照 |
| `v3.0.0rc1` | 文档、场景验证和完整测试 |
| `v3.0.0rc2` | 更名、安装及打包验证 |
| `v3.0.0` | 最终完整版本 |

例如 `git switch -c restore-skyhad v3.0.0a3` 可建立恢复分支，当前工作分支不受影响。
