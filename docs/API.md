# SkyHAD 3.0.0 接口与协议

项目显示名称为 **SkyHAD**，Python 发行包保持 `had-env`，导入路径保持 `had_env`。历史命令 `had-workbench` 保留；`skyhad` 为新入口。软件版本、物理协议、任务协议是不同标识：运行时以 `info["core_version"]`、`info["physics_protocol"]` 和录制任务配置为准。数学建模与动力学来源见 [MODELING.md](MODELING.md)，实际验证范围见 [VALIDATION.md](VALIDATION.md)。

## 统一工厂与配置

```python
from had_env import make_env, parallel_env
from had_env.config import EnvConfig

env = make_env(
    "defense", api="parallel",
    config={"env_agent_type": "UAV_quadrotor", "spatial_dim": 3},
    env_agent_action_type="position", task_mode="damage",
    red_count=4, blue_count=4, target_count=2, max_cycles=100,
)
obs, infos = env.reset(seed=42)
try:
    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terminated, truncated, infos = env.step(actions)
finally:
    env.close()
```

`make_env(scenario_name="defense", *, config=None, **kwargs)` 接受配置字典、UTF-8 JSON 文件路径或不可变 `EnvConfig`。先读取 config，再由显式关键字覆盖同名值。`EnvConfig.to_dict()` 返回实际参数；`env.effective_config` 和 `world.task_info()["effective_config"]` 记录有效配置。`parallel_env(**kwargs)` 是 Parallel 的便捷入口。场景名为 defense，兼容 defense.py。

| api | 返回接口 | 动作与步长 |
|---|---|---|
| parallel，默认 | HADParallelEnv / PettingZoo Parallel | 同时提交红蓝双方动作；一次 step 为一个物理步 |
| mpe | MPEEnv 固定列表适配器 | 固定 roster 顺序；物理和奖励与 Parallel 相同 |
| grouping | KnownOpponentEnv | 红方提交 Grouping；事件/周期边界之间为一个宏步 |
| open_score | HADEntityEnv | 红方 9 类二维动作；固定蓝方上层/下层规则 |

统一工厂的原生与 grouping 默认 `env_agent_type="particle"`、`env_agent_action_type="acceleration"`、`spatial_dim=3`。直接构造 KnownOpponentEnv 则保留 particle 二维、UAV 三维的默认；Open-SCORE 兼容固定为二维。Parallel/MPE 默认 survival、红 4/蓝 4/目标 2、max_cycles=100；分组直接构造默认红 8/蓝 8/目标 2、max_steps=100。

| 参数 | 含义 |
|---|---|
| red_count / blue_count / target_count | 各队总人数、目标数；均为正整数 |
| red_scouts / blue_scouts、red_disturbers / blue_disturbers | 总人数中的侦察、干扰角色，其余为攻击者；蓝方至少一名攻击者 |
| env_agent_type | particle / UAV_fixedwing / UAV_quadrotor；UAV 要求三维 |
| env_agent_action_type | acceleration / actuator / position；particle 不支持 actuator |
| continuous | acceleration 是否用连续归一化向量；actuator/position 自动用 Box |
| world_bounds、target_region | x/y/z 三行 `[low,high]`，单位米 |
| scene_scale | 默认 particle 1、固定翼 .4、四旋翼 .16 |
| red_spawn_annulus、blue_spawn_x、spawn_altitude | 红方环带半径、蓝方 x 范围、三维出生高度范围 |
| plane_altitude | 二维固定高度；默认 `100*scene_scale`，位于高度边界内 |
| fire_range | 自动开火阈值，正值且不超过满伤半径 |
| task_mode、target_health | survival/damage；目标默认 HP=2，显式值为正有限数 |
| render_mode、record_events | None / human / rgb_array；是否记录详细事件 |

实例独立保存几何、模型和 RNG。scene_scale 缩放默认地图、生成区域、碰撞/攻击距离，不缩放真实飞行器的质量、惯量、机翼或旋翼尺寸；显式 world_bounds/target_region 等是最终米制值。

`initialization="random"/"uniform"` 控制无人机开局。`target_initialization="random"/"fixed"` 独立控制目标：random 每次从目标区域采样；fixed 默认 x/z 取区域中点、y 在内部等距排列。`target_positions=[[x,y,z],...]` 优先并令实际模式为 fixed。二维所有实体固定 z，垂直速度为零。完整配置下 reset 相同 seed 重现同一开局；连续训练无需每次重复相同 seed。

## 控制空间与同步提交

| 模型/控制 | Parallel 单个 agent 空间 | 解释 |
|---|---|---|
| 三维 acceleration，离散 | Discrete(27) | 0 为零加速度意图，其余为归一化三维方向 |
| 二维 particle acceleration，离散 | Discrete(9) | 原 27 基元中 z=0 的保序子集 |
| acceleration，连续 | Box(-1,1,(spatial_dim,)) | 按模型尺度转换；UAV 经内环转成物理执行器 |
| 固定翼 actuator | Box(-1,1,(4,)) | elevator/aileron/rudder/throttle；舵面 ±25°，油门 `(u+1)/2` |
| 四旋翼 actuator | Box(0,1,(4,)) | 前、右、后、左旋翼推力比例；0 为无推力 |
| position | Box(world_low,world_high,(3,)) | 绝对世界目标点；二维 z 上下界都等于 plane_altitude |

position 是目标，不是位置赋值：粒子经速度/加速度反馈移动，UAV 经姿态、推力/舵面反馈与刚体积分移动。固定翼朝目标飞行并可能穿越，不能悬停。acceleration 也不直接指定 UAV 瞬时物理加速度。actuator 零向量不等于悬停/配平；四旋翼理想悬停各旋翼 .4，固定翼使用求得的 trim 输入。

动作字典必须恰好覆盖当前 env.agents，所有有效动作在碰撞/物理更新前整批验证。survival 死亡 agent 在最后一次转移后移除；damage 固定死亡槽仍要求字典键，其动作被中性输入替代。双方基于同一步开始观测提交动作。

Parallel 二维编号为 0..8，对应全局 27 类编号 `[0,2,5,8,11,16,19,22,25]`。Flight 历史 action_mode=discrete27 保留全局编号，二维仅允许该子集。MPE 离散输入可用整数或严格匹配 9/27 长度的 one-hot，不可跨维度复用索引。

## 返回值、观测与集中状态

Parallel：`reset(seed=None,options=None) -> (observations,infos)`，`step(actions) -> (observations,rewards,terminations,truncations,infos)`。名称为 red_0...、blue_0...；possible_agents 和空间在实例内固定。state() 在 reset 后可读取，包括最终状态。render() 依模式返回图像、显示窗口或 None，close() 释放显示资源。

设无人机数 A、目标数 K、总实体数 E=A+K。每个相对实体行 11 列：`[relative_position3,relative_velocity3,health,alive,is_red,is_blue,is_target]`。排除自身，其余按固定世界顺序排列；死亡行全零，observation_entities 给出名称。位置按实例世界对角线、速度按 `2*模型速度参考上限` 归一化，不强制落入 [-1,1]。公开观测没有按 Scout 扇区隐藏实体。

| 模式 | observations[agent] | state() 长度 |
|---|---|---|
| particle acceleration | (E-1,11) 数组 | 11E |
| particle position | Dict：entities(E-1,11)、self_state(6,) | 11E |
| UAV 任一控制 | Dict：entities(E-1,11)、self_state(13,) | 11E+13A+3 |

particle self_state 为米制 p3/v3；UAV 为 `[p3,v3,quaternion4,body_omega3]`，四元数 scalar-first、机体到世界。UAV 集中状态在 11E 之后追加各无人机 13D 块及 `[cycle,target_damage,is_damage]`。死亡刚体块全零。particle 集中 state 保留旧布局，没有时钟和累计 D，不能称为数学完整 Markov S；UAV 增强块也不自动公开对手隐状态。

MPE reset() 返回固定顺序 obs_n；step() 返回 `obs_n,reward_n,done_n,{"n":infos}`。Dict 观测依 Gymnasium flatten 展平，observation_space 给出最终长度。done_n 合并 termination/truncation，但 info[n] 保留区别；已移除旧槽后续为零观测、零奖励、done=True。

## 任务、奖励和 bootstrap

| 规则 | survival | damage |
|---|---|---|
| 目标 HP | 扣血至 0 | 保持有限初始 HP，仍累计原始伤害 |
| 自然结束 | 任一目标 HP<1e-3，或蓝攻击者全灭 | 蓝攻击者 HP<=0 全灭 |
| 红方全灭 | 不独立触发全局结束 | 不独立触发，后续目标损失继续计入 |
| 原生默认奖励 | 红 `10*outcome_red`，蓝相反 | 红 -step_target_damage，蓝相反 |
| outcome_red | 目标毁坏 -1，蓝攻击者全灭 +1，其余 0 | 0，不映射为胜负 |
| episode_returns | None | Red=-D、Blue=D，未折扣 |
| reward_weights | 可取四维 latent 加权和 | 拒绝，保留原始伤害计分 |

survival 的 LatentReward=[hit,target,enemy,episode] 与 RealReward 均保留；第四项已含终局奖励，不能重复相加。默认标量用 RealReward，显式四维 reward_weights 才用 latent 加权和。damage latent 为零。每队员收到完整本队 R，不是人数分摊；同队 rewards 或重复 episode_returns 不能求和当作队伍得分。

step_target_damage 是本物理步全部目标的原始伤害和，target_damage 是 reset 起累计 D。计量在 HP 截断之前，多个同时来源全部计入，不按 HP 封顶/归一化。damage 显示 HP=2 时累计 D 可超过 2。target_damage_by_target 键是原生实体 Id，不是分组的目标局部下标。

Parallel max_cycles 为采样截断，不因到时授予新胜利；自然结束与到时重合时优先自然结束。survival 个体死亡可令自身 terminated=True，但不等于团队结束。damage 死亡槽继续领未来团队奖励，直到全局结束/截断。

| info 字段 | 含义 |
|---|---|
| alive、agent_mask | 返回观测的存活位，1=活；供下一次 actor 使用 |
| entity_mask | 原生相对实体行 1=有效/存活，0=死亡；死亡 damage 观察者全 0 |
| action_mask | 离散活槽全 1，死槽仅 noop 为 1；连续为 None |
| terminated、truncated | 当前个体转移标志 |
| global_terminated、global_truncated | 团队自然结束、采样到时 |
| bootstrap_mask | float(not global_terminated)，死亡/采样截断仍可为 1 |
| cycle、sim_time、max_cycles、remaining_cycles | 实际物理时钟与采样预算 |
| termination_reason | target_destroyed / blue_attackers_destroyed / None |
| events、effective_config | 真实事件与实例配置 |

actor 损失使用致死动作发生前的 mask。团队 critic 不应再乘 agent_mask，不能把 terminated or truncated 一律当零 bootstrap；截断估值用最后真实状态，不用下一次 reset 状态。

## 分组与 nv1

```python
from had_env import make_env
from had_env.grouping.policies import RulePolicy

env = make_env(api="grouping", red=8, blue=8, targets=2,
               task_mode="damage", max_steps=100, spatial_dim=2)
policy = RulePolicy()
state = env.reset(seed=42)
while not env.done:
    state, reward, done, info = env.step(policy.act(state))
env.close()
```

KnownOpponentEnv 直接构造默认 seed=0、opponent=reactive、command_interval=5。蓝上层可为 reactive/balanced/concentrated，下层用 rush；HADStage3Adapter 另支持 split_rush。分组只接受 acceleration 意图，position/actuator 使用 Parallel/MPE/Flight。

Grouping(groups,reserve) 的 Group(target,members) 中，target 为稳定 0..K-1 下标，members 为原生全局 agent Id；所有存活红方恰好出现一次。reserve 不分配目标；默认 group_max_size=None 无人工人数上限。死亡只从旧关系中删除，不自动修补上层分配。

红蓝上层基于同一份提交前状态独立决策。一个 step 至少推进一个物理步，在绝对周期 tick、非终局伤亡或结束处返回。info[delta] 是真实推进步数；红全灭后继续真实蓝方调度至结束/时限。damage 宏奖为物理步负伤害的未折扣和，macro_target_damage=-reward。分组 survival 返回终局红胜 1，否则 0；horizon_policy=red_win/draw/blue_win 属于分组任务定义，与 Parallel 采样到时不同。

`had_env.open_score_compat.rules.CoveragePolicy` / CoverageExecutor 提供 rule_nv1_v1：每物理步预测蓝方一步直线位置，解二元整数分配，每红对应一名选中蓝、每选中蓝至少一红覆盖。红不足时按蓝到最近红的距离择取，余下蓝记为 ignored。Grouping.target 仍是保护目标，追击蓝 Id 位于 last_plan[assignment]；执行器只复用当前相同物理状态的解。

## Open-SCORE / ALMA 兼容接口

```python
from had_env.open_score_compat import make_open_score_env

env = make_open_score_env(profile="training", scale=(8,8,2), seed=0)
# 统一入口：make_env(api="open_score", profile="training", scale=(8,8,2))
entities, masks = env.reset(seed=42)
reward, done, info = env.step([0] * env.n_agents)
env.close()
```

原构造参数保持：`HADEntityEnv(seed=0,worker_id=0,train_dist="mixed_le10",scale=None,config=None,max_steps=100,episode_limit=None,blue_upper="reactive",blue_lower="rush",entity_scheme=True,gamma=.99,**kwargs)`。此处直接 config 是原规模三元组或 `{N_R,N_B,K}`，与统一工厂的配置文件/环境配置不是同层参数。episode_limit 优先于 max_steps；scale=None 每回合从 N∈{4,6,8,10}、双方等人数、K∈{1,2,3} 的 mixed_le10 分布采样。上限红/蓝各 50、目标 12。

包内 `make_entity_env` 为 make_open_score_env 别名；HADWrapper 是同一物理封装的低层入口。scales 模块的 Scale/as_scale、ScaleSampler、SCALE_POOLS 提供原版规模表示和独立采样；PLANAR_NATIVE_IDS/NATIVE_TO_PLANAR 与 native_actions_to_planar 提供两种编号映射。EpisodeDiagnostics/trajectory_frame 保存实际物理过程；不导入 Torch、SMAC 或 FireFighters。

| 默认项 | evaluation（eval） | training（train） |
|---|---|---|
| pad | 50 红/50 蓝/12 目标 | 10 红/10 蓝/3 目标 |
| shaping_coef / shaping_range | 0 / 4000 m | 1 / 4000 m |
| blue_upper / blue_lower | reactive / rush | 同左 |
| command_interval | 5；decision_interval 为别名 | 同左 |
| gamma / fold_wipeout_tail | .99 / True | 同左 |
| reward_mode / friendly_penalty | damage / 1（friendly 才应用） | 同左 |
| subtask_set / allocation_clock | targets / interval | 同左 |
| action_length / max_alloc_hold | 5 / 10 | 同左 |

显式参数覆盖 profile 默认。pad 可为 eval/train 或 `(n_red,n_blue,n_targets)`；reset 规模不得超 pad；pool_slots 可再限制有序槽预算。兼容只支持 particle、二维、离散 acceleration、damage、entity scheme；UAV、三维、连续、position/actuator、survival 明确拒绝。

实体表为固定 red/blue/target 槽、float32、10 列：`[x/2500,y/2500,vx/120,vy/120,alive,HP/initial_HP,target_cumulative_D/N_B,is_red,is_blue,is_target]`。死亡无人机/padding 全零；目标只填位置、D 比例及 target 位。state 为实体表展平加 `[red_alive_fraction,blue_alive_fraction,1-step/episode_limit]`；obs 为每红槽重复同一展平表，死亡/padding 观察者全零，与原生相对 11 列不同。

| 公共方法 | 返回/用途 |
|---|---|
| reset(seed=None,config=None,evaluate=False,retain_trajectory=False,test=False,**kwargs) | entities,masks；evaluate/test 开诊断，显式 seed 不消耗 episode seed stream |
| step(actions) | float reward、bool done、info；长度为实际到 padded 红人数之间，编号 0..8 |
| get_entities()、get_entity_size() | 表副本；10 |
| get_masks() | obs_mask[E,E]、entity_mask[E]；**1=屏蔽/不存在，0=有效** |
| get_agent_mask() | 红 actor mask，1=活；1-entity_mask[:n_agents] |
| get_initial_agent_mask() | 1=初始 padding；真实红槽后来死亡仍为 0 |
| get_avail_actions()、get_avail_agent_actions(i) | [n_agents,9] 或单行；死槽仅 noop |
| get_total_actions() | 9 |
| get_state()、get_state_size() | 10E+3 集中状态及长度 |
| get_obs()、get_obs_agent(i)、get_obs_size() | [n_agents,10E]、单行、10E |
| get_env_info(args=None) | n_agents/n_entities/n_actions/entity_shape/state_shape/obs_shape/episode_limit/n_tasks 等 |
| get_task_masks() | task_mask、entity2task_mask（0=属于任务）、hier_decision |
| get_policy_state() | 全局 Id 的 DecisionState，含 initial_health/cumulative_damage |
| episode_summary()、get_stats() | 物理 -D、D、rho=D/N_B、ep_len、存活数、逐目标 D、诊断 |
| get_agg_stats(stats) | 保留原合约，空字典 |
| get_trajectory()、save_replay() | 物理轨迹副本；默认不录制 |
| get_rng_state()、set_rng_state(state) | 规模采样与 episode seed stream，非物理快照 |
| close() | 关闭底层 native |
| env.wrapper.snapshot()/restore(state) | 内存物理分支点，包括 RNG、诊断、动作及层次时钟 |

targets 子任务中每个真实目标一项，蓝按公开最近目标距离归属，不读取私有蓝分配；红初始连全部有效任务。blues 子任务中每蓝槽一项，死亡蓝任务屏蔽，存活目标为共享上下文。interval 按 action_length 决策；event 在伤亡、最近目标变化且持有至少 2 步、或达到 max_alloc_hold 时决策。

task_rewards 之和等于标量 reward，tasks_terminated 只标真实任务。shaping 为 `gamma*Phi(next)-Phi(now)`，自然终局 Phi=0，截断保留 successor 势。friendly 额外惩罚红红碰撞对及多余截击自毁，可能改变最优策略。episode_summary.return 始终是物理未折扣 -D，不含 shaping/friendly。

默认 fold_wipeout_tail=True：红全灭后继续实际蓝方物理，把未来折扣奖励折入当前转移，令红决策过程 terminated=True、truncated=False、bootstrap=0。folded_steps/step 保留实际工作量，terminated_naturally 表示底层是否自然结束。关闭折叠时到时 truncated、bootstrap=1；数学定义见 MODELING。

兼容代码与 literal 基准来源是 Open-SCORE revision `6f119e9904b9dfed9cb08b660aff078a2d82247d` 的 envs、utils/seeding.py、rules/coverage_rule.py；捕获时这些来源文件无未提交修改。tests/data/open_score_reference.json 保存原版输出：seed/规模序列、实体与 state、子任务奖励、25 个正常物理转移、4 步折叠尾段、诊断轨迹、nv1 分配/动作。期望值由原版模块捕获，没有用新实现生成。

## 工作台、策略、回放和分支

FlightSession(FlightScenarioSpec(...),red_policy=...,blue_policy=...) 接入两队 act(observation)。每队收到固定 roster 的 observation、agent_ids、alive_mask、动作协议和边界；可用 `session.step({"red":actions,"blue":actions})` 覆盖，缺省队由注册策略产生动作。动作按 roster 给数组或按全局 Id 给字典。Flight continuous_native 为历史原生意图入口，不应假设与 Parallel normalized Box 完全相同。

SimulationSession(ScenarioSpec(...),policy=...) 接分组策略；act(state) 收到复制状态，可信扩展 act_env(branch) 收到独立规划环境。PolicyAdapter 复制策略并隔离 Python/NumPy 与可选 Torch 随机流。Torch 是调用方可选依赖，需要时先初始化模型/设备，再建适配器。Flight 策略不获取实时引擎对象。

session.snapshot()/branch() 支持独立内存续演。完整录制保存配置、模型/控制标识、每物理步、HP、逐步/累计 D、UAV q/omega、指派、动作、事件和决策边界白名单 JSON 快照。离线恢复校验类型/形状/协议/来源身份，不加载任意 pickle；跨任务、维度、模型、配置或物理协议应拒绝。历史/稀疏轨迹可只读查看。显示插值只影响视觉位置，不改物理事件/得分。

CLI 保留 record/live/view/inspect/export/branch/evaluate；--agent-type、--agent-action-type、--spatial-dim、--config 选实例配置。原生 position/actuator 用 `--native --action-mode continuous_native`。历史 known-rule-mixed-v5.1 的 50 步 survival 为特定任务；修改目标/对手/HP/时限须用独立 ScenarioSpec 标识，damage 使用独立协议。旧胜率、黄金轨迹和吞吐保留原版本/原协议，不能自动混入当前结果。
