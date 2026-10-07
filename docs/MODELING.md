# SkyHAD 任务与动力学建模

本文按 SkyHAD 3.0.0 的实现区分数学完整状态、公开模型输入、环境任务、控制意图和物理执行器。接口字段见 [API.md](API.md)。环境不绑定训练算法；gamma 是调用方的折扣约定，只有明确接受 gamma 的兼容封装参与奖励折扣/shaping。

## 联合博弈与固定对手 POMDP

双方独立学习、同时提交动作时，环境是部分可观测随机博弈 POSG：联合动作 `(a_R,a_B)` 决定共同物理转移，各方保留自己的策略/历史/奖励。当前对手待提交动作不属于己方可观察量。

固定蓝方策略 piB 后，红方可建模为

\[
\mathcal M=(\mathcal S,\mathcal A,T,\mathcal O,Z,R,\gamma,\rho_0,H).
\]

T 是积分固定蓝策略的转移核；O 为实际红方观测空间，Z 是状态到观测的映射/核；rho0 为 reset 分布；H 为任务时限或采样预算。蓝策略的隐状态、承诺分配、决策时钟和随机流必须纳入 S，才能保持 Markov 性。原生同步接口本身不假定蓝方固定；KnownOpponentEnv 与 Open-SCORE 提供固定规则过程。

完整 S 包含所有 Attack、Scout、Disturb、Target 的稳定 Id/角色/阵营、位置 p、速度 v、HP、存活位、逐步及累计 D；物理时间、任务模式与边界；红蓝分组/导航承诺、reserve、上层决策时钟、下层最后动作/必要记忆，以及影响未来的策略内部状态。UAV 增加机体到世界单位四元数 q4 和机体系角速度 omega3。配置和 seed 是回合固定上下文；精确分支还保留 RNG 状态。

公开观测无传感噪声，Z 为确定性映射。原生相对实体表公开全部其他存活实体的 p/v/HP/存活和阵营类型位，没有根据 Scout 探测扇区隐藏实体。Scout 功能规则/辅助奖励保留，但不能据此宣称 actor 输入只有 Scout 视野。相对表仍可能遗漏绝对边界位置、自身刚体、角色细分、对手承诺、时钟和策略记忆，因此不等于完整可观测 Markov S。

particle 的原生 11E 集中 state 没有时钟、累计 D 和上层承诺。有限时限影响价值时，应显式补入剩余时间。D 是计分/恢复状态，尽管下一步增量伤害通常无需累计值即可计算。UAV 增强 state 追加所有无人机 13D 刚体块和 `[cycle,D,is_damage]`，修复刚体转移与任务上下文的重要缺项；隐藏对手内部状态仍需另行处理。完整快照比训练 feature vector 包含更多信息。

rho0 由实例私有 RNG 生成：目标 random/fixed，红方在目标附近环带，蓝方在来袭区域。二维/UAV 拒绝过近开局，不禁止后续碰撞。uniform 是独立展示初始化。固定翼 reset 建立参考配平水平飞行；四旋翼允许零速度。


### 状态、观测和核的具体含义

令无人机数为 N，目标数为 K，所有实体集合为 E。回合固定配置 c 包含场景尺度、边界、角色名册、模型参数、控制形式与目标初始 HP。配置确定后，完整状态可写为

\[
s_t=(\{p_i,v_i,h_i,\ell_i,r_i,\chi_i\}_{i\in E},
\{q_i,\omega_i\}_{i\in\mathrm{UAV}},\{D_k\}_{k=1}^K,t,g_R,g_B,m_R,m_B).
\]

| 分量 | 空间/单位 | 为什么属于状态 |
|---|---|---|
| p_i, v_i | 三维米、米/秒 | 决定运动、碰撞、攻击距离和扇区方向 |
| h_i, ell_i | 非负 HP、存活布尔值 | 决定参与碰撞/功能动作/奖励及终止；ell_i 由 HP 按对应阈值派生 |
| r_i, chi_i | Attack/Scout/Disturb/Target 与阵营 | 决定自动功能规则、哪些攻击源能命中该实体 |
| q_i, omega_i | 单位 S³ 四元数、三维 rad/s | 决定力与力矩的方向，以及下一步姿态和速度 |
| D_k | 非负累计原始伤害 | damage 最终报告、精确回合恢复和统计的计分状态 |
| t | 已执行物理步数 | 决定时限、周期分配机会和剩余任务时长 |
| g_R, g_B | 分组、目标归属及 reserve | 在两次上层决策之间持续影响下层导航 |
| m_R, m_B | 会影响未来的策略记忆 | 有记忆的固定对手/执行器需要该分量；当前刚体内环无隐藏积分器 |

变量并非都独立：存活位派生自 HP，q 的四个存储量只有三个姿态自由度。随机数生成器状态用于精确复现；在概率模型中可等价视作每步从规定随机核采样，不必把伪随机实现细节提供给 actor。

原生存活 agent i 看到的每个其他实体 j 的十一列为

\[
o_{ij}=\left[\frac{p_j-p_i}{L_c},\frac{v_j-v_i}{2V_{max,c}},
h_j,1,\mathbf1_{Red},\mathbf1_{Blue},\mathbf1_{Target}\right],
\quad L_c=\sqrt{\sum_{a=x,y,z}(b_a^+-b_a^-)^2}.
\]

死亡实体整行置零，行顺序固定；这些阵营标志不编码 Attack/Scout/Disturb 的角色细分。UAV 自身块提供未归一化 `[p_i,v_i,q_i,omega_i]`，particle position 自身块提供 `[p_i,v_i]`。死亡 damage agent 的整个输入为零。Z 是该确定性编码：`Z(o|s)=delta(o-encode(s))`。Open_Score 的十列及 padding 属于另一观测空间，不能把其张量解释为原生十一列。

对固定蓝策略，转移核定义为

\[
T(s'\mid s,a_R)=\int \delta\!\left(s'-F_c(s,a_R,a_B)\right)
\pi_B(da_B\mid o_B,m_B).
\]

F_c 是下文的运动积分、边界、碰撞及同时交互组成的确定性转移。reactive 蓝方的随机分配在其决策时钟采样；原生 actuator 直接接收蓝动作时，应使用联合 F_c，而不是假定上述 piB 存在。观测空间 O 是所有可能编码值的集合，Z 是生成这些值的核，二者含义不同。

如果策略维护信念 b_t(s)，标准更新为

\[
b_{t+1}(s')\propto Z(o_{t+1}\mid s')\int T(s'\mid s,a_t)b_t(s)ds.
\]

该公式描述在遗漏对手记忆、承诺或自身绝对状态时如何利用历史推断，并不表示本仓库实现了信念滤波器。当前接口可直接接入无记忆策略，也可由算法维护历史。

## 两种任务分别定义

### Survival POMDP

\[
\mathcal M_{survival}=(\mathcal S_{survival},\mathcal A,T_{survival},\mathcal O,Z_{survival},R_{survival},\gamma,\rho_{0,survival},H_{survival}).
\]

S 使用目标实际剩余 HP。A 为所选接口的红方飞行输入或 Grouping，蓝方由固定 piB 决定。T 包含运动、碰撞、自动开火、攻击/干扰扣血和开火者自毁；O/Z 按接口实际布局。任一目标 HP<1e-3 时蓝胜，蓝攻击者 HP<1e-3 全灭时红胜；红全灭不独立触发全局结束。

原生红方默认 `R_t=10*outcome_red`，蓝方相反；显式 reward_weights 才改用四维 latent 加权信号。分组 R 为宏步终局红胜指标 0/1，不是原生 ±10。H 在 Parallel/Flight 为采样上限，到时 truncation；分组 survival 的 horizon_policy 明确规定 red_win/draw/blue_win，属于有限任务规则。gamma 由算法选择，当前 KnownOpponentEnv 宏奖采用 gamma=1 的累计约定。

### Damage POMDP

\[
\mathcal M_{damage}=(\mathcal S_{damage},\mathcal A,T_{damage},\mathcal O,Z_{damage},R_{damage},\gamma,\rho_{0,damage},H_{damage}).
\]

S 包含目标固定初始 HP 和逐目标累计 D，不设目标伤害预算。T 继续执行飞行、碰撞、攻击和自毁，目标不因攻击扣 HP/死亡。蓝攻击者 HP<=0 全灭才自然结束，红全灭后的目标损失仍属于红方回报。

A 随所选接口而变，O/Z 使用对应实际布局及死亡槽规则；rho0 与 survival 共享几何采样方式，但 task_mode 和计分状态不同。gamma 由算法或兼容构造定义，H 是该配置的采样上限。

\[
d_t=\sum_{k=1}^K\sum_b d_{b\to k,t},\qquad
D_T=\sum_{t=0}^{T-1}d_t,\qquad
R^R_t=-d_t,\quad R^B_t=d_t.
\]

d 是所有来源在 HP 截断前的原始贡献，不除队伍人数/目标数/HP。报告 episode_returns 为未折扣 ±D，算法可另优化折扣和。每个队员收到全队 R，不能再对 agent 奖励求和。rho=D/N_B 是兼容诊断而非原生奖励。H 到时保留 bootstrap，不附加胜负或存活奖。


### 伤害函数、两任务的状态差异与奖励时序

令 I_b 为碰撞检查后蓝 Attack 的自动开火位，r_in/r_out 为对应实例的内/外半径。共同步开始位置产生

\[
\eta(d)=\begin{cases}1&d<r_{in},\\
(r_{out}-d)/(r_{out}-r_{in})&r_{in}\le d<r_{out},\\
0&d\ge r_{out},\end{cases}\qquad
L_k=\sum_{b\in BlueAttack}I_b\,\eta(\|p_b-p_k\|).
\]

开火条件是严格 `d<fire_range`，而 eta 在内半径边界连续为 1；触发条件和爆炸伤害衰减是两个不同判断。同一来源可对爆炸范围内多个敌方实体贡献伤害，且本步多个来源的贡献相加。

survival 目标更新 `h'_k=max(0,h_k-L_k)`；damage 目标更新 `h'_k=h_k`、`D'_k=D_k+L_k`。移动无人机照常扣攻击及干扰伤害，开火者本步自毁。两个任务的动力学、动作和观测编码相同，但目标 HP 转移、自然终止条件和团队奖励函数不同。

原生 survival 默认每个本步还在接口中的 agent 收到对应阵营的终局 ±10；终局后 Parallel 不再推进，避免重复领取。latent 的 hit、target、enemy 是既有几何辅助量，episode 槽已经含团队奖励，显式加权时应直接做四维点积。damage 每物理步只奖励 `-(D'−D)`，重复调用 info/reward 不增加 D，不再叠加 survival 终局奖励。

## 运动学代理与刚体模型

particle 保留运动学基准，每外部步 dt=1 s：先按步开始速度更新位置，再用加速度意图更新速度，并施加其速度/转向规则；二维固定 z、垂直速度为零。它没有姿态、惯量、机翼或旋翼，不用于推断实机 UAV 动力学。

两种 UAV 均用 `[p3,v3,q4,omega3]`。世界为右手 ENU（东/北/上），机体为右手 FLU（前/左/上），R(q) 把机体向量旋转到世界；q 为 `[w,x,y,z]`，omega 用 rad/s。统一 Newton–Euler 方程为

\[
\dot p=v,\qquad m\dot v=R(q)F_B-mg e_3,
\]
\[
\dot q=\tfrac12q\otimes(0,\omega),\qquad
J\dot\omega=M_B-\omega\times(J\omega).
\]

F 不含重力，g=9.81。积分用 RK4，默认子步 .01 s，1 s 外部步包含 100 子步；其他 dt 按 dt/ceil(dt/.01) 等分，每子步归一化 q。actuator 在外部步保持；高层意图保持，内环在每个 RK4 stage 用真实 stage 状态重新反馈，无隐藏积分器。

模型本身不把实际空速/姿态钳在参考控制界内。场景边界另裁剪位置并去掉朝外速度分量，这是任务边界规则。UAV 碰撞使用同步子步轨迹。当前无风、传感噪声和执行器动态，数值积分验证不等于实机系统辨识。

## 近期研究与本项目的选择

[Wang 等的固定翼六自由度研究](https://www.sciencedirect.com/science/article/pii/S294985542500070X) 于 2025 年 11 月在线发表，刊于 2026 年 6 月的 Journal of Automation and Intelligence。论文将高层速度指令和直接执行器控制分开，以 JSBSim 推进刚体状态。本项目据此区分控制意图、内环控制器和实际动力学；参数使用下面的 Aerosonde 数据，未采用该论文的 Skywalker X8 模型。

[2025 年 Aerial Gym Simulator 论文](https://arxiv.org/abs/2503.01471) 为不同驱动形式的多旋翼提供模块化模型和几何控制器。本项目采用相同的分层思路：位置或加速度意图经过几何反馈与推力分配，直接执行器动作则进入刚体模型。具体方程与可复现参数来自下面的 Lee 模型；场景任务、观测和自动攻击规则沿用 HAD。

## 固定翼：Aerosonde 数据与参考控制

物理、机翼、气动、电机/螺旋桨参数取自 BYU MAGICC 官方 [Aerosonde 参数文件](https://github.com/byu-magicc/mavsim_public/blob/main/mavsim_python/parameters/aerosonde_parameters.py)。力学和控制代码独立编写，不导入其仿真器。传统 FRD（前/右/下）到 FLU 的 `C=diag(1,-1,-1)` 是 det=+1 的旋转；力、力矩和惯量均转换，`J_FLU=C J_FRD C^T`。

| 物理量 | 当前值 |
|---|---|
| m | 11 kg |
| J_FRD | `[[.8244,0,-.1204],[0,1.135,0],[-.1204,0,1.759]]` kg·m² |
| 机翼面积/翼展/弦长 | .55 m² / 2.8956 m / .18994 m |
| 空气密度/Oswald 系数 | 1.2682 kg/m³ / .9 |
| 螺旋桨直径/电机电阻/空载电流/最高电压 | .508 m / .042 Ω / 1.5 A / 44.4 V |
| 电机常数 | `60/(145*2*pi)` |
| CQ0,CQ1,CQ2 | .005230, .004970, -.01664 |
| CT0,CT1,CT2 | .09357, -.06044, -.1079 |

alpha/beta 从机体相对气流求取，动态压强为 rho*Va²/2。纵向含升力/阻力/俯仰力矩与速率/升降舵项，横侧向含侧力/滚转/偏航力矩与 beta、滚转/偏航速率、副翼/方向舵项。失速用平滑线性升力/平板升力混合，M=50、alpha0=.47 rad。阻力使用 `CDp+(CL0+CLalpha*alpha)^2/(pi*e*AR)` 诱导阻力极曲线，再加当前速率/舵面项；保留的可选 CDalpha 不等于实际阻力公式。推力/反扭矩来自稳态直流电机平衡和二次螺旋桨拟合，无转速滞后。

参考速度 18–30 m/s，配平/巡航 30，加速度意图尺度 5 m/s²；bank/pitch/path 参考限幅 45°/20°/15°，舵面 ±25°。trim 同时求解三轴力与三轴力矩平衡，reset 用该水平飞行解。高层控制按参考速度缓存并调度完整配平，使用对应的俯仰角、平衡侧滑角、全部舵面与油门作为反馈基准；18、24、30 m/s 的水平平衡均由六轴方程决定。

控制限制及增益属于工程选择：航迹修正 .5，升降舵 pitch 反馈 -2/q 阻尼 .25，副翼 roll 反馈 .5/p 阻尼 -.08，方向舵 beta .8/偏航误差 .1，油门速度反馈 .08/爬升前馈 .7。这不是 BYU 完整自动驾驶器或最优控制参数。position 飞向/穿越目标，不瞬移、不悬停。


气动公式中的机体系空速 `(u,v,w)` 先转换至 FRD，`Va=||(u,v,w)||`、`alpha=atan2(w,u)`、`beta=asin(v/Va)`。平滑失速系数为

\[
\sigma(\alpha)=\frac{1+e^{-M(\alpha-\alpha_0)}+e^{M(\alpha+\alpha_0)}}
{(1+e^{-M(\alpha-\alpha_0)})(1+e^{M(\alpha+\alpha_0)})},
\]
\[
C_L=(1-\sigma)(C_{L0}+C_{L\alpha}\alpha)+
\sigma\,2\operatorname{sign}(\alpha)\sin^2\alpha\cos\alpha+
C_{Lq}\frac{c q}{2V_a}+C_{L\delta_e}\delta_e.
\]

`L=qbar*S*CL`、`D=qbar*S*CD`，纵向 FRD 力为 `[-D cos(alpha)+L sin(alpha), Y, -D sin(alpha)-L cos(alpha)]`；横侧向系数由 beta、`bp/(2Va)`、`br/(2Va)` 和舵面输入组成。滚转/俯仰/偏航力矩分别乘 `qbar*S*b`、`qbar*S*c`、`qbar*S*b`。零空速时动态压强为零，速率分母仅使用有限小下界。

稳态电机转速 Omega 由二次平衡的非负根决定。螺旋桨推力与反扭矩采用

\[
T_p=\rho\!\left(C_{T0}\frac{D_p^4\Omega^2}{4\pi^2}
+C_{T1}\frac{D_p^3V_a\Omega}{2\pi}+C_{T2}D_p^2V_a^2\right),
\]
\[
Q_p=\rho\!\left(C_{Q0}\frac{D_p^5\Omega^2}{4\pi^2}
+C_{Q1}\frac{D_p^4V_a\Omega}{2\pi}+C_{Q2}D_p^3V_a^2\right).
\]

配平同时要求三轴 `v_dot=0` 与 `omega_dot=0`；这包含螺旋桨反扭矩，不能用单独的“升力等于重力”代替完整配平。实现中的全部实际气动系数可在 `FixedWingDynamics.coefficients` 查看；没有额外依赖或预计算黑箱轨迹。

## 四旋翼：理想模型与几何反馈

刚体结构、物理量和姿态增益参照 Lee、Leok、McClamroch 的 [arXiv:1003.2005v2，II/VII 节](https://arxiv.org/pdf/1003.2005v2)，其轴约定一致旋转至 FLU/ENU。m=4.34 kg、J=diag(.0820,.0845,.1377) kg·m²、l=.315 m、扭矩/推力系数 c=.008004 m；kR=8.81、kOmega=2.54 来自论文数值例。

前/右/后/左旋翼 `u_i∈[0,1]` 转成 `f_i=u_i*f_max`，机体推力沿 +z：

\[
\begin{bmatrix}f\\M_x\\M_y\\M_z\end{bmatrix}=
\begin{bmatrix}1&1&1&1\\0&-l&0&l\\-l&0&l&0\\c&-c&c&-c\end{bmatrix}
\begin{bmatrix}f_1\\f_2\\f_3\\f_4\end{bmatrix}.
\]

推重比 2.5 为**工程假设**，`f_max=2.5mg/4`，理想水平悬停各 u=.4。分配器保留可行总推力，统一缩放差分力矩以满足旋翼上下界，不独立截断后假定总推力不变。

acceleration 的速度参考上限为 20 m/s；position 外环另用 `v_des=clip_norm(.5*(p_des-p),12)`，保持到点导航的速度尺度。速度外环 `a_des=clip_norm(2*(v_des-v),6)`。目标推力方向为 `m*(a_des+g e3)`，参考朝向固定东；`eR=.5*(Rd^T R-R^T Rd)^vee`，力矩 `-8.81eR-2.54omega+omega×Jomega`。20/12 m/s、6 m/s² 和外环 .5/2 是本项目参考值，不是论文完整位置增益。

代码保留姿态 PD 与陀螺补偿，未实现论文完整轨迹导数/期望角速度前馈，不能直接继承其完整稳定性结论。忽略气动阻力、旋翼陀螺和电机滞后。[Faessler 等 RAL 2018 原论文](https://rpg.ifi.uzh.ch/docs/RAL18_Faessler.pdf) 讨论 rotor drag；这里是去掉该项的理想刚体基线，不是高速 drag-aware 已辨识模型。

## 三种动作语义

acceleration 为世界系高层意图，经模型尺度转换一次建立 `v_des=v_start+a_intent*dt`，外部步内目标速度保持、内环随 stage 状态反馈。position 为米制绝对世界点，不统一承诺“到点即停”。actuator 直接进入气动力或推力/力矩模型。三种 A 不同，策略输出/checkpoint 不可混用。

分组给导航职责，规则给 acceleration 意图，内环转成有限执行器，动力学从真实状态推进。当前分组只接受 acceleration，避免把 Grouping 的职责语义当成舵面/旋翼动作。

## 碰撞、自动攻击和同时结算

整批动作校验后，UAV 预测全部存活无人机同步 RK4 轨迹，particle 用本步线段，以同一时间参数求扫掠最近距离。不同时间路过同一点不构成碰撞。两机中心距离阈值为 `20*scene_scale`，目标不参加无人机碰撞，同阵营也会撞毁。按固定遍历处理，已在该遍历中死亡者不继续链式撞毁他者。

碰撞后存活实体在**步开始位置**判定开火，再用共同 WorldKinematics 快照结算。红 Attack 因蓝 Attack 距离严格小于 fire_range 触发，蓝 Attack 因目标触发；无独立射击动作。友军不触发开火，也不受友军爆炸伤害，开火攻击者本步自毁。

满伤/归零半径二维 `(200,400)*scale`，三维 `(300,600)*scale`；内半径满伤，中间线性衰减，外半径及外为零，默认 AttackIntensity=1。开火/命中都用共同步开始位置，不混用更新前攻击者与更新后受害者位置。survival 目标扣血；damage 目标 HP 免疫但累计原始 d。干扰规则作用于无人机，目标不承受该干扰扣血。

事件 damage_scope=source_unclipped 是来源未截断贡献，多条同一步来源为同时事件，不能视作依次扣血。关闭详细事件不关闭 D。回放只插值视觉位置，不改物理时钟、HP、攻击、动作、指派和得分。

## 宏步 SMDP、折扣和全灭折叠

分组上层从一个决策边界执行到下一周期/伤亡/终局，持续 Delta 个物理步，是半马尔可夫过程 SMDP。按物理步折扣 gamma 的严格目标为

\[
\bar R_t=\sum_{j=0}^{\Delta-1}\gamma^jR_{t+j},\qquad
y_t=\bar R_t+\gamma^\Delta b_tV(s_{t+\Delta}).
\]

KnownOpponentEnv damage 当前返回未折扣 sum R 并提供 Delta，直接使用对应 gamma=1。gamma<1 时仅将 bootstrap 改为 gamma^Delta，而不折扣宏步内部奖励，不等价于物理步折扣任务；应由物理记录重建折扣宏奖或明确选择其他宏步目标。

Open-SCORE 默认红全灭折叠是另一约定：继续真实蓝方调度至自然结束或 H，将 `R_t+gamma R_{t+1}+...` 合入当前红转移，并终止红决策过程。即使底层到 H 尚未自然结束，封装 terminal/bootstrap=0；诊断保留真实物理步和 terminated_naturally。关闭 fold_wipeout_tail 则保留普通自然/采样截断语义。原生/分组不会把红全灭凭空判作底层胜利。

兼容势 Phi 为所有存活蓝方到公开最近目标的闭合比例 `clip(1-distance/4000,0,1)` 的负和乘 shaping_coef。每步加 `gamma*Phi(next)-Phi(now)`；自然终局 Phi=0，截断保留实际 successor 势；折叠终止退款去掉已记入却被丢弃的未来势。诊断 -D 不变。friendly 惩罚为非势代价，不能宣称保持原任务最优策略。

## Mask 与 bootstrap

原生 entity_mask 为 1=有效/存活；兼容 obs_mask/entity_mask 为 ALMA 屏蔽约定，1=缺席；兼容 agent_mask 又为 1=活，initial_agent_mask 为 1=初始 padding。不可跨接口复用同一“1 表示什么”的假设。

团队物理自然结束 bootstrap=0；Parallel/Flight 采样到时和未折叠 compat 到时 bootstrap=1，自然结束优先于同一步到时。survival 个体死亡不同于团队结束；damage 死亡槽继续领全队 R。actor 用致死前存活 mask，团队 critic 用全局 bootstrap。存活/padding/归属/任务终止/episode 终止描述不同事实。

Open-SCORE literal revision 见 [API.md](API.md)，当前执行范围/结果见 [VALIDATION.md](VALIDATION.md)。
