# SkyHAD v4 mathematical model

The [33-page formulation](PROBLEM_FORMULATION.pdf) and its [editable LaTeX source](PROBLEM_FORMULATION.tex) give sixteen defender-centric problem formulations. There is one overview page and two pages for each model/control/task combination. This document explains the shared assumptions, implementation correspondence and learning semantics. Public calls and tensor contracts are documented in [API.md](API.md).

| Model | Control interfaces | Tasks | Themes |
| --- | --- | --- | --- |
| Particle | Acceleration, absolute position | Survival, Damage | 4 |
| Fixed wing | Acceleration, direct actuator, absolute position | Survival, Damage | 6 |
| Quadrotor | Acceleration, direct actuator, absolute position | Survival, Damage | 6 |

The primary formulation is three-dimensional. Planar particle compatibility is a setting of the particle themes, not a seventeenth theme. Grouping is a general decision-layer extension. It currently executes acceleration controls and does not define a universal navigation action for every physical model.

## Defender perspective and complete state

Let Red defenders be \(\mathcal R\), Blue opponents \(\mathcal B\), stationary protected assets \(\mathcal P\), and UAVs \(\mathcal U=\mathcal R\cup\mathcal B\). Write \(N=|\mathcal U|\), \(K=|\mathcal P|\), \(M=N+K\). Attack, Scout and Disturb are fixed roles, separate from team membership. A complete state, conditional on the episode configuration \(C\), is

\[
s_t=(\{x_i,h_i,r_i,c_i,\mathrm{id}_i\}_{i\in\mathcal U},
\{p_k,h_k,D_k\}_{k\in\mathcal P},t,\zeta_t).
\]

Here \(x_i=(p_i,v_i)\) for particles and \(x_i=(p_i,v_i,q_i,\omega_i)\) for rigid vehicles. Health \(h_i>0\) determines participation. Targets start at finite health 2 by default and UAVs at 1; \(D_k\) is cumulative raw asset damage. Configuration fixes bounds, roster, parameters, control interface and task. Persistent assignments, reserve, decision clocks and policy memory belong in \(\zeta_t\) when an extension or opponent uses them. The implemented rigid-body controllers have no hidden integrators.

Last commands, functional flags, previous positions and step-damage counters are overwritten transition records; snapshots retain them for reward/event replay. An RNG state is needed for exact deterministic replay, but is not an actor feature. In a probabilistic model it can instead be represented by the stipulated random law.

Native execution accepts navigation actions for both teams. With both teams learning, the system is a partially observable stochastic game. After specifying and freezing an external Blue policy \(\pi_B\), Red has a cooperative Dec-POMDP:

\[
\mathcal M_R=(\mathcal R,\mathcal S,\{\mathcal A_i\},T^{\pi_B},
\{\mathcal O_i\},Z,R,\gamma,\rho_0,H),
\]
\[
T^{\pi_B}(s'\mid s,a_R)=\int\delta_{F_C(s,a_R,a_B)}(s')
\pi_B(da_B\mid o_B,\zeta_t).
\]

The transition includes any opponent-memory update. A learned opponent may be frozen by the experiment; the native environment does not supply that navigation policy automatically. Automatic firing is a separate functional rule, described below.

Decentralized actors use local histories \(\tau_i=(o_{i,0},a_{i,0},\ldots,o_{i,t})\). The defender objective is

\[
\pi_R^*\in\arg\max_{\pi_R}\mathbb E\left[\sum_{t=0}^{T-1}\gamma^tR_t\right],
\qquad \pi_R(a_R\mid\tau_R)=\prod_{i\in\mathcal R}\pi_i(a_i\mid\tau_i).
\]

The stopping time T is natural completion (possibly infinite); H limits sampled rollouts. A return observed at sampling truncation is partial, so a continuing-task critic retains the successor value. Treating H as a finite objective boundary would define a different learning task and require different target semantics. The environment does not prescribe an optimizer or discount factor. Centralized training may use privileged information; decentralized execution retains the actual local inputs.

## Exact observation and critic features

For every other living entity \(j\), agent \(i\) receives the eleven-value row

\[
e_{ij}=\left[(p_j-p_i)/L,(v_j-v_i)/(2V_{\max}),h_j,1,
\mathbf1_{c_j=R},\mathbf1_{c_j=B},\mathbf1_{j\in\mathcal P}\right],
\quad L=\sqrt{\sum_a(b_a^+-b_a^-)^2}.
\]

The velocity denominator is twice `effective_config.preset.max_speed`: **240, 60 and 40 m/s** for particle, fixed wing and quadrotor respectively. It is not twice `reference_speed`. It normalizes observations; it does not clamp rigid-body velocity.

Rows follow fixed Red Scout/Disturb/Attack, Blue Scout/Disturb/Attack, target order, excluding self. Dead entity rows are zero. The side flags do not encode role. All living entities appear regardless of Scout sectors; every role uses the same packing. There is no sensor noise, so \(Z_i(o\mid s)=\delta_{\mathrm{encode}_i(s)}(o)\).

| Interface | Actor input |
| --- | --- |
| Particle acceleration | \((M-1)\times11\) entity table |
| Particle position | Entity table plus raw six-value `[p,v]` self state |
| Any rigid-body control | Entity table plus raw thirteen-value `[p,v,q,omega]` self state |

Own health is absent from the self block. Other UAV attitudes/rates, policy memory and time are absent from actor rows. An already-dead Damage observer receives zeros for its entire observation, including the self block. Survival retires dead actors after their final transition.

`env.state()` is a privileged feature vector \(g(s)\), not a complete Markov state. Its base packing is eleven values per entity:

\[
[(p-b^-)/(b^+-b^-),v/(2V_{\max}),h,\ell,R,B,P].
\]

Dead blocks are zero. Particle state has length \(11M\), omitting the clock and cumulative damage. Rigid models append all UAV raw thirteen-value states (dead zero), then `[cycle,total_target_damage,is_damage]`, giving \(11M+13N+3\). Opponent memory and grouping commitments still require explicit augmentation. A recurrent critic or augmented state can be appropriate when the packed feature is insufficient.

## Geometry and reset distribution

Scene scales are \(\lambda=1,.4,.16\) for particle, fixed wing and quadrotor. Default bounds are \(\lambda([-2500,2500]^2\times[0,2500])\) metres. Assets are independent uniform samples from \(\lambda([-2300,-1900]\times[-1200,1200]\times[500,1500])\), or a supplied fixed layout. Red samples a target uniformly and an area-uniform XY annulus with radii \([800,2200]\lambda\); 3D XY coordinates are clipped to bounds. Blue samples x from \([0,2500]\lambda\) and y across the scene. UAV altitude is uniform on \([200,1500]\lambda\), clipped to the height bounds.

Particle velocity has a Gaussian unit direction and speed \(\|(u_1,u_2,u_3)\|/\sqrt3\), with independent \(u_a\sim U(35,120)\). Fixed-wing random resets use solved 30 m/s level flight, east for Red and west for Blue. Quadrotors reset at rest with identity quaternion and zero body rate. UAV and planar resets reject openings within \(\max(30\lambda,20\lambda+10^{-6})\) of already placed entities, including targets; ordinary 3D particles do not apply this rejection. Subsequent collisions remain possible. Uniform initialization is a separate display/experiment layout.

Native seedless resets advance the instance RNG. The grouping environment instead retains its saved seed and reproduces that seeded initialization when reset without a new seed.

Planar particles retain three coordinate slots, with fixed \(p_z=100\lambda\) by default and \(v_z=\alpha_z=0\). They use planar heading/speed rules, nine acceleration primitives, and two-dimensional spawn rejection. Rigid models require 3D. Scene scaling changes task geometry and interaction distances, never aircraft mass, inertia, wing size or rotor size.

## Particle transition and controls

For \(\Delta=1\) s, a living particle first moves using old velocity:

\[
p^*=p+\Delta v,\qquad p'=\Pi_{\mathcal W}(p^*),\qquad
w=v+\Delta\alpha,\qquad \widetilde v=\mathcal S_{35,120}(w;v).
\]

For \(\|w\|\ge10^{-3}\), \(\mathcal S_{m,V}(w;v)=w\,\mathrm{clip}(\|w\|,m,V)/\|w\|\). Otherwise use \(mv/\|v\|\) if \(\|v\|\ge10^{-3}\), or \(me_1\). This cancellation fallback preserves the previous heading. Remove outward velocity on each axis clipped during the position update, then apply the turn restriction. The default turn cap is \(\pi\) per step; a lower cap rotates the old heading toward the new one while preserving candidate speed, using a deterministic orthogonal axis in the antiparallel case. Boundary projection can reduce speed below the free-flight minimum.

There is no \(\tfrac12\alpha\Delta^2\) position term. Continuous acceleration actions lie in \([-1,1]^3\) and set \(\alpha=40a\). Discrete actions are zero followed by normalized nonzero directions from \(\{-1,0,1\}^3\). Thus diagonal continuous inputs may have norm \(40\sqrt3\), whereas each nonzero discrete intent has norm 40.

An absolute position command \(p_d\) uses

\[
v_d=\mathrm{sat}_{120}(.5(p_d-p)),\qquad
\alpha=\mathrm{sat}_{40}(2(v_d-v)),
\]

where \(\mathrm{sat}_L(z)=z\min(1,L/\|z\|)\), extended by zero at \(z=0\). It does not teleport, and the particle's minimum-speed map does not promise stopping at the point.

## Rigid-body transition

World coordinates are east-north-up (ENU); body coordinates are forward-left-up (FLU). The scalar-first unit quaternion maps body to world. The thirteen stored values represent twelve physical degrees of freedom since \(q\in S^3\) and \(q\sim-q\). With body force excluding gravity,

\[
\dot p=v,\quad \dot v=R(q)F_B/m-ge_3,\quad
\dot q=\tfrac12q\otimes(0,\omega),\quad
\dot\omega=J^{-1}(M_B-\omega\times J\omega),\qquad g=9.81.
\]

RK4 uses \(n=\lceil\Delta/.01\rceil\) equal substeps, hence 100 substeps for a native step. For feedback law \(U\), stages are \(k_1=f(x,U(x))\), \(k_2=f(x+hk_1/2,U(x+hk_1/2))\), \(k_3=f(x+hk_2/2,U(x+hk_2/2))\), \(k_4=f(x+hk_3,U(x+hk_3))\). Update \(x^+=x+h(k_1+2k_2+2k_3+k_4)/6\), then normalize the quaternion. Derivatives also use normalized quaternions.

Direct actuators are constant over the external step. An acceleration intent establishes \(v_d=v_{start}+\Delta\alpha\) once; velocity feedback uses this fixed reference at every RK4 stage. Position feedback recomputes references from the held absolute destination and the current stage state. It never recomputes acceleration intent as an increment from each stage velocity.

Dynamics do not clamp physical attitude or speed to controller references. The world separately clips predicted substep positions for collision tests and clips final position, removing outward final velocity on clipped axes. Integrator states inside the substeps are not projected back onto the scene. There is no ground-impact crash rule, wind, sensor noise or actuator lag.

### Fixed wing

Physical parameters come from the [BYU MAGICC Aerosonde parameter set](https://github.com/byu-magicc/mavsim_public/blob/main/mavsim_python/parameters/aerosonde_parameters.py); the project independently implements the force and controller equations. The FRD-to-FLU rotation is \(C_f=\mathrm{diag}(1,-1,-1)\), with \(J_{FLU}=C_fJ_{FRD}C_f^T\).

| Parameter | Value |
| --- | --- |
| Mass | 11 kg |
| FRD inertia | `[[.8244,0,-.1204],[0,1.135,0],[-.1204,0,1.759]]` kg m² |
| Wing area, span, chord | .55 m², 2.8956 m, .18994 m |
| Density, Oswald efficiency | 1.2682 kg/m³, .9 |
| Propeller diameter | .508 m |
| Motor resistance, idle current, maximum voltage | .042 ohm, 1.5 A, 44.4 V |
| Motor constant | `60/(145*2*pi)` |
| Propeller thrust coefficients | .09357, -.06044, -.1079 |
| Propeller torque coefficients | .005230, .004970, -.01664 |

FRD body airflow is \((u,v_b,w)=C_fR^Tv\), with \(V=\|(u,v_b,w)\|\), \(\alpha=\mathrm{atan2}(w,u)\), \(\beta=\arcsin(v_b/\max(V,10^{-6}))\). Dynamic pressure is \(\rho V^2/2\). Lift blends linear \(.23+5.61\alpha\) with flat-plate \(2\operatorname{sign}(\alpha)\sin^2\alpha\cos\alpha\), using the smooth stall factors with \(M=50\) and \(\alpha_0=.47\). Drag uses \(.043+(.23+5.61\alpha)^2/(\pi(.9)b^2/S)+.0135\delta_e\). The PDF gives every rate, lateral and control coefficient, force decomposition, and steady motor/propeller polynomial.

Direct actions are `[elevator,aileron,rudder,throttle]` in \([-1,1]^4\), with surfaces \(25^\circ a_{1:3}\), throttle \((a_4+1)/2\). Zero input is half throttle with neutral surfaces, not trim. Trim solves all three translational and three rotational equilibrium equations, including propeller reaction torque; the six unknowns are pitch, heading, three surfaces and throttle. Reset uses the 30 m/s solution rotated to the requested heading.

Acceleration intent is \(5a\) m/s². Position guidance commands course toward the target and flight path \(\mathrm{atan2}(d_z,\max(\|d_{xy}\|,30))\) at 30 m/s. Reference speed is limited to 18-30 m/s, bank/pitch/path to 45/20/15 degrees. The controller schedules the complete feasible trim with requested speed, then applies course-to-bank, pitch, sideslip/rate and speed feedback. The PDF states the exact gains and signs. These gains and bounds are project choices, not the full BYU autopilot. The aircraft flies toward and through a point and cannot hover there.

### Quadrotor

The rigid-body parameters and attitude gains follow [Lee, Leok and McClamroch, arXiv:1003.2005v2, Sections II and VII](https://arxiv.org/pdf/1003.2005v2), with axis conventions rotated to ENU/FLU. Mass is 4.34 kg, \(J=\mathrm{diag}(.0820,.0845,.1377)\) kg m², arm length \(l=.315\) m and yaw ratio \(c_\tau=.008004\) m. The thrust-to-weight ratio 2.5 is a project assumption.

In front/right/rear/left order, \(u_j\in[0,1]\), \(f_j=f_{max}u_j\), \(f_{max}=2.5mg/4\), and

\[
\begin{bmatrix}f\\M_x\\M_y\\M_z\end{bmatrix}
=A\begin{bmatrix}f_1\\f_2\\f_3\\f_4\end{bmatrix},\qquad
A=\begin{bmatrix}1&1&1&1\\0&-l&0&l\\-l&0&l&0\\c_\tau&-c_\tau&c_\tau&-c_\tau\end{bmatrix}.
\]

Body force is \((0,0,f)\). Direct zero action means zero thrust; level hover requires each \(u_j=.4\). Acceleration intent uses scale 6 m/s² and held velocity reference, capped to 20 m/s. Position guidance sets \(v_d=\mathrm{sat}_{12}(.5(p_d-p))\). At each stage,

\[
a_d=\mathrm{sat}_6(2(v_d-v)),\quad F_d=m(a_d+ge_3),\quad
b_{3d}=F_d/\|F_d\|,\quad b_{2d}=(b_{3d}\times e_1)/\|b_{3d}\times e_1\|,
\]
\[
R_d=[b_{2d}\times b_{3d},b_{2d},b_{3d}],\quad
e_R=\tfrac12(R_d^TR-R^TR_d)^\vee,\quad
M_d=-8.81e_R-2.54\omega+\omega\times J\omega,\quad f_d=F_d^TRe_3.
\]

Fixed east heading avoids hidden yaw memory. Since the commanded acceleration norm is below gravity, the thrust/heading construction is nonsingular. This is attitude PD with gyro compensation; it omits trajectory derivatives and desired-rate feedforward from the full source controller.

The allocator clips collective thrust, divides it equally into \(\bar f\), and finds moment differential \(d=A^{-1}(0,M_d)^T\). It chooses the largest \(\kappa\in[0,1]\) satisfying \(0\le\bar f+\kappa d_j\le f_{max}\) for all rotors. Thus it preserves feasible collective thrust while uniformly reducing moment demand. It does not assume that independent rotor clipping preserves the requested total. Rotor drag, gyroscopic rotor effects and motor lag are omitted; [Faessler et al., RAL 2018](https://rpg.ifi.uzh.ch/docs/RAL18_Faessler.pdf) provides the corresponding drag-inclusive context.

## Collision, automatic interaction and event timing

Predict living UAV motion before applying functional actions. Particle collision paths are the unprojected line \(p+\tau\Delta v\); rigid paths use synchronized, clipped RK4 endpoint segments. For a matched segment pair, write \(r=p_i-p_j\), \(d=(q_i-p_i)-(q_j-p_j)\). Closest simultaneous separation is \(\|r+\tau_*d\|\), where \(\tau_*=\mathrm{clip}(-r^Td/\|d\|^2,0,1)\), or zero when \(\|d\|^2\le10^{-12}\). The same time parameter is used for both vehicles. Distance at most \(d_c=20\lambda\) kills both, including friendly pairs; assets are excluded.

The traversal is i-major with inner indices j<i. It checks the outer agent's health once on entry and skips dead inner agents. **An outer agent killed within its inner sweep can still collide with another inner agent during that sweep.** Collision victims do not execute their predicted motion.

After collision, living Blue Attack fires automatically when any living asset is strictly closer than `fire_range`; Red Attack fires when a living Blue Attack is strictly closer. Native Red triggering is specific to Blue Attack, even though explosion damage can affect any enemy UAV role. There is no separately chosen native firing action. Default `fire_range` is the full-damage radius and may be configured within it.

For 3D, \((r_0,r_1)=(300,600)\lambda\), and for planar particles \((200,400)\lambda\). Unit attack intensity gives

\[
\eta(d)=\begin{cases}1&d<r_0,\\(r_1-d)/(r_1-r_0)&r_0\le d<r_1,\\0&d\ge r_1.\end{cases}
\]

All firing decisions and damage use the common post-collision, pre-motion snapshot. Loss at asset k is \(L_k=\sum_{b\in\mathcal B_A:h_b>0}I_b\eta(\|p_b-p_k\|)\). UAV attack losses sum firing enemy Attack sources; friendly explosions are excluded. One source can damage multiple enemies and multiple source contributions add before HP clipping. A firing attacker self-destructs after this step's motion/combat. Other combat victims also complete their motion; their velocity is zeroed on the next inactive update.

Disturb uses a forward sector of radius \(600\lambda\) and half-angle \(\pi/6\), with

\[
G_{ji}=\mathbf1_{10^{-3}<d_{ji}<600\lambda,\;\theta_{ji}<\pi/6}
\exp[-(d_{ji}/\lambda)^2/2]/\sqrt{2\pi}.
\]

The angle is zero when either direction norm is below \(10^{-3}\). Let \(S_c(j)\) sum \(G_{ji}\) over living UAVs of side c. Red Disturb activates if \(S_R<S_B\); Blue if \(S_R\ge S_B\). Disturb damage is \(.1\sum_jJ_jG_{ji}\) and affects either team's UAVs, never assets. Scout has radius \(2000\lambda\), half-angle \(\pi/3\); its functional sector does not filter native actor observations.

Detailed attack events label damage `source_unclipped` and `simultaneous=True`. These contributions are not sequential health decrements; simultaneous overkill remains in raw damage. Disabling event recording does not disable damage accounting. Replay position interpolation does not change the physical state, clock or score.

## Survival and Damage tasks

Survival updates \(h_k'=\max(0,h_k-L_k)\) and \(D_k'=D_k+L_k\). Its Red outcome is -1 if any asset HP is below \(10^{-3}\), otherwise +1 if every Blue Attack HP is below \(10^{-3}\), otherwise zero. Target loss has priority. Default team reward is \(R_R=10z\), \(R_B=-10z\). Red elimination alone does not naturally terminate the world. Individual nonpositive health retires a Survival actor after its final transition.

Explicit `reward_weights` replaces the Survival scalar by a dot product with `[hit,target,enemy,episode]`. For a living Attack agent, hit is the sum of \(\max(0,1-d/r_1)\) over living enemy UAVs, target is negative nearest live-asset distance divided by L, and enemy is negative nearest Blue distance/L for Red or nearest Red distance/L minus 1 for Blue. Empty sets give zero, including Blue's enemy slot. These are post-step geometric proxies, not recorded explosion damage. Scout/Disturb use their role scalar in the first slot and zero in the next two; dead agents have zero first three slots. The episode slot already contains the signed team terminal reward. The PDF specifies the role scalar and boundary penalty in full. Damage rejects these weights.

Damage keeps every asset at finite reset health \(h_k^0\), while accumulating raw \(D_k'=D_k+L_k\). Its per-step team reward and natural completion are

\[
d_t=\sum_kL_k,\qquad R_R=-d_t,\quad R_B=d_t,\qquad
\mathrm{done}=\mathbf1_{\forall b\in\mathcal B_A:h_b'\le0}.
\]

There is no health budget, team-size normalization, asset-count normalization or Survival bonus. Every team member receives the full shared team reward, including dead Damage slots and damage occurring after Red extinction. Summing teammates' duplicate rewards would change the objective. At \(\gamma=1\), team returns are \((-\sum_kD_k,+\sum_kD_k)\); a caller may instead optimize a discounted task.

Native horizon H is sampling truncation, not a victory. The team bootstrap mask is \(b_t=1-\mathbf1_{\text{natural completion}}\), including one at truncation and after individual death. Natural completion takes priority if it coincides with H. A CTDE target is \(y_t=R_t+\gamma b_tQ^-(g(s'),a_R')\), with target-policy successor actions. Actor losses use participation before the transition, so a lethal final action remains trainable. Entity masks use 1 for living/valid entries; participation, actor retirement and global completion express different facts.

## General grouping/SMDP extension

Grouping assigns defenders to protected targets and reserve; it is an acceleration-only execution extension. Built-in fixed opposing policies are `reactive`, `balanced` and `concentrated`, all with lower-level `rush` execution. Balanced divides Blue by target quotas; concentrated selects the minimum mean-travel target. Reactive samples a target with probabilities proportional to \(\exp(-2\,\mathrm{coverage}-0.25\,\mathrm{travel})\), where coverage sums living Red health times \(\exp(-d/(850\lambda))\) and travel is mean Blue distance divided by \(5000\lambda\). These complete rules are separate from externally supplied frozen learned policies.

A macro transition lasts d physical steps until a command opportunity, casualty or completion. If Red has no living member, the current call continues Blue scheduling through natural completion or the horizon and sets `no_red_continuation=True`; it does not return intermediate command/casualty decisions for an empty Red team.

For a physical-step discount \(\gamma\), the exact SMDP target is

\[
\bar R_t=\sum_{j=0}^{d-1}\gamma^jR_{t+j},\qquad
y_t=\bar R_t+\gamma^db_tV(s_{t+d}).
\]

Current grouping Damage reward is the undiscounted sum, directly matching \(\gamma=1\). Changing only the bootstrap to \(\gamma^d\) does not discount internal rewards; a discounted physical objective must reconstruct those rewards or explicitly choose a different macro objective. Grouping Survival uses terminal success 0/1 and its configured horizon rule (`red_win`, `draw` or `blue_win`), unlike native ±10 with horizon truncation. Grouping Damage retains sampling truncation and its associated bootstrap. These extension conventions do not modify the sixteen native themes.

## Implementation correspondence and editing

The v4 public simulation is `env.simulation`, and `Simulation.entities` owns the stable entity sequence. The relevant flat package modules are:

| Concern | Source |
| --- | --- |
| Configuration and model scales | [had_env/config.py](../had_env/config.py) |
| Reset law | [had_env/initialization.py](../had_env/initialization.py) |
| Synchronized collisions/combat | [had_env/world.py](../had_env/world.py) |
| Entity movement and health | [had_env/agents/base.py](../had_env/agents/base.py) |
| Attack, Scout and Disturb rules | [had_env/agents](../had_env/agents) |
| Geometry and snapshot damage | [had_env/geometry.py](../had_env/geometry.py) |
| Rigid RK4, fixed wing, quadrotor | [had_env/dynamics](../had_env/dynamics) |
| Observation and critic packing | [had_env/observations.py](../had_env/observations.py), [had_env/environment.py](../had_env/environment.py) |
| Task rewards and completion | [had_env/tasks.py](../had_env/tasks.py) |
| Grouping and opponent laws | [had_env/grouping](../had_env/grouping) |

The source is authoritative where a mathematical abbreviation omits an implementation detail. The reference article supplied for this documentation was used only for the scenario/dynamics/game structure; its alternative particle position and health equations were not copied. The PDF uses native v4 semantics and does not describe removed algorithm adapters.

To rebuild, run `tectonic --keep-logs --outdir <external-build-directory> docs/PROBLEM_FORMULATION.tex`, inspect all rendered pages, then copy the resulting PDF into `docs/PROBLEM_FORMULATION.pdf`. Keep build auxiliaries and the local reference article outside the deliverable directory.
