import copy
from had_env.core.config import *
from had_env.core.function.Function import distance, distances_from, attack_intensity_ratio, disturb_intensity_ratio
from had_env.core.version import CORE_VERSION, PHYSICS_PROTOCOL
from had_env.core.agents.base import Entity
from had_env.core.agents.attack import AttackAgent
from had_env.core.agents.disturb import DisturbAgent
from had_env.core.agents.scout import ScoutAgent


class World:
    def __init__(self, red_scout_n, red_disturb_n, red_attack_n,
                 blue_scout_n, blue_disturb_n, blue_attack_n,
                 target_n):
        # 读取世界信息
        self.red_scout_n = red_scout_n
        self.red_disturb_n = red_disturb_n
        self.red_attack_n = red_attack_n
        self.blue_scout_n = blue_scout_n
        self.blue_disturb_n = blue_disturb_n
        self.blue_attack_n = blue_attack_n
        self.target_n = target_n

        # 创建智能体（这里的智能体是否跟决策的智能体保持一致，如何保持顺序一致的问题呢？还是说不需要保持顺序一致呢？）
        self.world = self.create_world()
        self.red_agent_n = self.red_disturb_n + self.red_attack_n + self.red_scout_n
        self.blue_agent_n = self.blue_disturb_n + self.blue_attack_n + self.blue_scout_n

        self.agents = [agent for agent in self.world if agent.Type != 'Entity']
        self.targets = [entity for entity in self.world if entity.Type == 'Entity']
        self.red_agents = [agent for agent in self.agents if agent.Color == "Red"]
        self.blue_agents = [agent for agent in self.agents if agent.Color == "Blue"]
        self.update_alive_agents()
        self.core_version = CORE_VERSION
        self.physics_protocol = PHYSICS_PROTOCOL
        self.record_events = False
        self.physics_step_count = 0
        self.last_physics_events = []

    def reset_episode_state(self):
        """Refresh derived state after every native random or uniform reset."""
        self.update_alive_agents()
        self.physics_step_count = 0
        self.last_physics_events = []
        if hasattr(self, "_render_closed"):
            self._render_closed = False

    def update_alive_agents(self):
        self.alive_agents = [agent for agent in self.agents if agent.Health > 0]
        self.alive_targets = [target for target in self.targets if target.Health > 0]

    def get_agents_dim_info(self):
        n_entities = self.red_agent_n + self.blue_agent_n + self.target_n
        agents_dim_info = {
            "target_n": self.target_n,
            "red_agent_n": self.red_agent_n,
            "blue_agent_n": self.blue_agent_n,
            "red_scout_n": self.red_scout_n,
            "red_disturb_n": self.red_disturb_n,
            "red_attack_n": self.red_attack_n,
            "blue_scout_n": self.blue_scout_n,
            "blue_disturb_n": self.blue_disturb_n,
            "blue_attack_n": self.blue_attack_n,
            # 局部观测维度：相对位置 + 相对速度 + is_alive
            "scout_obs_dim": EnvDim * 2 + 1,
            "disturb_obs_dim": EnvDim * 2 + 1,
            "attack_obs_dim": EnvDim * 2 + 1,
            "scout_action_dim": EnvDim,
            "disturb_action_dim": EnvDim,
            "attack_action_dim": EnvDim,
            # 全局状态维度：每个实体的 [绝对位置 + 绝对速度 + is_alive]
            "global_state_dim": n_entities * (EnvDim * 2 + 1),
        }
        return agents_dim_info

    def create_world(self):
        world = []

        # 按照次序初始化智能体
        Id = 0
        render_id = 0

        # 初始化红方侦查智能体
        for _ in range(self.red_scout_n):
            world.append(ScoutAgent('Red', Id, render_id))
            Id += 1
            render_id += 1

        # 初始化红方软杀伤智能体
        for _ in range(self.red_disturb_n):
            world.append(DisturbAgent('Red', Id, render_id))
            Id += 1
            render_id += 1

        # 初始化红方打击智能体
        for _ in range(self.red_attack_n):
            world.append(AttackAgent('Red', Id, render_id))
            Id += 1
            render_id += 1

        render_id = 0

        # 初始化蓝方侦查智能体
        for _ in range(self.blue_scout_n):
            world.append(ScoutAgent('Blue', Id, render_id))
            Id += 1
            render_id += 1

        # 初始化蓝方软杀伤智能体
        for _ in range(self.blue_disturb_n):
            world.append(DisturbAgent('Blue', Id, render_id))
            Id += 1
            render_id += 1

        # 初始化蓝方打击智能体
        for _ in range(self.blue_attack_n):
            world.append(AttackAgent('Blue', Id, render_id))
            Id += 1
            render_id += 1

        render_id = 0

        # 初始化保护目标点
        for _ in range(self.target_n):
            world.append(Entity(Id, render_id))
            Id += 1
            render_id += 1

        return world

    def get_agents(self):
        return self.agents

    def get_world(self):
        return self.world

    def get_agents_flying_actions(self):
        return [one.get_flying_action() for one in self.agents]

    def get_agents_actions(self):
        return [one.get_action() for one in self.agents]

    def get_status(self):
        return [agent.get_status() for agent in self.world]

    def step(self, flying_action_n):
        self.physics_step_count += 1
        self.last_physics_events = []
        # 碰撞检测
        # Preserve the original i-major/j-minor ordering and cached roster,
        # including entities killed by an earlier pair in this same pass.
        for i, first in enumerate(self.alive_agents):
            separations = distances_from(first.get_position(), [
                other.get_position() for other in self.alive_agents[:i]
            ])
            for j in np.flatnonzero(separations <= AvoidanceDistance):
                second = self.alive_agents[j]
                if self.record_events:
                    for source, target in ((first, second), (second, first)):
                        if target.Health > 0:
                            self._record_physics_event(
                                "collision", source, target, target.Health, 0,
                                target.Health, separations[j],
                            )
                first.Health = 0
                second.Health = 0

        # 智能体根据action_n设定动作
        for i, agent in enumerate(self.agents):
            agent.set_flying_action(flying_action_n[i])
            function_action = agent.choose_function_ruled_action(self.world)
            agent.set_function_action(function_action)
            if self.record_events and agent.Health > 0 and getattr(agent, "IsFire", False):
                self._record_physics_event("fire", agent, None, agent.Health, agent.Health, 0, None)

        # 更新状态
        agents_copy = copy.deepcopy(self.world)
        for agent in self.world:
            health_before = float(agent.Health) if self.record_events else None
            agent.update_status(agents_copy)
            if self.record_events and health_before > 0:
                self._record_damage_events(agent, agents_copy, health_before)

        # 更新还在存活的agent列表
        self.update_alive_agents()

    def _record_physics_event(self, kind, source, target, before, after, damage, separation, **extra):
        self.last_physics_events.append({
            "kind": kind,
            "step": self.physics_step_count,
            "source_id": int(source.Id),
            "target_id": None if target is None else int(target.Id),
            "health_before": float(before),
            "health_after": float(after),
            "damage": float(damage),
            "distance": None if separation is None else float(separation),
            "source_position": [float(value) for value in source.position],
            "target_position": None if target is None else [float(value) for value in target.position],
            **extra,
        })

    def _record_damage_events(self, target, previous_world, health_before):
        """Observe the original old-source/new-victim calculation after update.

        This opt-in diagnostic never writes physical state. All source damage
        events describe the same simultaneous health transition. Their damage
        is unclipped source strength (and can exceed the victim's remaining HP),
        not a sequential allocation of kill credit. Firing self-destruction is
        recorded separately, after ordinary combat damage, as in AttackAgent.
        """
        attacks = [source for source in previous_world if source.Type == "Attack"
                   and source.Health > 0
                   and (source.Color == "Blue" if target.Type == "Entity"
                        else source.Color != target.Color)]
        disturbers = [] if target.Type == "Entity" else [
            source for source in previous_world if source.Type == "Disturb" and source.Health > 0
        ]
        if target.Type == "Entity":
            combat_after = float(target.Health)
        else:
            combat_after = float(np.max([
                0, health_before - target.calculate_attack_health_loss(attacks)
                - target.calculate_disturb_health_loss(disturbers),
            ]))
        for source in attacks:
            separation = distance(target.position, source.position)
            damage = AttackIntensity * attack_intensity_ratio(float(separation)) * source.IsFire
            if damage > 0:
                self._record_physics_event(
                    "attack_damage", source, target, health_before, combat_after,
                    damage, separation, damage_scope="source_unclipped", simultaneous=True,
                )
        for source in disturbers:
            damage = DisturbIntensity * disturb_intensity_ratio(
                source.position, target.position, source.velocity,
            ) * source.IsDisturb
            if damage > 0:
                self._record_physics_event(
                    "disturb_damage", source, target, health_before, combat_after,
                    damage, distance(target.position, source.position),
                    damage_scope="source_unclipped", simultaneous=True,
                )
        if target.Type == "Attack" and target.IsFire:
            self._record_physics_event(
                "self_destruct", target, target, combat_after, target.Health,
                combat_after - float(target.Health), 0,
            )

    def get_red_blue_relative_distance(self):
        red_agents = [agent for agent in self.agents if agent.Color == "Red"]
        blue_agents = [agent for agent in self.agents if agent.Color == "Blue"]
        red_agents_mean_position = np.mean([agent.get_position() for agent in red_agents], axis=0)
        blue_agents_mean_position = np.mean([agent.get_position() for agent in blue_agents], axis=0)
        return np.linalg.norm(red_agents_mean_position - blue_agents_mean_position)

    def get_blue_target_relative_distance(self):
        blue_agents = [agent for agent in self.agents if agent.Color == "Blue"]
        targets_mean_position = np.mean([agent.get_position() for agent in self.targets], axis=0)
        blue_agents_mean_position = np.mean([agent.get_position() for agent in blue_agents], axis=0)
        return np.linalg.norm(targets_mean_position - blue_agents_mean_position)
