"""Entity ownership and synchronized physical stepping/combat."""
import numpy as np
from had_env.config import AttackIntensity, DisturbIntensity, Interval, initial_health
from had_env.geometry import (
    distance,
    attack_intensity_ratio,
    disturb_intensity_ratio,
    closest_segment_distance,
    WorldKinematics,
)
from had_env.config import CORE_VERSION, PHYSICS_PROTOCOL
from had_env.agents.base import Entity
from had_env.agents.attack import AttackAgent
from had_env.agents.disturb import DisturbAgent
from had_env.agents.scout import ScoutAgent


class World:
    def __init__(self, red_scout_n, red_disturb_n, red_attack_n,
                 blue_scout_n, blue_disturb_n, blue_attack_n,
                 target_n, task_mode='survival', target_health=None,
                 spatial_dim=2, plane_altitude=None, effective_config=None):

        self.red_scout_n = red_scout_n
        self.red_disturb_n = red_disturb_n
        self.red_attack_n = red_attack_n
        self.blue_scout_n = blue_scout_n
        self.blue_disturb_n = blue_disturb_n
        self.blue_attack_n = blue_attack_n
        self.target_n = target_n
        if isinstance(spatial_dim, bool) or spatial_dim not in (2, 3):
            raise ValueError("spatial_dim must be 2 or 3")
        from had_env.config import EnvConfig
        self.effective_config = effective_config or EnvConfig(spatial_dim=spatial_dim,
            task_mode=task_mode, plane_altitude=plane_altitude)
        spatial_dim = self.effective_config.spatial_dim
        task_mode = self.effective_config.task_mode
        self.env_agent_type = self.effective_config.env_agent_type
        self.env_agent_action_type = self.effective_config.env_agent_action_type
        self.scene_scale = self.effective_config.scene_scale
        self.world_bounds = np.asarray(self.effective_config.world_bounds, dtype=float)
        self.collision_distance = self.effective_config.collision_distance
        self.dynamics = None
        if self.env_agent_type != 'particle':
            from had_env.dynamics import FixedWingDynamics, QuadrotorDynamics
            self.dynamics = (FixedWingDynamics if self.env_agent_type == 'UAV_fixedwing' else QuadrotorDynamics)()
        self.spatial_dim = int(spatial_dim)
        self.attack_distance = self.effective_config.attack_distance
        self.fire_range = self.attack_distance[0]
        self.plane_altitude = self.effective_config.plane_altitude
        if not np.isfinite(self.plane_altitude) or not self.world_bounds[2, 0] <= self.plane_altitude <= self.world_bounds[2, 1]:
            raise ValueError("plane_altitude must be finite and inside the world height bounds")
        if task_mode not in ('survival', 'damage'):
            raise ValueError("task_mode must be survival or damage")
        self.task_mode = task_mode
        self.target_initialization = 'random'
        self.target_health = float(initial_health if target_health is None else target_health)
        if not np.isfinite(self.target_health) or self.target_health <= 0:
            raise ValueError("target_health must be finite and positive")

        self.entities = self.create_entities()
        for entity in self.entities:
            entity.spatial_dim = self.spatial_dim
            entity.env_agent_type = self.env_agent_type
            entity.env_agent_action_type = self.env_agent_action_type
            entity.world_bounds = self.world_bounds
            entity.scene_scale = self.scene_scale
            entity.dynamics = self.dynamics if entity.Type != 'Entity' else None
            entity.vMin = self.effective_config.preset.min_speed
            entity.vMax = self.effective_config.preset.max_speed
            entity.aMax = self.effective_config.preset.acceleration_limit
            entity.attack_distance = self.attack_distance
            entity.fire_range = self.fire_range
            entity.plane_altitude = self.plane_altitude
            if self.spatial_dim == 2:
                entity.position[2] = self.plane_altitude
                entity.pre_position[2] = self.plane_altitude
        self.red_agent_n = self.red_disturb_n + self.red_attack_n + self.red_scout_n
        self.blue_agent_n = self.blue_disturb_n + self.blue_attack_n + self.blue_scout_n

        self.agents = [agent for agent in self.entities if agent.Type != 'Entity']
        self.targets = [entity for entity in self.entities if entity.Type == 'Entity']
        self.red_agents = [agent for agent in self.agents if agent.Color == "Red"]
        self.blue_agents = [agent for agent in self.agents if agent.Color == "Blue"]
        self.update_alive_agents()
        self.core_version = CORE_VERSION
        self.physics_protocol = PHYSICS_PROTOCOL
        self.record_events = False
        self.physics_step_count = 0
        self.last_physics_events = []
        color_map = {"Red": 0, "Blue": 1, "Entity": 2}
        type_map = {"Entity": 0, "Attack": 1, "Disturb": 2, "Scout": 3}
        self._color_code = np.asarray([color_map[entity.Color] for entity in self.entities], dtype=np.int8)
        self._type_code = np.asarray([type_map[entity.Type] for entity in self.entities], dtype=np.int8)
        self.entity_mask = None
        self.boundary_clips = 0

    def reset_episode_state(self):
        """Refresh derived state after every native random or uniform reset."""
        for target in self.targets:
            target.step_damage = 0.0
            target.cumulative_damage = 0.0
        self.update_alive_agents()
        self.physics_step_count = 0
        self.last_physics_events = []
        self.boundary_clips = 0
        self.entity_mask = None
        if hasattr(self, "_render_closed"):
            self._render_closed = False

    def update_alive_agents(self):
        self.alive_agents = [agent for agent in self.agents if agent.Health > 0]
        self.alive_targets = [target for target in self.targets if target.Health > 0]

    @property
    def step_target_damage(self):
        return float(sum(target.step_damage for target in self.targets))

    @property
    def target_damage(self):
        return float(sum(target.cumulative_damage for target in self.targets))


    def create_entities(self):
        """Stable order: Red scout/disturb/attack, Blue roles, then targets."""
        entities = []
        entity_id = 0
        for color, counts in (
            ('Red', (self.red_scout_n, self.red_disturb_n, self.red_attack_n)),
            ('Blue', (self.blue_scout_n, self.blue_disturb_n, self.blue_attack_n)),
        ):
            render_id = 0
            for agent_class, count in zip((ScoutAgent, DisturbAgent, AttackAgent), counts):
                for _ in range(count):
                    entities.append(agent_class(color, entity_id, render_id))
                    entity_id += 1
                    render_id += 1
        for render_id in range(self.target_n):
            entities.append(Entity(entity_id, render_id, task_mode=self.task_mode,
                                   target_health=self.target_health))
            entity_id += 1
        return entities


    def step(self, flying_action_n):
        if self.dynamics is not None:
            # Predict all synchronized paths before collisions/fire mutate health.
            paths = {}
            for agent, command in zip(self.agents, flying_action_n):
                if agent.Health <= 0:
                    continue
                candidate, path = self.dynamics.advance(agent.rigid_state, command,
                                                        self.env_agent_action_type, dt=Interval)
                clipped_path = np.clip(path, self.world_bounds[:, 0], self.world_bounds[:, 1])
                axes = []
                for axis in range(3):
                    if candidate[axis] > self.world_bounds[axis, 1]:
                        candidate[axis] = self.world_bounds[axis, 1]
                        candidate[3+axis] = min(0., candidate[3+axis])
                        axes.append((axis, 1))
                    elif candidate[axis] < self.world_bounds[axis, 0]:
                        candidate[axis] = self.world_bounds[axis, 0]
                        candidate[3+axis] = max(0., candidate[3+axis])
                        axes.append((axis, -1))
                agent._next_rigid_state = candidate
                agent._next_clamped_axes = axes
                paths[agent.Id] = clipped_path
        self.physics_step_count += 1
        self.last_physics_events = []
        self.boundary_clips = 0
        # Swept collision on the upcoming displacement. Skip agents already
        # killed earlier in this same i-major/j-minor pass.
        for i, first in enumerate(self.alive_agents):
            if first.Health <= 0:
                continue
            p1 = np.asarray(first.position, dtype=np.float64)
            v1 = np.asarray(first.velocity, dtype=np.float64)
            q1 = p1 + v1 * Interval
            for j in range(i):
                second = self.alive_agents[j]
                if second.Health <= 0:
                    continue
                p2 = np.asarray(second.position, dtype=np.float64)
                v2 = np.asarray(second.velocity, dtype=np.float64)
                if self.dynamics is None and float(np.linalg.norm(p1 - p2)) > self.collision_distance + float(np.linalg.norm(v1 - v2)) * Interval:
                    continue
                q2 = p2 + v2 * Interval
                if self.dynamics is None:
                    separation = closest_segment_distance(p1, q1, p2, q2)
                else:
                    path1, path2 = paths[first.Id], paths[second.Id]
                    separation = min(closest_segment_distance(path1[k], path1[k+1], path2[k], path2[k+1])
                                     for k in range(len(path1)-1))
                if separation <= self.collision_distance:
                    if self.record_events:
                        for source, target in ((first, second), (second, first)):
                            if target.Health > 0:
                                self._record_physics_event(
                                    "collision", source, target, target.Health, 0,
                                    target.Health, separation,
                                )
                    first.Health = 0
                    second.Health = 0

        self._apply_automatic_fire(flying_action_n)

        snapshot = WorldKinematics(self.entities, self._color_code, self._type_code,
                                   attack_distance=self.attack_distance)
        for agent in self.entities:
            health_before = float(agent.Health) if self.record_events else None
            agent.update_status(snapshot)
            if getattr(agent, "_clamped_axes", None):
                self.boundary_clips += len(agent._clamped_axes)
            if self.record_events and health_before is not None and health_before > 0:
                self._record_damage_events(agent, snapshot, health_before)

        self.update_alive_agents()

    def _apply_automatic_fire(self, flying_action_n):
        positions = np.asarray([entity.position for entity in self.entities], dtype=np.float64)
        health = np.asarray([entity.Health for entity in self.entities], dtype=np.float64)
        fire_range = self.fire_range
        for i, agent in enumerate(self.agents):
            agent.set_flying_action(flying_action_n[i])
            if agent.Type != "Attack":
                agent.set_function_action(agent.choose_function_ruled_action(self.entities))
                continue
            fire = False
            if agent.Health > 0:
                origin = positions[i]
                if agent.Color == "Blue":
                    mask = (self._type_code == 0) & (health > 0)
                else:
                    mask = (self._color_code == 1) & (self._type_code == 1) & (health > 0)
                if np.any(mask):
                    fire = bool(np.min(np.linalg.norm(positions[mask] - origin, axis=1)) < fire_range)
            agent.set_function_action(fire)
            if self.record_events and agent.Health > 0 and fire:
                self._record_physics_event("fire", agent, None, agent.Health, agent.Health, 0, None)

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
        """Record simultaneous combat using the same pre-update positions as hits."""
        victim_pos = getattr(target, "pre_position", target.position)
        if isinstance(previous_world, WorldKinematics):
            if target.Type == "Entity":
                combat_after = float(target.Health)
            else:
                combat_after = float(np.max([
                    0.0,
                    health_before
                    - previous_world.attack_loss(victim_pos, target.Color, target.Type)
                    - previous_world.disturb_loss(victim_pos),
                ]))
            fire_mask = (previous_world.type_code == 1) & (previous_world.health > 0) & previous_world.is_fire
            if target.Type == "Entity":
                fire_mask &= previous_world.color_code == 1
            else:
                victim_code = 0 if target.Color == "Red" else 1
                fire_mask &= previous_world.color_code != victim_code
            for index in np.flatnonzero(fire_mask):
                source = self.entities[int(index)]
                separation = distance(victim_pos, previous_world.positions[int(index)])
                damage = AttackIntensity * attack_intensity_ratio(float(separation), previous_world.attack_distance)
                if damage > 0:
                    self._record_physics_event(
                        "attack_damage", source, target, health_before, combat_after,
                        damage, separation, damage_scope="source_unclipped", simultaneous=True,
                    )
            if target.Type == "Attack" and getattr(target, "IsFire", False):
                self._record_physics_event(
                    "self_destruct", target, target, combat_after, target.Health,
                    combat_after - float(target.Health), 0,
                )
            return
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
                0, health_before - target.calculate_attack_health_loss(attacks, victim_position=victim_pos)
                - target.calculate_disturb_health_loss(disturbers, victim_position=victim_pos),
            ]))
        for source in attacks:
            separation = distance(victim_pos, source.position)
            damage = AttackIntensity * attack_intensity_ratio(float(separation), self.attack_distance) * source.IsFire
            if damage > 0:
                self._record_physics_event(
                    "attack_damage", source, target, health_before, combat_after,
                    damage, separation, damage_scope="source_unclipped", simultaneous=True,
                )
        for source in disturbers:
            damage = DisturbIntensity * disturb_intensity_ratio(
                source.position, victim_pos, source.velocity,
            ) * source.IsDisturb
            if damage > 0:
                self._record_physics_event(
                    "disturb_damage", source, target, health_before, combat_after,
                    damage, distance(victim_pos, source.position),
                    damage_scope="source_unclipped", simultaneous=True,
                )
        if target.Type == "Attack" and target.IsFire:
            self._record_physics_event(
                "self_destruct", target, target, combat_after, target.Health,
                combat_after - float(target.Health), 0,
            )
