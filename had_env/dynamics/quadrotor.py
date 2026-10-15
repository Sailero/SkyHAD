"""Ideal quadrotor using Lee, Leok and McClamroch's rigid-body model.

Parameters and geometric attitude PD gains: arXiv:1003.2005v2, Section VII
(J, m, d, c_tau_f, k_R, k_Omega). The paper's down-pointing body axis is
rotated to FLU. The ENU equations also match Faessler et al., RA-L 2018,
equations (1)-(4) with rotor drag, gyroscopic rotor effects and motor lag set
to zero. These are documented ideal assumptions, not an identified model.

Rotor order is front, right, rear, left. Inputs are thrust fractions [0, 1].
The maximum thrust/weight ratio 2.5 is a simulation choice.
"""
import numpy as np

from .control import clip_norm, cross, vee
from .rigid_body import RigidBodyDynamics, as_batch


class QuadrotorDynamics(RigidBodyDynamics):
    acceleration_limit = 6.
    max_speed = 20.
    position_speed = 12.
    actuator_low = 0.
    arm_length = .315
    yaw_ratio = .008004
    thrust_to_weight = 2.5
    attitude_gain = 8.81
    angular_velocity_gain = 2.54

    def __init__(self):
        super().__init__(4.34, np.diag([.0820, .0845, .1377]))
        self.max_rotor_thrust = self.thrust_to_weight * self.mass * self.gravity / 4.
        arm, yaw = self.arm_length, self.yaw_ratio
        # Rows: collective thrust and FLU roll, pitch, yaw moments.
        self.allocation = np.array([[1., 1., 1., 1.], [0., -arm, 0., arm],
                                    [-arm, 0., arm, 0.], [yaw, -yaw, yaw, -yaw]])
        self.inverse_allocation = np.linalg.inv(self.allocation)
        self.hover_actuator = np.full(4, 1. / self.thrust_to_weight)

    def _forces(self, states, actuators, rotation):
        wrench = (np.clip(actuators, 0., 1.) * self.max_rotor_thrust) @ self.allocation.T
        force = np.zeros((len(states), 3))
        force[:, 2] = wrench[:, 0]
        return force, wrench[:, 1:]

    def allocate(self, collective, moment):
        """Rotor fractions preserving feasible collective thrust; moments shrink uniformly to fit."""
        moments, single = as_batch(moment, 3, "moment")
        collective = np.clip(collective, 0., 4. * self.max_rotor_thrust)
        baseline = np.reshape(collective, (-1, 1)) / 4.
        differential = moments @ self.inverse_allocation[:, 1:].T
        # Largest kappa in [0, 1] with 0 <= baseline + kappa * differential <= max thrust.
        raising, lowering = differential > 0., differential < 0.
        divisor = np.where(raising | lowering, differential, 1.)
        limits = np.where(raising, (self.max_rotor_thrust - baseline) / divisor,
                          np.where(lowering, -baseline / divisor, np.inf))
        scale = np.minimum(1., limits.min(axis=1, keepdims=True))
        actuators = np.clip((baseline + scale * differential) / self.max_rotor_thrust, 0., 1.)
        return actuators[0] if single else actuators

    def _acceleration_control(self, states, rotation, acceleration):
        """Geometric attitude PD on SO(3) with a fixed east heading (no hidden yaw memory)."""
        desired_force = self.mass * (clip_norm(acceleration, self.acceleration_limit) + [0., 0., self.gravity])
        z_axis = desired_force / np.linalg.norm(desired_force, axis=1, keepdims=True)
        y_axis = cross(z_axis, [1., 0., 0.])
        y_axis /= np.linalg.norm(y_axis, axis=1, keepdims=True)
        desired = np.stack([cross(y_axis, z_axis), y_axis, z_axis], axis=2)
        relative = np.swapaxes(desired, 1, 2) @ rotation
        error = .5 * vee(relative - np.swapaxes(relative, 1, 2))
        omega = states[:, 10:13]
        moment = (-self.attitude_gain * error - self.angular_velocity_gain * omega
                  + cross(omega, omega @ self.inertia.T))
        collective = np.einsum("ni,ni->n", desired_force, rotation[:, :, 2])
        return self.allocate(collective, moment)

    def _velocity_control(self, states, rotation, velocity):
        acceleration = 2. * (clip_norm(velocity, self.max_speed) - states[:, 3:6])
        return self._acceleration_control(states, rotation, acceleration)

    def _position_control(self, states, rotation, position):
        velocity = clip_norm(.5 * (position - states[:, :3]), self.position_speed)
        return self._velocity_control(states, rotation, velocity)


Quadrotor = QuadrotorDynamics
