"""Ideal quadrotor using Lee, Leok and McClamroch's rigid-body model.

Parameters and geometric attitude PD gains: arXiv:1003.2005v2, section VII
https://arxiv.org/pdf/1003.2005v2 . Its down-axis convention is rotated to FLU.
The ENU equations also match Faessler et al., RAL 2018, equations (1)-(4)
with drag, gyroscopic rotor effects and motor lag set to zero:
https://rpg.ifi.uzh.ch/docs/RAL18_Faessler.pdf . These are documented ideal
assumptions, rather than an identified high-speed quadrotor model.

Rotor order is front, right, rear, left. Inputs are thrust fractions [0,1].
Maximum thrust/weight ratio 2.5 is a simulation choice, not a paper parameter.
"""
import numpy as np

from .control import clip_norm, quaternion_matrix, vee
from .rigid_body import RigidBodyDynamics


class QuadrotorDynamics(RigidBodyDynamics):
    max_acceleration = acceleration_limit = 6.
    max_speed = 12.
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
        self.allocation = np.array([[1., 1., 1., 1.], [0., -arm, 0., arm],
                                    [-arm, 0., arm, 0.], [yaw, -yaw, yaw, -yaw]])
        self.inverse_allocation = np.linalg.inv(self.allocation)
        self.hover_actuator = np.full(4, 1. / self.thrust_to_weight)

    def forces_and_moments(self, state, actuator):
        thrusts = np.clip(actuator, 0., 1.) * self.max_rotor_thrust
        wrench = self.allocation @ thrusts
        return np.array([0., 0., wrench[0]]), wrench[1:]

    def allocate(self, collective, moment):
        """Preserve feasible collective, scaling torque uniformly into bounds."""
        collective = float(np.clip(collective, 0., 4. * self.max_rotor_thrust))
        baseline = collective / 4.
        differential = self.inverse_allocation @ np.r_[0., moment]
        scale = 1.
        for value in differential:
            if value > 0.:
                scale = min(scale, (self.max_rotor_thrust-baseline) / value)
            elif value < 0.:
                scale = min(scale, -baseline / value)
        return np.clip((baseline + scale*differential) / self.max_rotor_thrust, 0., 1.)

    def _acceleration_control(self, state, acceleration):
        acceleration = clip_norm(acceleration, self.max_acceleration)
        desired_force = self.mass * (acceleration + [0., 0., self.gravity])
        z_axis = desired_force / np.linalg.norm(desired_force)
        # Fixed world-east heading removes any hidden yaw target from snapshots.
        y_axis = np.cross(z_axis, [1., 0., 0.])
        y_axis /= np.linalg.norm(y_axis)
        desired_rotation = np.column_stack([np.cross(y_axis, z_axis), y_axis, z_axis])
        rotation = quaternion_matrix(state[6:10])
        error = .5 * vee(desired_rotation.T @ rotation - rotation.T @ desired_rotation)
        omega = state[10:13]
        moment = (-self.attitude_gain*error - self.angular_velocity_gain*omega
                  + np.cross(omega, self.inertia @ omega))
        collective = desired_force @ rotation[:, 2]
        return self.allocate(collective, moment)

    def velocity_to_actuator(self, state, velocity):
        velocity = clip_norm(velocity, self.max_speed)
        return self._acceleration_control(state, 2. * (velocity - state[3:6]))

    def position_to_actuator(self, state, position):
        velocity = clip_norm(.5 * (np.asarray(position) - state[:3]), self.max_speed)
        return self.velocity_to_actuator(state, velocity)


Quadrotor = QuadrotorDynamics
