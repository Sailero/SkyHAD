"""Independently authored Newton-Euler dynamics in ENU and body FLU.

State is [world position 3, world velocity 3, scalar-first body-to-world
quaternion 4, body angular velocity 3]. Forces exclude gravity. The ideal
models have no wind, actuator lag or artificial state clipping.
"""
import numpy as np

from .control import quaternion_matrix


class RigidBodyDynamics:
    gravity = 9.81
    max_acceleration = 6.
    acceleration_limit = 6.
    actuator_low = -1.
    actuator_high = 1.

    def __init__(self, mass, inertia):
        self.mass = float(mass)
        self.inertia = np.asarray(inertia, dtype=np.float64)
        self.inverse_inertia = np.linalg.inv(self.inertia)

    def initial_state(self, position, velocity):
        state = np.zeros(13, dtype=np.float64)
        state[:3] = self._vector(position, 3, "position")
        state[3:6] = self._vector(velocity, 3, "velocity")
        state[6] = 1.
        return state

    @staticmethod
    def _vector(value, length, name):
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (length,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a finite vector of length {length}")
        return array

    def derivative(self, state, actuator):
        state = np.asarray(state, dtype=np.float64)
        quaternion = state[6:10] / np.linalg.norm(state[6:10])
        rotation = quaternion_matrix(quaternion)
        omega = state[10:13]
        force, moment = self.forces_and_moments(state, actuator)
        derivative = np.empty(13, dtype=np.float64)
        derivative[:3] = state[3:6]
        derivative[3:6] = rotation @ force / self.mass + [0., 0., -self.gravity]
        w, x, y, z = quaternion
        p, q, r = omega
        derivative[6:10] = .5 * np.array([
            -x*p-y*q-z*r, w*p+y*r-z*q, w*q+z*p-x*r, w*r+x*q-y*p])
        derivative[10:13] = self.inverse_inertia @ (moment - np.cross(omega, self.inertia @ omega))
        return derivative

    def _integrate(self, state, command, dt, substep):
        current = self._vector(state, 13, "state").copy()
        qnorm = np.linalg.norm(current[6:10])
        if qnorm < 1e-12:
            raise ValueError("state quaternion must be nonzero")
        current[6:10] /= qnorm
        if not np.isfinite(dt) or dt < 0. or not np.isfinite(substep) or substep <= 0.:
            raise ValueError("dt must be nonnegative and substep positive")
        count = int(np.ceil(dt / substep))
        path = np.empty((count + 1, 3), dtype=np.float64)
        path[0] = current[:3]
        if count == 0:
            return current, path
        step = dt / count
        for index in range(count):
            k1 = self.derivative(current, command(current))
            s2 = current + .5 * step * k1
            k2 = self.derivative(s2, command(s2))
            s3 = current + .5 * step * k2
            k3 = self.derivative(s3, command(s3))
            s4 = current + step * k3
            k4 = self.derivative(s4, command(s4))
            current += step / 6. * (k1 + 2*k2 + 2*k3 + k4)
            current[6:10] /= np.linalg.norm(current[6:10])
            path[index + 1] = current[:3]
        return current, path

    def integrate(self, state, actuator, dt=1., substep=.01):
        """Hold physical actuators for dt seconds; include every substep in path."""
        actuator = self._vector(actuator, 4, "actuator")
        actuator = np.clip(actuator, self.actuator_low, self.actuator_high)
        return self._integrate(state, lambda _: actuator, dt, substep)

    def acceleration_to_actuator(self, state, acceleration, dt=1.):
        desired = np.asarray(state)[3:6] + self._vector(acceleration, 3, "acceleration") * dt
        return self.velocity_to_actuator(state, desired)

    def advance(self, state, action, action_type, dt=1., substep=.01):
        """Hold an intent, recomputing feedback at every RK4 stage.

        Acceleration intent defines a velocity target once, v_start + a*dt.
        Position intent is an absolute world point. Controller state is entirely
        contained in the 13-vector, so snapshot replay needs no hidden integrals.
        """
        if action_type in ("actuator", "raw"):
            return self.integrate(state, action, dt, substep)
        action = self._vector(action, 3, "action").copy()
        if action_type in ("acc", "acceleration"):
            velocity = np.asarray(state)[3:6] + action * dt
            return self._integrate(state, lambda stage: self.velocity_to_actuator(stage, velocity), dt, substep)
        if action_type in ("pos", "position"):
            return self._integrate(state, lambda stage: self.position_to_actuator(stage, action), dt, substep)
        raise ValueError("action_type must be acceleration, position or actuator")
