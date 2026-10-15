"""Newton-Euler rigid-body dynamics in ENU world and FLU body axes.

A state row is [world position 3, world velocity 3, scalar-first body-to-world
quaternion 4, body angular velocity 3]. Model forces exclude gravity. There is
no wind, actuator lag or artificial state clipping.

Public methods accept one state of shape (13,) or a batch of shape (N, 13).
The environment integrates every aircraft in one vectorized RK4 loop; batching
does not couple aircraft, each row evolves independently.
"""
import numpy as np

from .control import cross, quaternion_matrix


def as_batch(value, width, name):
    """Return a finite (N, width) float array and whether the input was a single row."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim not in (1, 2) or array.shape[-1] != width or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of length {width} or a batch of them")
    return array.reshape(-1, width), array.ndim == 1


def _unbatch(single, *arrays):
    if len(arrays) == 1:
        return arrays[0][0] if single else arrays[0]
    return tuple(array[0] for array in arrays) if single else arrays


class RigidBodyDynamics:
    gravity = 9.81
    acceleration_limit = 6.
    actuator_low = -1.
    actuator_high = 1.

    def __init__(self, mass, inertia):
        self.mass = float(mass)
        self.inertia = np.asarray(inertia, dtype=np.float64)
        self.inverse_inertia = np.linalg.inv(self.inertia)

    def initial_state(self, position, velocity):
        state = np.zeros(13, dtype=np.float64)
        state[:3] = as_batch(position, 3, "position")[0][0]
        state[3:6] = as_batch(velocity, 3, "velocity")[0][0]
        state[6] = 1.
        return state

    # Subclasses implement the batched model on (N, 13) stage states and their
    # FLU-to-ENU matrices: _forces(states, actuators, rotation) returns body force
    # (excluding gravity) and body moment; _velocity_control and _position_control
    # return actuator rows.

    def forces_and_moments(self, state, actuator):
        states, single = as_batch(state, 13, "state")
        actuators = np.broadcast_to(as_batch(actuator, 4, "actuator")[0], (len(states), 4))
        return _unbatch(single, *self._forces(states, actuators, quaternion_matrix(states[:, 6:10])))

    def derivative(self, state, actuator):
        states, single = as_batch(state, 13, "state")
        actuators = np.broadcast_to(as_batch(actuator, 4, "actuator")[0], (len(states), 4))
        return _unbatch(single, self._derivative(states, actuators, quaternion_matrix(states[:, 6:10])))

    def _derivative(self, states, actuators, rotation):
        quaternion = states[:, 6:10] / np.linalg.norm(states[:, 6:10], axis=1, keepdims=True)
        omega = states[:, 10:13]
        force, moment = self._forces(states, actuators, rotation)
        derivative = np.empty_like(states)
        derivative[:, :3] = states[:, 3:6]
        derivative[:, 3:6] = (rotation @ force[:, :, None])[:, :, 0] / self.mass
        derivative[:, 5] -= self.gravity
        w, x, y, z = quaternion.T
        p, q, r = omega.T
        derivative[:, 6] = -.5*(x*p+y*q+z*r)
        derivative[:, 7] = .5*(w*p+y*r-z*q)
        derivative[:, 8] = .5*(w*q+z*p-x*r)
        derivative[:, 9] = .5*(w*r+x*q-y*p)
        derivative[:, 10:13] = (moment - cross(omega, omega @ self.inertia.T)) @ self.inverse_inertia.T
        return derivative

    def _integrate(self, states, command, dt, substep):
        """Classic RK4 with ``ceil(dt/substep)`` equal steps; ``command(stage, rotation)`` is feedback."""
        if not np.isfinite(dt) or dt < 0. or not np.isfinite(substep) or substep <= 0.:
            raise ValueError("dt must be nonnegative and substep positive")
        current = states.copy()
        norms = np.linalg.norm(current[:, 6:10], axis=1, keepdims=True)
        if np.any(norms < 1e-12):
            raise ValueError("state quaternion must be nonzero")
        current[:, 6:10] /= norms
        count = int(np.ceil(dt / substep))
        path = np.empty((len(current), count + 1, 3), dtype=np.float64)
        path[:, 0] = current[:, :3]
        step = dt / max(count, 1)

        def slope(stage):
            rotation = quaternion_matrix(stage[:, 6:10])
            return self._derivative(stage, command(stage, rotation), rotation)

        for index in range(count):
            k1 = slope(current)
            k2 = slope(current + .5 * step * k1)
            k3 = slope(current + .5 * step * k2)
            k4 = slope(current + step * k3)
            current += step / 6. * (k1 + 2*k2 + 2*k3 + k4)
            current[:, 6:10] /= np.linalg.norm(current[:, 6:10], axis=1, keepdims=True)
            path[:, index + 1] = current[:, :3]
        return current, path

    def integrate(self, state, actuator, dt=1., substep=.01):
        """Hold physical actuators for ``dt`` seconds; ``path`` holds every substep position."""
        states, single = as_batch(state, 13, "state")
        actuators = np.clip(as_batch(actuator, 4, "actuator")[0], self.actuator_low, self.actuator_high)
        actuators = np.broadcast_to(actuators, (len(states), 4))
        return _unbatch(single, *self._integrate(states, lambda stage, rotation: actuators, dt, substep))

    def advance(self, state, action, action_type, dt=1., substep=.01):
        """Hold one intent for ``dt`` seconds, recomputing feedback at every RK4 stage.

        Acceleration intent fixes the velocity reference once, v_start + a*dt.
        Position intent is an absolute world point. Controllers are stateless, so
        the 13-value state fully determines every continuation.
        """
        if action_type == "actuator":
            return self.integrate(state, action, dt, substep)
        states, single = as_batch(state, 13, "state")
        actions = np.broadcast_to(as_batch(action, 3, "action")[0], (len(states), 3))
        if action_type == "acceleration":
            command = self._velocity_command(states, states[:, 3:6] + actions * dt)
        elif action_type == "position":
            command = self._position_command(actions)
        else:
            raise ValueError("action_type must be acceleration, position or actuator")
        return _unbatch(single, *self._integrate(states, command, dt, substep))

    def _velocity_command(self, states, velocity):
        return lambda stage, rotation: self._velocity_control(stage, rotation, velocity)

    def _position_command(self, position):
        return lambda stage, rotation: self._position_control(stage, rotation, position)

    def velocity_to_actuator(self, state, velocity):
        states, single = as_batch(state, 13, "state")
        velocity = np.broadcast_to(as_batch(velocity, 3, "velocity")[0], (len(states), 3))
        command = self._velocity_command(states, velocity)
        return _unbatch(single, command(states, quaternion_matrix(states[:, 6:10])))

    def position_to_actuator(self, state, position):
        states, single = as_batch(state, 13, "state")
        position = np.broadcast_to(as_batch(position, 3, "position")[0], (len(states), 3))
        return _unbatch(single, self._position_command(position)(states, quaternion_matrix(states[:, 6:10])))

    def acceleration_to_actuator(self, state, acceleration, dt=1.):
        states, single = as_batch(state, 13, "state")
        acceleration = as_batch(acceleration, 3, "acceleration")[0]
        command = self._velocity_command(states, states[:, 3:6] + acceleration * dt)
        return _unbatch(single, command(states, quaternion_matrix(states[:, 6:10])))
