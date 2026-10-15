"""Aerosonde six-degree-of-freedom aircraft, independently authored.

Physical, aerodynamic, motor and propeller parameters are BYU MAGICC's
published Aerosonde set (Beard and McLain, Small Unmanned Aircraft):
https://github.com/byu-magicc/mavsim_public/blob/main/mavsim_python/parameters/aerosonde_parameters.py
Forces use the book's nonlinear lift/drag model and McLain's propeller addendum.
Forward-right-down (FRD) aerodynamic vectors are rotated to forward-left-up
(FLU) by the proper rotation diag(1, -1, -1), including moments and inertia.

Inputs are normalized [elevator, aileron, rudder, throttle] in [-1, 1];
surface limits are 25 degrees and throttle = (input + 1) / 2. Guidance limits
are reference commands, never physical state clamps. The position controller
flies toward and through its target; a fixed-wing aircraft cannot hover.
"""
from functools import lru_cache

import numpy as np
from scipy.optimize import root

from .control import flight_angles, matrix_angles, quaternion_from_euler, quaternion_matrix, wrap_angle
from .rigid_body import RigidBodyDynamics

FRD = np.array([1., -1., -1.])  # diag(1, -1, -1) maps FRD <-> FLU.


def body_frd_velocity(states, rotation):
    """Air-relative velocity in FRD body axes (no wind)."""
    return (states[:, None, 3:6] @ rotation)[:, 0] * FRD


class FixedWingDynamics(RigidBodyDynamics):
    acceleration_limit = 5.
    min_speed = 18.
    max_speed = reference_speed = 30.
    max_bank = np.deg2rad(45.)
    max_pitch = np.deg2rad(20.)
    max_flight_path = np.deg2rad(15.)
    surface_limit = np.deg2rad(25.)
    coefficients = {
        "CL0": .23, "Cm0": .0135, "CLalpha": 5.61, "Cmalpha": -2.74,
        "CLq": 7.95, "CDq": 0., "Cmq": -38.21,
        "CLde": .13, "CDde": .0135, "Cmde": -.99,
        "M": 50., "alpha0": .47, "CDp": .043,
        "CY0": 0., "Cl0": 0., "Cn0": 0.,
        "CYbeta": -.98, "Clbeta": -.13, "Cnbeta": .073,
        "CYp": 0., "Clp": -.51, "Cnp": .069,
        "CYr": 0., "Clr": .25, "Cnr": -.095,
        "CYda": .075, "Clda": .17, "Cnda": -.011,
        "CYdr": .19, "Cldr": .0024, "Cndr": -.069,
    }
    wing_area = .55
    wingspan = 2.8956
    chord = .18994
    air_density = 1.2682
    oswald_efficiency = .9
    propeller_diameter = .508
    motor_resistance = .042
    no_load_current = 1.5
    max_voltage = 44.4
    motor_constant = 60. / (145. * 2. * np.pi)
    propeller_torque = np.array([.005230, .004970, -.01664])
    propeller_thrust = np.array([.09357, -.06044, -.1079])
    trim_speeds = np.linspace(18., 30., 49)  # 0.25 m/s grid for the guidance feedforward

    def __init__(self):
        inertia_frd = np.array([[.8244, 0., -.1204], [0., 1.135, 0.], [-.1204, 0., 1.759]])
        super().__init__(11., inertia_frd * np.outer(FRD, FRD))
        self.aspect_ratio = self.wingspan**2 / self.wing_area
        c = self.coefficients
        # Lateral stability derivatives; rows multiply [1, beta, p b/2V, r b/2V, aileron, rudder].
        self._lateral = np.array([[c["CY" + k], c["Cl" + k], c["Cn" + k]]
                                  for k in ("0", "beta", "p", "r", "da", "dr")])
        self.trim_state, self.trim_actuator = self.trim(self.reference_speed)
        self._schedule = _trim_schedule(self)

    def propeller_force_torque(self, airspeed, throttle):
        """Steady DC-motor balance and quadratic propeller polynomials (McLain addendum)."""
        rho, diameter, k = self.air_density, self.propeller_diameter, self.motor_constant
        cq0, cq1, cq2 = self.propeller_torque
        quadratic = rho * diameter**5 * cq0 / (4.*np.pi**2)
        linear = rho * diameter**4 * cq1 * airspeed / (2.*np.pi) + k*k/self.motor_resistance
        constant = (rho * diameter**3 * cq2 * airspeed**2
                    - k*self.max_voltage*throttle/self.motor_resistance + k*self.no_load_current)
        discriminant = np.maximum(0., linear**2 - 4.*quadratic*constant)
        speed = np.maximum(0., (-linear + np.sqrt(discriminant)) / (2.*quadratic))
        basis = np.stack([diameter**4*speed**2/(4.*np.pi**2), diameter**3*airspeed*speed/(2.*np.pi),
                          diameter**2*airspeed**2], axis=-1)
        return rho * basis @ self.propeller_thrust, rho * diameter * (basis @ self.propeller_torque)

    def _forces(self, states, actuators, rotation):
        controls = np.clip(actuators, -1., 1.)
        elevator, aileron, rudder = (controls[:, :3] * self.surface_limit).T
        throttle = (controls[:, 3] + 1.) / 2.
        velocity = body_frd_velocity(states, rotation)
        airspeed = np.linalg.norm(velocity, axis=1)
        # The floor only defines zero-airspeed rate terms; dynamic pressure still vanishes.
        denominator = np.maximum(airspeed, 1e-6)
        alpha = np.arctan2(velocity[:, 2], velocity[:, 0])
        beta = np.arcsin(np.clip(velocity[:, 1] / denominator, -1., 1.))
        p, q, r = (states[:, 10:13] * FRD).T
        c = self.coefficients
        # Smooth flat-plate stall blend and induced-drag polar (Beard and McLain, Sec. 4.2).
        e_minus = np.exp(np.clip(-c["M"]*(alpha-c["alpha0"]), -500., 500.))
        e_plus = np.exp(np.clip(c["M"]*(alpha+c["alpha0"]), -500., 500.))
        stall = (1. + e_minus + e_plus) / ((1. + e_minus)*(1. + e_plus))
        cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
        linear_lift = c["CL0"] + c["CLalpha"]*alpha
        pitch_rate = self.chord*q/(2.*denominator)
        lift = ((1.-stall)*linear_lift + stall*2.*np.sign(alpha)*sin_alpha**2*cos_alpha
                + c["CLq"]*pitch_rate + c["CLde"]*elevator)
        drag = (c["CDp"] + linear_lift**2/(np.pi*self.oswald_efficiency*self.aspect_ratio)
                + c["CDq"]*pitch_rate + c["CDde"]*elevator)
        pressure_area = .5*self.air_density*airspeed**2*self.wing_area
        lateral = np.stack([np.ones_like(beta), beta, self.wingspan*p/(2.*denominator),
                            self.wingspan*r/(2.*denominator), aileron, rudder], axis=1) @ self._lateral
        thrust, torque = self.propeller_force_torque(airspeed, throttle)
        force = np.empty((len(states), 3))
        force[:, 0] = pressure_area*(-drag*cos_alpha + lift*sin_alpha) + thrust
        force[:, 1] = pressure_area*lateral[:, 0]
        force[:, 2] = pressure_area*(-drag*sin_alpha - lift*cos_alpha)
        moment = np.empty((len(states), 3))
        moment[:, 0] = pressure_area*self.wingspan*lateral[:, 1] - torque
        moment[:, 1] = pressure_area*self.chord*(c["Cm0"] + c["Cmalpha"]*alpha + c["Cmq"]*pitch_rate
                                                 + c["Cmde"]*elevator)
        moment[:, 2] = pressure_area*self.wingspan*lateral[:, 2]
        return force * FRD, moment * FRD

    @lru_cache(maxsize=64)
    def _trim_values(self, airspeed):
        def unpack(unknown):
            pitch, heading, elevator, aileron, rudder, throttle = unknown
            state = super(FixedWingDynamics, self).initial_state([0., 0., 0.], [airspeed, 0., 0.])
            state[6:10] = quaternion_from_euler(pitch=-pitch, yaw=heading)
            actuator = np.r_[np.array([elevator, aileron, rudder])/self.surface_limit, 2.*throttle-1.]
            return state, actuator

        def residual(unknown):
            state, actuator = unpack(unknown)
            derivative = self.derivative(state, actuator)
            return np.r_[derivative[3:6], derivative[10:13]]

        solved = root(residual, [.04, .001, -.10, .01, .001, .65], tol=1e-11)
        state, actuator = unpack(solved.x)
        if not solved.success or np.linalg.norm(residual(solved.x)) > 1e-6 or np.max(np.abs(actuator)) > 1.:
            raise RuntimeError(f"could not solve feasible Aerosonde trim at {airspeed:g} m/s")
        return state, actuator

    def trim(self, airspeed=30.):
        """Solve all three force and three moment balances for straight level flight."""
        state, actuator = self._trim_values(float(airspeed))
        return state.copy(), actuator.copy()

    def initial_state(self, position, velocity):
        """Level 30 m/s reference trim; the requested velocity only sets the heading."""
        requested = super().initial_state(position, velocity)
        horizontal = np.linalg.norm(requested[3:5])
        heading = np.arctan2(requested[4], requested[3]) if horizontal > 1e-6 else 0.
        state = self.trim_state.copy()
        state[:3] = requested[:3]
        state[3:6] = self.reference_speed * np.array([np.cos(heading), np.sin(heading), 0.])
        pitch, trim_heading = flight_angles(self.trim_state[6:10])[1:]
        state[6:10] = quaternion_from_euler(pitch=-pitch, yaw=heading+trim_heading)
        return state

    def _trim_feedforward(self, desired_speed):
        """Interpolated level trim (pitch, sideslip, actuators) for each requested speed."""
        speed = np.clip(desired_speed, self.min_speed, self.max_speed)
        values = np.stack([np.interp(speed, self.trim_speeds, row) for row in self._schedule], axis=1)
        return speed, values[:, 0], values[:, 1], values[:, 2:]

    def _guidance(self, states, rotation, desired_heading, desired_path, trim):
        desired_speed, trim_pitch, trim_beta, trim_actuator = trim
        roll, pitch, heading = matrix_angles(rotation)
        velocity = states[:, 3:6]
        speed = np.linalg.norm(velocity, axis=1)
        horizontal = np.linalg.norm(velocity[:, :2], axis=1)
        course = np.where(horizontal > 1e-6, np.arctan2(velocity[:, 1], velocity[:, 0]), heading)
        path = np.arctan2(velocity[:, 2], horizontal)
        desired_path = np.clip(desired_path, -self.max_flight_path, self.max_flight_path)
        # Coordinated-turn bank toward the desired course; positive bank turns right.
        turn_speed = np.maximum(speed, self.min_speed)
        desired_roll = np.clip(-np.arctan2(turn_speed * wrap_angle(desired_heading - course), self.gravity),
                               -self.max_bank, self.max_bank)
        desired_pitch = np.clip(trim_pitch + desired_path + .5*(desired_path - path), -self.max_pitch, self.max_pitch)
        p, q, r = (states[:, 10:13] * FRD).T
        beta = np.arcsin(np.clip(body_frd_velocity(states, rotation)[:, 1] / np.maximum(speed, 1e-6), -1., 1.))
        trim_elevator, trim_aileron, trim_rudder = (trim_actuator[:, :3] * self.surface_limit).T
        elevator = trim_elevator - 2.*(desired_pitch - pitch) + .25*q
        aileron = trim_aileron + .5*(desired_roll - roll) - .08*p
        # Cn_dr < 0: negative rudder yaws the nose right, toward positive sideslip.
        desired_yaw_rate = self.gravity*np.tan(desired_roll) / turn_speed
        rudder = trim_rudder - .8*(beta - trim_beta) + .1*(r - desired_yaw_rate)
        throttle = (trim_actuator[:, 3] + 1.)/2. + .08*(desired_speed - speed) + .7*np.sin(desired_path)
        return np.clip(np.c_[np.stack([elevator, aileron, rudder], axis=1)/self.surface_limit, 2.*throttle - 1.],
                       -1., 1.)

    def _velocity_command(self, states, velocity):
        horizontal = np.linalg.norm(velocity[:, :2], axis=1)
        course = np.arctan2(velocity[:, 1], velocity[:, 0])
        path = np.arctan2(velocity[:, 2], np.maximum(horizontal, 1e-6))
        trim = self._trim_feedforward(np.linalg.norm(velocity, axis=1))
        level = horizontal <= 1e-6

        def command(stage, rotation):
            # A vertical-only request keeps the current heading.
            heading = np.where(level, np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0]), course)
            return self._guidance(stage, rotation, heading, path, trim)
        return command

    def _position_command(self, position):
        trim = self._trim_feedforward(np.full(len(position), self.reference_speed))

        def command(stage, rotation):
            displacement = position - stage[:, :3]
            horizontal = np.linalg.norm(displacement[:, :2], axis=1)
            heading = np.where(horizontal > 1e-6, np.arctan2(displacement[:, 1], displacement[:, 0]),
                               np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0]))
            path = np.arctan2(displacement[:, 2], np.maximum(horizontal, 30.))
            return self._guidance(stage, rotation, heading, path, trim)
        return command


_SCHEDULES = {}


def _trim_schedule(model):
    """Rows: pitch, sideslip, elevator, aileron, rudder, throttle over ``trim_speeds`` (cached per class)."""
    key = type(model)
    if key not in _SCHEDULES:
        rows = []
        for speed in model.trim_speeds:
            state, actuator = model.trim(speed)
            body = body_frd_velocity(state[None], quaternion_matrix(state[None, 6:10]))[0]
            rows.append([flight_angles(state[6:10])[1], np.arcsin(body[1] / np.linalg.norm(body)), *actuator])
        _SCHEDULES[key] = np.asarray(rows).T.copy()
    return _SCHEDULES[key]


FixedWing = FixedWingDynamics
