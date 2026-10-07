"""Aerosonde six-degree-of-freedom aircraft, independently authored.

Physical/aerodynamic/motor/propeller parameter source is BYU MAGICC's
published Aerosonde parameter set (retrieved 2026-10-08):
https://raw.githubusercontent.com/byu-magicc/mavsim_public/main/mavsim_python/parameters/aerosonde_parameters.py
Newton-Euler equations and lift/drag decomposition are implemented directly;
no reference implementation is imported or copied. Traditional forward-right-
down (FRD) aerodynamic vectors are rotated to forward-left-up (FLU) by the
proper rotation diag(1,-1,-1), including moments and the inertia tensor.

Inputs are normalized [elevator, aileron, rudder, throttle], all [-1,1];
surface limits are 25 degrees and throttle=(input+1)/2. Guidance limits are
reference commands, never physical state clamps. The position controller
flies toward and through its target; it does not try to hover.
"""
from functools import lru_cache

import numpy as np
from scipy.optimize import root

from .control import flight_angles, quaternion_from_euler, quaternion_matrix, wrap_angle
from .rigid_body import RigidBodyDynamics


class FixedWingDynamics(RigidBodyDynamics):
    max_acceleration = acceleration_limit = 5.
    min_speed = 18.
    max_speed = reference_speed = 30.
    max_bank = np.deg2rad(45.)
    max_pitch = np.deg2rad(20.)
    max_flight_path = np.deg2rad(15.)
    surface_limit = np.deg2rad(25.)
    frd_to_flu = np.diag([1., -1., -1.])
    # Scalars are reference data; the forces below are original code.
    coefficients = {
        "CL0": .23, "CD0": .0424, "Cm0": .0135,
        "CLalpha": 5.61, "CDalpha": .132, "Cmalpha": -2.74,
        "CLq": 7.95, "CDq": 0., "Cmq": -38.21,
        "CLde": .13, "CDde": .0135, "Cmde": -.99,
        "M": 50., "alpha0": .47, "epsilon": .16, "CDp": .043,
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

    def __init__(self):
        inertia_frd = np.array([[.8244, 0., -.1204], [0., 1.135, 0.], [-.1204, 0., 1.759]])
        super().__init__(11., self.frd_to_flu @ inertia_frd @ self.frd_to_flu)
        self.aspect_ratio = self.wingspan**2 / self.wing_area
        self.trim_state, self.trim_actuator = self.trim(self.reference_speed)
        self.trim_alpha = flight_angles(self.trim_state[6:10])[1]

    def propeller_force_torque(self, airspeed, throttle):
        """Steady DC motor balance and quadratic propeller polynomials."""
        rho, diameter, k = self.air_density, self.propeller_diameter, self.motor_constant
        cq0, cq1, cq2 = self.propeller_torque
        quadratic = rho * diameter**5 * cq0 / (4.*np.pi**2)
        linear = rho * diameter**4 * cq1 * airspeed / (2.*np.pi) + k*k/self.motor_resistance
        constant = (rho * diameter**3 * cq2 * airspeed**2
                    - k*self.max_voltage*throttle/self.motor_resistance + k*self.no_load_current)
        discriminant = max(0., linear**2 - 4.*quadratic*constant)
        speed = max(0., (-linear + np.sqrt(discriminant)) / (2.*quadratic))
        basis = np.array([diameter**4*speed**2/(4.*np.pi**2),
                          diameter**3*airspeed*speed/(2.*np.pi), diameter**2*airspeed**2])
        thrust = rho * self.propeller_thrust @ basis
        torque = rho * diameter * (self.propeller_torque @ basis)
        return float(thrust), float(torque)

    def forces_and_moments(self, state, actuator):
        controls = np.clip(actuator, -1., 1.)
        elevator, aileron, rudder = controls[:3] * self.surface_limit
        throttle = (controls[3] + 1.) / 2.
        rotation = quaternion_matrix(state[6:10])
        velocity = self.frd_to_flu @ (rotation.T @ state[3:6])
        airspeed = float(np.linalg.norm(velocity))
        # The denominator floor only defines the zero-airspeed rate terms;
        # dynamic pressure still vanishes at rest.
        denominator = max(airspeed, 1e-6)
        alpha = np.arctan2(velocity[2], velocity[0])
        beta = np.arcsin(np.clip(velocity[1]/denominator, -1., 1.))
        p, q, r = self.frd_to_flu @ state[10:13]
        c = self.coefficients
        # Smooth flat-plate stall blend; induced-drag polar replaces the
        # parameter file's optional linear CDalpha approximation.
        e_minus = np.exp(np.clip(-c["M"]*(alpha-c["alpha0"]), -500., 500.))
        e_plus = np.exp(np.clip(c["M"]*(alpha+c["alpha0"]), -500., 500.))
        stall = (1. + e_minus + e_plus) / ((1. + e_minus)*(1. + e_plus))
        linear_lift = c["CL0"] + c["CLalpha"]*alpha
        lift = ((1.-stall)*linear_lift + stall*2.*np.sign(alpha)*np.sin(alpha)**2*np.cos(alpha)
                + c["CLq"]*self.chord*q/(2.*denominator) + c["CLde"]*elevator)
        drag = (c["CDp"] + linear_lift**2/(np.pi*self.oswald_efficiency*self.aspect_ratio)
                + c["CDq"]*self.chord*q/(2.*denominator) + c["CDde"]*elevator)
        pressure_area = .5*self.air_density*airspeed**2*self.wing_area
        roll_rate, yaw_rate = self.wingspan*p/(2.*denominator), self.wingspan*r/(2.*denominator)
        side = c["CY0"]+c["CYbeta"]*beta+c["CYp"]*roll_rate+c["CYr"]*yaw_rate+c["CYda"]*aileron+c["CYdr"]*rudder
        force_frd = pressure_area * np.array([-drag*np.cos(alpha)+lift*np.sin(alpha), side,
                                              -drag*np.sin(alpha)-lift*np.cos(alpha)])
        roll = c["Cl0"]+c["Clbeta"]*beta+c["Clp"]*roll_rate+c["Clr"]*yaw_rate+c["Clda"]*aileron+c["Cldr"]*rudder
        pitch = c["Cm0"]+c["Cmalpha"]*alpha+c["Cmq"]*self.chord*q/(2.*denominator)+c["Cmde"]*elevator
        yaw = c["Cn0"]+c["Cnbeta"]*beta+c["Cnp"]*roll_rate+c["Cnr"]*yaw_rate+c["Cnda"]*aileron+c["Cndr"]*rudder
        moment_frd = pressure_area * np.array([self.wingspan*roll, self.chord*pitch, self.wingspan*yaw])
        propeller_force, propeller_torque = self.propeller_force_torque(airspeed, throttle)
        force_frd[0] += propeller_force
        moment_frd[0] -= propeller_torque
        return self.frd_to_flu @ force_frd, self.frd_to_flu @ moment_frd

    @lru_cache(maxsize=32)
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
        requested = self._vector(velocity, 3, "velocity")
        horizontal = np.linalg.norm(requested[:2])
        heading = np.arctan2(requested[1], requested[0]) if horizontal > 1e-6 else 0.
        # Resets establish the documented level reference trim. Velocity conveys
        # heading, while actual speed is the 30 m/s solved trim reference.
        state = self.trim_state.copy()
        state[:3] = self._vector(position, 3, "position")
        state[3:6] = self.reference_speed * np.array([np.cos(heading), np.sin(heading), 0.])
        pitch, trim_heading = flight_angles(self.trim_state[6:10])[1:]
        state[6:10] = quaternion_from_euler(pitch=-pitch, yaw=heading+trim_heading)
        return state

    def _guidance(self, state, desired_heading, desired_path, desired_speed):
        roll, pitch, heading = flight_angles(state[6:10])
        velocity = state[3:6]
        speed = np.linalg.norm(velocity)
        course = np.arctan2(velocity[1], velocity[0]) if np.linalg.norm(velocity[:2]) > 1e-6 else heading
        path = np.arctan2(velocity[2], np.linalg.norm(velocity[:2]))
        desired_speed = float(np.clip(desired_speed, self.min_speed, self.max_speed))
        desired_path = float(np.clip(desired_path, -self.max_flight_path, self.max_flight_path))
        heading_error = wrap_angle(desired_heading-course)
        desired_roll = np.clip(-np.arctan2(max(speed, self.min_speed)*heading_error, self.gravity),
                               -self.max_bank, self.max_bank)
        desired_pitch = np.clip(self.trim_alpha + desired_path + .5*(desired_path-path),
                                -self.max_pitch, self.max_pitch)
        p, q, r = self.frd_to_flu @ state[10:13]
        body_velocity = self.frd_to_flu @ (quaternion_matrix(state[6:10]).T @ velocity)
        beta = np.arcsin(np.clip(body_velocity[1]/max(speed, 1e-6), -1., 1.))
        trim_e, trim_a, trim_r = self.trim_actuator[:3]*self.surface_limit
        elevator = trim_e - 2.*(desired_pitch-pitch) + .25*q
        aileron = trim_a + .5*(desired_roll-roll) - .08*p
        desired_yaw_frd = self.gravity*np.tan(desired_roll)/max(speed, self.min_speed)
        rudder = trim_r + .8*beta + .1*(r-desired_yaw_frd)
        trim_throttle = (self.trim_actuator[3]+1.)/2.
        throttle = trim_throttle + .08*(desired_speed-speed) + .7*np.sin(desired_path)
        return np.clip(np.r_[np.array([elevator, aileron, rudder])/self.surface_limit, 2.*throttle-1.], -1., 1.)

    def velocity_to_actuator(self, state, velocity):
        velocity = np.asarray(velocity)
        horizontal = np.linalg.norm(velocity[:2])
        heading = np.arctan2(velocity[1], velocity[0]) if horizontal > 1e-6 else flight_angles(state[6:10])[2]
        path = np.arctan2(velocity[2], max(horizontal, 1e-6))
        return self._guidance(state, heading, path, np.linalg.norm(velocity))

    def position_to_actuator(self, state, position):
        displacement = np.asarray(position)-state[:3]
        horizontal = np.linalg.norm(displacement[:2])
        heading = np.arctan2(displacement[1], displacement[0]) if horizontal > 1e-6 else flight_angles(state[6:10])[2]
        path = np.arctan2(displacement[2], max(horizontal, 30.))
        return self._guidance(state, heading, path, self.reference_speed)


FixedWing = FixedWingDynamics
