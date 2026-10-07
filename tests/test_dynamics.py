"""Physical behavior checks for the independently implemented aircraft models."""
import importlib.util

import numpy as np
import pytest


def models():
    # Missing dynamics is the first observable failure, before source is written.
    assert importlib.util.find_spec("had_env.core.dynamics") is not None
    from had_env.core.dynamics import FixedWingDynamics, QuadrotorDynamics
    return FixedWingDynamics, QuadrotorDynamics


def attitude_distance(a, b):
    return 2 * np.arccos(np.clip(abs(np.dot(a[6:10], b[6:10])), 0., 1.))


def test_quadrotor_hover_has_no_artificial_motion():
    _, Quad = models()
    model = Quad()
    start = model.initial_state([0., 0., 100.], [0., 0., 0.])
    result, path = model.integrate(start, np.full(4, .4), dt=10.)
    assert np.linalg.norm(result[:3] - start[:3]) < 1e-5
    assert attitude_distance(result, start) < 1e-5
    assert np.linalg.norm(result[3:6]) < 1e-5
    assert path.shape == (1001, 3)


def test_fixedwing_trim_balances_all_force_and_moment_axes():
    Wing, _ = models()
    model = Wing()
    state = model.initial_state([0., 0., 100.], [30., 0., 0.])
    derivative = model.derivative(state, model.trim_actuator)
    assert np.linalg.norm(derivative[3:6]) < 1e-6
    assert np.linalg.norm(derivative[10:13]) < 1e-6
    result, _ = model.integrate(state, model.trim_actuator, dt=10.)
    np.testing.assert_allclose(result[:3], [300., 0., 100.], atol=1e-5)
    assert np.linalg.norm(result[3:6] - state[3:6]) < 1e-5


@pytest.mark.parametrize("kind", ["quadrotor", "fixedwing"])
def test_controller_substep_converges_over_ten_seconds(kind):
    Wing, Quad = models()
    model = Quad() if kind == "quadrotor" else Wing()
    state = model.initial_state([0., 0., 100.], [0., 0., 0.] if kind == "quadrotor" else [30., 0., 0.])
    target = np.array([160., 50., 130.])
    coarse, _ = model.advance(state, target, "position", dt=10., substep=.01)
    fine, _ = model.advance(state, target, "position", dt=10., substep=.005)
    assert np.isfinite(coarse).all()
    assert abs(np.linalg.norm(coarse[6:10]) - 1.) < 1e-12
    assert np.linalg.norm(coarse[:3] - fine[:3]) < .1
    assert np.linalg.norm(coarse[3:6] - fine[3:6]) < .05
    assert attitude_distance(coarse, fine) < np.deg2rad(.5)


def test_quadrotor_velocity_and_position_controllers_move_in_requested_direction():
    _, Quad = models()
    model = Quad()
    state = model.initial_state([0., 0., 100.], [0., 0., 0.])
    moved, path = model.advance(state, [3., 0., 2.], "acceleration", dt=1.)
    assert moved[0] > .1 and moved[2] > 100.1
    assert moved[3] > .5 and moved[5] > .5
    assert path.shape == (101, 3)
    np.testing.assert_array_equal(path[0], state[:3])
    np.testing.assert_array_equal(path[-1], moved[:3])
    arrived, _ = model.advance(state, [10., 0., 105.], "position", dt=15.)
    assert np.linalg.norm(arrived[:3] - [10., 0., 105.]) < .1


def test_fixedwing_position_command_turns_and_keeps_flying():
    Wing, _ = models()
    model = Wing()
    state = model.initial_state([0., 0., 100.], [30., 0., 0.])
    moved, _ = model.advance(state, [200., 100., 130.], "position", dt=8.)
    assert moved[1] > 20.
    assert moved[2] > 105.
    assert np.linalg.norm(moved[3:6]) > 15.


def test_quadrotor_allocator_preserves_collective_when_torques_saturate():
    _, Quad = models()
    model = Quad()
    actuator = model.allocate(42.5754, [1e5, -2e5, 1e5])
    assert np.all((actuator >= 0.) & (actuator <= 1.))
    force, _ = model.forces_and_moments(model.initial_state([0., 0., 0.], [0., 0., 0.]), actuator)
    assert abs(force[2] - 42.5754) < 1e-9


@pytest.mark.parametrize("kind", ["quadrotor", "fixedwing"])
def test_actuator_bounds_and_stateless_replay(kind):
    Wing, Quad = models()
    model = Quad() if kind == "quadrotor" else Wing()
    state = model.initial_state([0., 0., 100.], [0., 0., 0.] if kind == "quadrotor" else [30., 0., 0.])
    command = model.acceleration_to_actuator(state, [1000., -1000., 1000.])
    lower = 0. if kind == "quadrotor" else -1.
    assert command.shape == (4,)
    assert np.all((command >= lower) & (command <= 1.))
    first, _ = model.advance(state, [2., 1., 0.], "acceleration")
    model.advance(first, [-1., -1., 1.], "acceleration")
    replay, _ = model.advance(state, [2., 1., 0.], "acceleration")
    np.testing.assert_array_equal(first, replay)


def test_quadrotor_zero_rotor_thrust_is_free_fall():
    _, Quad = models()
    state = Quad().initial_state([0., 0., 100.], [0., 0., 0.])
    result, _ = Quad().integrate(state, np.zeros(4), dt=1.)
    np.testing.assert_allclose(result[:6], [0., 0., 95.095, 0., 0., -9.81], atol=1e-10)


def test_quadrotor_zero_acceleration_preserves_feasible_sixteen_mps_velocity():
    _, Quad = models()
    model = Quad()
    state = model.initial_state([0., 0., 100.], [16., 0., 0.])
    result, _ = model.advance(state, [0., 0., 0.], "acceleration", dt=2.)
    # The ideal model has no drag. Its velocity reference supports 20 m/s,
    # while the slower 12 m/s cap belongs only to position guidance.
    np.testing.assert_allclose(result[:6], [32., 0., 100., 16., 0., 0.], atol=1e-9)


def test_quadrotor_flu_tilt_and_rotor_moments_have_physical_signs():
    _, Quad = models()
    from had_env.core.dynamics.control import quaternion_from_euler
    model = Quad()
    state = model.initial_state([0., 0., 100.], [0., 0., 0.])
    # Nose-down tilt points the up-axis thrust toward world east.
    state[6:10] = quaternion_from_euler(pitch=np.deg2rad(10.))
    acceleration = model.derivative(state, np.full(4, .4))[3:6]
    np.testing.assert_allclose(acceleration, [1.70348862, 0., -.14903594], atol=1e-8)
    # Front rotor creates a nose-up (-body-y) pitch and positive body-z yaw.
    _, moment = model.forces_and_moments(state, [1., 0., 0., 0.])
    assert moment[0] == 0. and moment[1] < 0. and moment[2] > 0.
    # Left rotor creates positive roll about forward x.
    _, moment = model.forces_and_moments(state, [0., 0., 0., 1.])
    assert moment[0] > 0. and moment[1] == 0.


def test_fixedwing_trim_rotates_with_heading_without_changing_balance():
    Wing, _ = models()
    model = Wing()
    for direction in ([0., 30., 0.], [-30., 0., 0.], [0., -30., 0.]):
        state = model.initial_state([0., 0., 100.], direction)
        derivative = model.derivative(state, model.trim_actuator)
        assert np.linalg.norm(derivative[3:6]) < 1e-6
        assert np.linalg.norm(derivative[10:13]) < 1e-6
        np.testing.assert_allclose(state[3:6], direction, atol=1e-12)


@pytest.mark.parametrize("airspeed", [18., 24., 30.])
def test_fixedwing_velocity_controller_preserves_level_equilibrium_across_reference_speeds(airspeed):
    Wing, _ = models()
    model = Wing()
    state, _ = model.trim(airspeed)
    actuator = model.velocity_to_actuator(state, [airspeed, 0., 0.])
    derivative = model.derivative(state, actuator)
    # A correct controller should leave this feasible requested equilibrium
    # intact, including the small sideslip that balances propeller roll torque.
    assert np.linalg.norm(derivative[3:6]) < 1e-6
    assert np.linalg.norm(derivative[10:13]) < 1e-6


def test_fixedwing_airfoil_and_surface_moments_use_flu_convention():
    Wing, _ = models()
    model = Wing()
    state = model.initial_state([0., 0., 100.], [30., 0., 0.])
    force, baseline = model.forces_and_moments(state, model.trim_actuator)
    assert force[2] > 100.  # lift is up in FLU
    controls = model.trim_actuator.copy()
    controls[0] -= .1
    _, elevator = model.forces_and_moments(state, controls)
    assert elevator[1] < baseline[1]  # negative elevator makes the nose rise
    controls = model.trim_actuator.copy()
    controls[1] += .1
    _, aileron = model.forces_and_moments(state, controls)
    assert aileron[0] > baseline[0]
    controls = model.trim_actuator.copy()
    controls[2] += .1
    _, rudder = model.forces_and_moments(state, controls)
    assert rudder[2] > baseline[2]


@pytest.mark.parametrize("kind", ["quadrotor", "fixedwing"])
def test_all_extreme_physical_actuators_remain_finite_without_velocity_clamps(kind):
    Wing, Quad = models()
    model = Quad() if kind == "quadrotor" else Wing()
    state = model.initial_state([0., 0., 100.], [0., 0., 0.] if kind == "quadrotor" else [30., 0., 0.])
    for actuator in ([1., 1., 1., 1.], [-1., -1., -1., -1.], [1., -1., 1., -1.]):
        result, _ = model.integrate(state, actuator, dt=1.)
        assert np.isfinite(result).all()
        assert abs(np.linalg.norm(result[6:10])-1.) < 1e-12
    if kind == "quadrotor":
        full, _ = model.integrate(state, np.ones(4), dt=2.)
        assert full[5] > model.max_speed
