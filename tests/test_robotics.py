import numpy as np

from echoliner.robotics import (
    DHLink,
    ComputedTorqueController,
    JointDynamics,
    KinematicChain,
    ModelPredictiveController,
    OperationalSpaceController,
    RRTPlanner,
    RigidBodyParameters,
    TrajectorySmoother,
    WorkspaceObstacle,
    plan_minimum_jerk,
    simulate_dynamics,
)


def _planar_chain() -> KinematicChain:
    links = [
        DHLink(a=0.6, alpha=0.0, d=0.0),
        DHLink(a=0.4, alpha=0.0, d=0.0),
    ]
    return KinematicChain(links)


def _planar_dynamics() -> JointDynamics:
    chain = _planar_chain()
    params = [
        RigidBodyParameters(mass=5.0, com=np.array([0.12, 0.0, 0.0]), inertia=np.eye(3)),
        RigidBodyParameters(mass=3.5, com=np.array([0.08, 0.0, 0.0]), inertia=np.eye(3)),
    ]
    return JointDynamics(chain, params)


def test_forward_kinematics_planar_arm() -> None:
    chain = _planar_chain()
    thetas = np.deg2rad([40.0, -20.0])
    pose = chain.forward_kinematics(thetas)
    expected_x = 0.6 * np.cos(thetas[0]) + 0.4 * np.cos(thetas.sum())
    expected_y = 0.6 * np.sin(thetas[0]) + 0.4 * np.sin(thetas.sum())
    assert np.isclose(pose[0, 3], expected_x)
    assert np.isclose(pose[1, 3], expected_y)


def test_jacobian_matches_analytic_translation() -> None:
    chain = _planar_chain()
    thetas = np.deg2rad([25.0, -10.0])
    jac = chain.jacobian(thetas)
    analytic = np.array(
        [
            [
                -0.6 * np.sin(thetas[0]) - 0.4 * np.sin(thetas.sum()),
                -0.4 * np.sin(thetas.sum()),
            ],
            [
                0.6 * np.cos(thetas[0]) + 0.4 * np.cos(thetas.sum()),
                0.4 * np.cos(thetas.sum()),
            ],
            [0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(jac[3:, :], analytic, atol=1e-5)
    np.testing.assert_allclose(jac[0:2, :], np.zeros((2, 2)), atol=1e-6)
    np.testing.assert_allclose(jac[2, :], np.ones(2), atol=1e-5)


def test_inverse_kinematics_converges_to_target() -> None:
    chain = _planar_chain()
    target_angles = np.array([0.4, -0.7])
    target_pose = chain.forward_kinematics(target_angles)
    solution = chain.inverse_kinematics(target_pose, initial_guess=[0.0, 0.0])
    np.testing.assert_allclose(solution, target_angles, atol=1e-3)


def test_minimum_jerk_profile_has_zero_boundary_velocity() -> None:
    start = np.array([0.0, 0.5])
    goal = np.array([1.0, -0.2])
    times, positions = plan_minimum_jerk(start, goal, duration=2.0, samples=20)
    np.testing.assert_allclose(positions[0], start)
    np.testing.assert_allclose(positions[-1], goal)
    velocities = np.gradient(positions, times, axis=0)
    np.testing.assert_allclose(velocities[0], np.zeros_like(start), atol=2e-2)
    np.testing.assert_allclose(velocities[-1], np.zeros_like(goal), atol=2e-2)


def test_joint_dynamics_mass_matrix_positive() -> None:
    chain = _planar_chain()
    params = [
        RigidBodyParameters(mass=5.0, com=np.array([0.1, 0.0, 0.0]), inertia=np.eye(3)),
        RigidBodyParameters(mass=3.0, com=np.array([0.05, 0.0, 0.0]), inertia=np.eye(3)),
    ]
    model = JointDynamics(chain, params)
    mass = model.mass_matrix(np.zeros(chain.dof))
    assert np.all(np.linalg.eigvals(mass) > 0)


def test_computed_torque_controller_outputs_vector() -> None:
    chain = _planar_chain()
    params = [
        RigidBodyParameters(mass=4.0, com=np.zeros(3), inertia=np.eye(3)),
        RigidBodyParameters(mass=2.0, com=np.zeros(3), inertia=np.eye(3)),
    ]
    model = JointDynamics(chain, params)
    controller = ComputedTorqueController(model, kp=np.ones(2), kd=np.ones(2))
    torque = controller(np.zeros(2), np.zeros(2), target_positions=[0.1, -0.1])
    assert torque.shape == (2,)


def test_rrt_planner_returns_path() -> None:
    chain = _planar_chain()
    planner = RRTPlanner(chain, bounds=[(-np.pi, np.pi), (-np.pi, np.pi)], max_iterations=100)
    path = planner.plan([0.0, 0.0], [0.5, -0.5], lambda _: False)
    assert path[0].shape == (2,)


def test_simulate_dynamics_advances_state() -> None:
    chain = _planar_chain()
    params = [
        RigidBodyParameters(mass=5.0, com=np.zeros(3), inertia=np.eye(3)),
        RigidBodyParameters(mass=3.0, com=np.zeros(3), inertia=np.eye(3)),
    ]
    model = JointDynamics(chain, params)
    q, dq = simulate_dynamics(model, np.zeros(2), np.zeros(2), np.ones(2), dt=0.01)
    assert q.shape == (2,)
    assert dq.shape == (2,)


def test_operational_space_controller_matches_gravity_at_target() -> None:
    model = _planar_dynamics()
    controller = OperationalSpaceController(
        model,
        task_stiffness=np.ones(6) * 40.0,
        task_damping=np.ones(6) * 5.0,
    )
    q = np.zeros(model.chain.dof)
    pose = model.chain.forward_kinematics(q)
    torque = controller.compute(q, np.zeros(model.chain.dof), pose)
    np.testing.assert_allclose(torque, model.gravity(q), atol=1e-6)


def test_operational_space_controller_generates_correction_torque() -> None:
    model = _planar_dynamics()
    controller = OperationalSpaceController(
        model,
        task_stiffness=np.ones(6) * 30.0,
        task_damping=np.ones(6) * 4.0,
    )
    q = np.zeros(model.chain.dof)
    target = model.chain.forward_kinematics(q)
    target[0, 3] += 0.05
    torque = controller.compute(q, np.zeros(model.chain.dof), target)
    gravity = model.gravity(q)
    assert np.linalg.norm(torque - gravity) > 1e-6
