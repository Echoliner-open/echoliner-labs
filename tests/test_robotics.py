import numpy as np

from echoliner.robotics import DHLink, KinematicChain, plan_minimum_jerk


def _planar_chain() -> KinematicChain:
    links = [
        DHLink(a=0.6, alpha=0.0, d=0.0),
        DHLink(a=0.4, alpha=0.0, d=0.0),
    ]
    return KinematicChain(links)


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
