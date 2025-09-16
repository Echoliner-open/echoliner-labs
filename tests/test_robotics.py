import math
from echoliner.robotics.kinematics import Arm2D


def test_forward_kinematics_origin():
    arm = Arm2D(link1=1.0, link2=1.0)
    x, y = arm.forward_kinematics(0.0, 0.0)
    assert math.isclose(x, 2.0)
    assert math.isclose(y, 0.0)


def test_forward_kinematics_right_angle():
    arm = Arm2D(link1=1.0, link2=1.0)
    x, y = arm.forward_kinematics(math.pi / 2, 0.0)
    assert math.isclose(x, 0.0, abs_tol=1e-7)
    assert math.isclose(y, 2.0)
