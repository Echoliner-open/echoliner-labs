"""Basic robotics utilities for EchoLiner robotics module.

Provides a simple 2D planar arm model with forward kinematics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Arm2D:
    """Two-link planar robotic arm.

    Attributes
    ----------
    link1: float
        Length of the first link.
    link2: float
        Length of the second link.
    """

    link1: float
    link2: float

    def forward_kinematics(self, theta1: float, theta2: float) -> Tuple[float, float]:
        """Compute end-effector (x, y) position.

        Parameters
        ----------
        theta1: float
            Angle of the first joint in radians.
        theta2: float
            Angle of the second joint in radians.

        Returns
        -------
        Tuple[float, float]
            (x, y) position of the end effector.
        """
        x = self.link1 * math.cos(theta1) + self.link2 * math.cos(theta1 + theta2)
        y = self.link1 * math.sin(theta1) + self.link2 * math.sin(theta1 + theta2)
        return (x, y)
