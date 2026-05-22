"""Obstacle model. dist2 is written with plain arithmetic so it works on numpy
arrays (simulation/metrics) and CasADi symbols (NLP constraints) alike."""
from __future__ import annotations
import numpy as np
from .config import ObstacleSpec


class Obstacle:
    def __init__(self, spec: ObstacleSpec | None = None):
        self.spec = spec or ObstacleSpec()

    @property
    def r_safe(self) -> float:
        return self.spec.r_safe

    def dist2(self, x, y):
        return (x - self.spec.x) ** 2 + (y - self.spec.y) ** 2

    def distance(self, x, y):
        return np.hypot(x - self.spec.x, y - self.spec.y)

class TimedMovingObstacle(Obstacle):
    """Obstacle that moves from start_xy to end_xy over a time window.

    The simulator should call update(t) before each controller solve. Because
    the controller, safety filter, plotter, and simulator all share the same
    obstacle object, updating the object updates the current environment state.
    """
    def __init__(
        self,
        spec: ObstacleSpec | None = None,
        start_xy=(12.5, -4.0),
        end_xy=(12.5, 0.0),
        move_start_s: float = 2.0,
        move_duration_s: float = 2.0,
    ):
        super().__init__(spec)
        self.start_xy = tuple(start_xy)
        self.end_xy = tuple(end_xy)
        self.move_start_s = float(move_start_s)
        self.move_duration_s = float(move_duration_s)
        self.update(0.0)

    def update(self, t: float) -> None:
        if self.move_duration_s <= 0:
            alpha = 1.0 if t >= self.move_start_s else 0.0
        else:
            alpha = (t - self.move_start_s) / self.move_duration_s
            alpha = float(np.clip(alpha, 0.0, 1.0))

        self.spec.x = (1.0 - alpha) * self.start_xy[0] + alpha * self.end_xy[0]
        self.spec.y = (1.0 - alpha) * self.start_xy[1] + alpha * self.end_xy[1]
