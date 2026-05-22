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
