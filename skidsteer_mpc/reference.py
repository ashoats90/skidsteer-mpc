"""Reference paths behind one interface. Each subclass only defines its geometry;
the base class handles arc-length parameterization, time sampling, and horizon
preview. Swapping the demo path is a one-line change at the call site."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class ReferencePath(ABC):
    def __init__(self, v_target: float):
        self.v_target = v_target
        self.xg, self.yg = self._xy_grid()
        ds = np.hypot(np.diff(self.xg), np.diff(self.yg))
        self.s = np.concatenate([[0.0], np.cumsum(ds)])

    @abstractmethod
    def _xy_grid(self):
        """Return dense (xg, yg) arrays defining the geometric path."""

    def sample(self, t: float) -> np.ndarray:
        """Return [x_ref, y_ref, yaw_ref] at time t (constant-speed arc-length)."""
        s = min(self.v_target * t, self.s[-1])
        xr = np.interp(s, self.s, self.xg)
        yr = np.interp(s, self.s, self.yg)
        x2 = np.interp(s + 1e-3, self.s, self.xg)
        y2 = np.interp(s + 1e-3, self.s, self.yg)
        return np.array([xr, yr, np.arctan2(y2 - yr, x2 - xr)])

    def preview(self, t0: float, n: int, dt: float) -> np.ndarray:
        """Return an (n+1, 3) array of [x, y, yaw] over the horizon."""
        return np.array([self.sample(t0 + k * dt) for k in range(n + 1)])


class SCurveLaneChange(ReferencePath):
    """Smooth lane change: y transitions 0 -> lane via a tanh sigmoid."""
    def __init__(self, v_target, lane=3.0, x0=13.0, width=2.2, length=35.0):
        self.lane, self.x0, self.width, self.length = lane, x0, width, length
        super().__init__(v_target)

    def _xy_grid(self):
        xg = np.linspace(0.0, self.length, 6000)
        yg = self.lane * 0.5 * (1.0 + np.tanh((xg - self.x0) / self.width))
        return xg, yg


class StraightWithSwerve(ReferencePath):
    """Straight line with a single Gaussian swerve out and back."""
    def __init__(self, v_target, amp=2.5, center=13.0, width=2.0, length=35.0):
        self.amp, self.center, self.width, self.length = amp, center, width, length
        super().__init__(v_target)

    def _xy_grid(self):
        xg = np.linspace(0.0, self.length, 6000)
        yg = self.amp * np.exp(-(((xg - self.center) / self.width) ** 2))
        return xg, yg


class ConstantRadiusCircle(ReferencePath):
    """Circle of radius R, starting at the origin heading +x, curving left."""
    def __init__(self, v_target, radius=8.0, sweep=2 * np.pi):
        self.radius, self.sweep = radius, sweep
        super().__init__(v_target)

    def _xy_grid(self):
        th = np.linspace(0.0, self.sweep, 6000)
        xg = self.radius * np.sin(th)
        yg = self.radius * (1.0 - np.cos(th))
        return xg, yg
