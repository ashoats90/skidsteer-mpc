"""Proximity-based safety filters for skidsteer_mpc.

These filters sit between the MPC solve and the CommandGovernor. They do not
replace the MPC obstacle constraints; they add an extra speed envelope based on
current obstacle proximity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controller import SolveResult, SolveStatus
from .obstacle import Obstacle


@dataclass
class ProximitySafetyConfig:
    """Distance-based longitudinal speed envelope.

    Distances are measured from the vehicle point-mass position to the obstacle
    center, matching Obstacle.distance(...).

    stop_distance:
        Latch a stop when distance <= this value.
    slow_distance:
        Begin reducing speed when distance is below this value.
    latch_stop:
        If True, once the stop_distance is crossed, all future commands are
        held at zero.
    use_smoothstep:
        If True, use a smooth cubic ramp instead of a linear ramp.
    """
    stop_distance: float = 1.0
    slow_distance: float = 4.0
    latch_stop: bool = True
    use_smoothstep: bool = True


class ProximitySpeedLimiter:
    """Scale MPC track-speed plans based on obstacle proximity.

    The limiter scales the longitudinal component of the track commands while
    preserving the steering difference between left and right tracks as much as
    possible. If the stop threshold is crossed, it commands zero velocity.

    This class also stores distance and speed-scale histories so experiment
    scripts can classify outcomes after the run.
    """

    def __init__(self, obstacle: Obstacle, cfg: ProximitySafetyConfig | None = None):
        self.obstacle = obstacle
        self.cfg = cfg or ProximitySafetyConfig()

        if self.cfg.slow_distance <= self.cfg.stop_distance:
            raise ValueError("slow_distance must be greater than stop_distance")

        self.stop_latched = False
        self.last_distance = np.inf
        self.last_scale = 1.0

        # Histories for validation / sweep metrics.
        self.distance_history: list[float] = []
        self.scale_history: list[float] = []

    def reset(self) -> None:
        self.stop_latched = False
        self.last_distance = np.inf
        self.last_scale = 1.0
        self.distance_history.clear()
        self.scale_history.clear()

    def speed_scale(self, distance: float) -> float:
        """Return speed scale in [0, 1] for the current obstacle distance."""
        c = self.cfg

        if distance <= c.stop_distance:
            return 0.0

        if distance >= c.slow_distance:
            return 1.0

        # x goes from 0 at stop_distance to 1 at slow_distance.
        x = (distance - c.stop_distance) / (c.slow_distance - c.stop_distance)
        x = float(np.clip(x, 0.0, 1.0))

        if c.use_smoothstep:
            return x * x * (3.0 - 2.0 * x)

        return x

    def __call__(self, state, result: SolveResult, t: float | None = None) -> SolveResult:
        """Return a modified SolveResult with obstacle-aware speed reduction."""
        distance = float(self.obstacle.distance(state[0], state[1]))

        if distance <= self.cfg.stop_distance:
            self.stop_latched = True

        if self.stop_latched and self.cfg.latch_stop:
            scale = 0.0
        else:
            scale = self.speed_scale(distance)

        self.last_distance = distance
        self.last_scale = scale
        self.distance_history.append(distance)
        self.scale_history.append(scale)

        if result.status != SolveStatus.OK or result.plan is None:
            return result

        plan = np.asarray(result.plan, dtype=float).copy()

        if len(plan) < 2:
            return result

        # Track plan is flat: [VL0, VR0, VL1, VR1, ...].
        for k in range(0, len(plan) - 1, 2):
            vl = plan[k]
            vr = plan[k + 1]

            longitudinal = 0.5 * (vl + vr)
            differential = 0.5 * (vr - vl)

            longitudinal *= scale

            # If fully stopped, command both tracks to zero instead of allowing
            # an in-place spin.
            if scale <= 0.0:
                plan[k] = 0.0
                plan[k + 1] = 0.0
            else:
                plan[k] = longitudinal - differential
                plan[k + 1] = longitudinal + differential

        return SolveResult(
            status=result.status,
            plan=plan.tolist(),
            solve_time_s=result.solve_time_s,
        )
