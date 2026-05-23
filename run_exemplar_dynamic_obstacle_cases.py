from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np

from skidsteer_mpc import (
    SkidSteerModel,
    SCurveLaneChange,
    TimedMovingObstacle,
    ScipyMPC,
    CommandGovernor,
    ClosedLoopSimulator,
    MpcAnimator,
    MPCConfig,
    SupervisorConfig,
    SimConfig,
)
from skidsteer_mpc.controller import SolveResult
from skidsteer_mpc.safety import ProximitySafetyConfig, ProximitySpeedLimiter

OUTDIR = Path("out")
OUTDIR.mkdir(exist_ok=True)

DT = 1.0 / 6.0
REPRESENTED_SOLVE_MS = 50.0


@dataclass
class ScenarioSpec:
    name: str
    horizon: int
    use_proximity_safety: bool
    move_start_s: float
    obstacle_start_xy: tuple[float, float]
    obstacle_end_xy: tuple[float, float]
    move_duration_s: float = 0.5


class TimingControllerWrapper:
    def __init__(self, controller, represented_solve_ms: float = 50.0):
        self.controller = controller
        self.represented_solve_ms = represented_solve_ms
        self.raw_solve_ms_history: list[float] = []

    def solve(self, state, ref_preview, u_prev):
        result = self.controller.solve(state, ref_preview, u_prev)
        self.raw_solve_ms_history.append(result.solve_time_s * 1000.0)

        return SolveResult(
            status=result.status,
            plan=result.plan,
            solve_time_s=self.represented_solve_ms / 1000.0,
        )


def make_mpc_config(horizon: int) -> MPCConfig:
    cfg = MPCConfig()
    cfg.horizon = int(horizon)
    cfg.dt = DT
    return cfg


def minimum_safety_distance(safety, log) -> float:
    if safety is not None and getattr(safety, "distance_history", None):
        if len(safety.distance_history) > 0:
            return float(np.min(safety.distance_history))
    return float(log.min_clearance())


def minimum_speed_scale(safety) -> float:
    if safety is None or not getattr(safety, "scale_history", None):
        return 1.0
    if len(safety.scale_history) == 0:
        return 1.0
    return float(np.min(safety.scale_history))


def classify_outcome_mpc_only(log) -> str:
    min_distance = float(log.min_clearance())
    safe_stop_samples = int(np.sum(log.mode >= 3))

    r_safe = float(log.obstacle.r_safe)
    tol = 0.05
    clear_margin = 0.15

    if safe_stop_samples > 0:
        return "SUPERVISOR_STOPPED"

    if min_distance < r_safe - tol:
        return "VIOLATED"

    if min_distance < r_safe + clear_margin:
        return "TIGHT_AVOID"

    return "CLEAR_AVOID"


def classify_outcome_with_safety(log, safety) -> str:
    min_distance = minimum_safety_distance(safety, log)
    min_scale = minimum_speed_scale(safety)
    safe_stop_samples = int(np.sum(log.mode >= 3))

    if getattr(safety, "stop_latched", False):
        return "PROXIMITY_STOP"

    if safe_stop_samples > 0:
        return "SUPERVISOR_STOPPED"

    if min_scale < 0.99:
        return "AVOIDED_WITH_SLOWDOWN"

    r_safe = float(log.obstacle.r_safe)
    if min_distance < r_safe - 0.05:
        return "VIOLATED"

    return "CLEAR_AVOID"


def classify_outcome(log, safety, use_proximity_safety: bool) -> str:
    if use_proximity_safety:
        return classify_outcome_with_safety(log, safety)
    return classify_outcome_mpc_only(log)


def build_sim(spec: ScenarioSpec):
    model = SkidSteerModel()
    mpc_cfg = make_mpc_config(spec.horizon)
    sup_cfg = SupervisorConfig()
    ref = SCurveLaneChange(v_target=mpc_cfg.v_target)

    obs = TimedMovingObstacle(
        start_xy=spec.obstacle_start_xy,
        end_xy=spec.obstacle_end_xy,
        move_start_s=spec.move_start_s,
        move_duration_s=spec.move_duration_s,
    )

    raw_ctrl = ScipyMPC(model, mpc_cfg, obs)
    ctrl = TimingControllerWrapper(raw_ctrl, represented_solve_ms=REPRESENTED_SOLVE_MS)

    gov = CommandGovernor(sup_cfg, v0=mpc_cfg.v_target)

    safety = None
    if spec.use_proximity_safety:
        safety = ProximitySpeedLimiter(
            obs,
            ProximitySafetyConfig(
                stop_distance=1.0,
                slow_distance=4.0,
                latch_stop=True,
                use_smoothstep=True,
            ),
        )

    sim = ClosedLoopSimulator(
        model,
        ctrl,
        gov,
        ref,
        obs,
        mpc_cfg,
        SimConfig(nominal_solve_ms=None),
        safety_filter=safety,
    )

    return sim, ref, obs, safety, ctrl


def save_summary_row(spec: ScenarioSpec, log, safety, ctrl):
    min_distance = minimum_safety_distance(safety, log)
    min_scale = minimum_speed_scale(safety)
    outcome = classify_outcome(log, safety, spec.use_proximity_safety)

    row = {
        "name": spec.name,
        "expected_mode": spec.name,
        "classified_outcome": outcome,
        "horizon": spec.horizon,
        "use_proximity_safety": spec.use_proximity_safety,
        "move_start_s": spec.move_start_s,
        "min_distance_m": min_distance,
        "rms_error_cm": log.rms_cm(),
        "peak_error_cm": log.peak_cm(),
        "min_speed_scale": min_scale,
        "avg_raw_solve_time_ms": float(np.mean(ctrl.raw_solve_ms_history)) if ctrl.raw_solve_ms_history else 0.0,
        "max_raw_solve_time_ms": float(np.max(ctrl.raw_solve_ms_history)) if ctrl.raw_solve_ms_history else 0.0,
    }
    return row


def save_csv(rows: list[dict], path: Path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    scenarios = [
    ScenarioSpec(
        name="clear_avoid",
        horizon=40,
        use_proximity_safety=False,
        move_start_s=4.0,
        obstacle_start_xy=(10.0, -4.0),
        obstacle_end_xy=(10.0, 1.2),
        move_duration_s=0.5,
    ),
    ScenarioSpec(
        name="tight_avoid",
        horizon=30,
        use_proximity_safety=False,
        move_start_s=5.6,
        obstacle_start_xy=(10.0, -4.0),
        obstacle_end_xy=(10.0, 0.0),
        move_duration_s=0.5,
    ),
    ScenarioSpec(
        name="proximity_stop",
        horizon=30,
        use_proximity_safety=True,
        move_start_s=5.8,
        obstacle_start_xy=(10.0, -4.0),
        obstacle_end_xy=(10.0, 0.0),
        move_duration_s=0.25,
    ),
    ScenarioSpec(
        name="supervisor_stop",
        horizon=30,
        use_proximity_safety=False,
        move_start_s=6.5,
        obstacle_start_xy=(10.0, -4.0),
        obstacle_end_xy=(10.0, 0.0),
        move_duration_s=0.5,
    ),
]

    rows = []

    for spec in scenarios:
        print(f"Running {spec.name} ...")
        sim, ref, obs, safety, ctrl = build_sim(spec)
        log = sim.run()

        row = save_summary_row(spec, log, safety, ctrl)
        rows.append(row)

        print(
            f"  outcome={row['classified_outcome']} | "
            f"min_dist={row['min_distance_m']:.2f} m | "
            f"RMS={row['rms_error_cm']:.1f} cm | "
            f"avg_solve={row['avg_raw_solve_time_ms']:.1f} ms"
        )

        # Hook your existing GIF generation here.
        # Example:
        # save_replay_gif(log, ref, obs, OUTDIR / f"{spec.name}.gif")
        MpcAnimator(log, outdir=OUTDIR).save_gif(
        filename=f"{spec.name}_{row['classified_outcome'].lower()}.gif",
            fps=12,
            every=2,
            dpi=120,
        )

    save_csv(rows, OUTDIR / "exemplar_dynamic_obstacle_cases.csv")
    print("Wrote out/exemplar_dynamic_obstacle_cases.csv")


if __name__ == "__main__":
    main()