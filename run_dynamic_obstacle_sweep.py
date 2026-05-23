"""Dynamic obstacle timing sweep.

This experiment sweeps the time at which a moving obstacle enters the vehicle's
path. It characterizes the transition between:

- clean avoidance
- avoidance with proximity-based slowdown
- proximity stop
- safety threshold violation

The script writes a CSV and a simple summary plot to out/.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from skidsteer_mpc import (
    SkidSteerModel,
    SCurveLaneChange,
    TimedMovingObstacle,
    ScipyMPC,
    CommandGovernor,
    ClosedLoopSimulator,
    MPCConfig,
    SupervisorConfig,
    SimConfig,
)
from skidsteer_mpc.safety import ProximitySafetyConfig, ProximitySpeedLimiter
from skidsteer_mpc.controller import SolveResult


OUTDIR = Path("out")
OUTDIR.mkdir(exist_ok=True)

STOP_DISTANCE_M = 1.0
SLOW_DISTANCE_M = 4.0

class TimingControllerWrapper:
    """Records raw controller solve time while feeding represented timing to supervisor."""

    def __init__(self, controller, represented_solve_ms: float = 50.0):
        self.controller = controller
        self.represented_solve_ms = represented_solve_ms
        self.raw_solve_ms_history = []

    def solve(self, state, ref_preview, u_prev):
        result = self.controller.solve(state, ref_preview, u_prev)

        # Store actual SLSQP wall-clock solve time for analysis.
        self.raw_solve_ms_history.append(result.solve_time_s * 1000.0)

        # Return same plan/status, but with represented solve time so supervisor
        # behavior is not dominated by Python/SLSQP runtime.
        return SolveResult(
            status=result.status,
            plan=result.plan,
            solve_time_s=self.represented_solve_ms / 1000.0,
        )


def build_sim(move_start_s: float, mpc_cfg: MPCConfig, sup_cfg: SupervisorConfig):
    """Build one dynamic-obstacle simulation for a given obstacle entry time."""
    model = SkidSteerModel()
    ref = SCurveLaneChange(v_target=mpc_cfg.v_target)

    obs = TimedMovingObstacle(
        start_xy=(10.0, -4.0),
        end_xy=(10.0, 0.0),
        move_start_s=move_start_s,
        move_duration_s=0.5,
    )

    raw_ctrl = ScipyMPC(model, mpc_cfg, obs)
    ctrl = TimingControllerWrapper(raw_ctrl, represented_solve_ms=50.0)
    gov = CommandGovernor(sup_cfg, v0=mpc_cfg.v_target)

    safety = ProximitySpeedLimiter(
        obs,
        ProximitySafetyConfig(
            stop_distance=STOP_DISTANCE_M,
            slow_distance=SLOW_DISTANCE_M,
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
        SimConfig(),
        safety_filter=safety,
    )

    return sim, safety, mpc_cfg, ctrl


def minimum_speed_scale(safety: ProximitySpeedLimiter) -> float:
    """Return the minimum commanded safety scale over the run."""
    if not safety.scale_history:
        return 1.0
    return float(np.min(safety.scale_history))


def minimum_safety_distance(safety: ProximitySpeedLimiter, log) -> float:
    """Prefer safety-filter history, fall back to SimLog min clearance."""
    if safety.distance_history:
        return float(np.min(safety.distance_history))
    return float(log.min_clearance())


def classify_outcome(
    log,
    safety: ProximitySpeedLimiter,
    stop_distance: float = STOP_DISTANCE_M,
    tol: float = 0.05,
) -> str:
    """Classify the run outcome.

    The key change from the first version is that successful avoidance and
    slowdown are not treated as mutually exclusive. A vehicle can avoid the
    obstacle successfully while still triggering the conservative speed envelope.
    """
    min_distance = minimum_safety_distance(safety, log)
    min_scale = minimum_speed_scale(safety)
    safe_stop_samples = int(np.sum(log.mode >= 3))
    stopped_by_supervisor = safe_stop_samples > 0

    # True violation: crossed the stop threshold without the intended stop
    # behavior engaging. The tolerance avoids labeling tiny discrete-time
    # numerical overshoots as failures.
    if min_distance < stop_distance - tol and not safety.stop_latched:
        return "VIOLATED"

    if safety.stop_latched or stopped_by_supervisor:
        return "STOPPED"

    if min_scale < 0.99:
        return "AVOIDED_WITH_SLOWDOWN"

    return "AVOIDED_CLEANLY"


def run_case(move_start_s: float, mpc_cfg: MPCConfig, sup_cfg: SupervisorConfig) -> dict:
    """Run one sweep case and return a metrics row."""
    sim, safety, mpc_cfg, ctrl = build_sim(move_start_s, horizon, sup_cfg)
    log = sim.run()

    raw_solve_ms = np.asarray(ctrl.raw_solve_ms_history, dtype=float)

    avg_solve_ms = float(np.mean(raw_solve_ms))
    max_solve_ms = float(np.max(raw_solve_ms))
    p95_solve_ms = float(np.percentile(raw_solve_ms, 95))

    speed = log.z[:, 3]
    min_distance = minimum_safety_distance(safety, log)
    min_scale = minimum_speed_scale(safety)
    safe_stop_samples = int(np.sum(log.mode >= 3))

    return {
        "move_start_s": move_start_s,
        "outcome": classify_outcome(log, safety),
        "min_distance_m": min_distance,
        "min_margin_to_stop_m": min_distance - STOP_DISTANCE_M,
        "rms_error_cm": log.rms_cm(),
        "peak_error_cm": log.peak_cm(),
        "stop_latched": safety.stop_latched,
        "safe_stop_samples": safe_stop_samples,
        "min_speed_mps": float(np.min(speed)),
        "final_speed_mps": float(speed[-1]),
        "min_speed_scale": min_scale,
        "final_speed_scale": float(safety.last_scale),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_sweep(rows: list[dict], path: Path) -> None:
    """Create a simple timing sweep plot for presentation/debugging."""
    xs = np.array([r["move_start_s"] for r in rows], dtype=float)
    min_dist = np.array([r["min_distance_m"] for r in rows], dtype=float)
    min_scale = np.array([r["min_speed_scale"] for r in rows], dtype=float)

    fig, ax1 = plt.subplots(figsize=(8.0, 4.6))

    ax1.plot(xs, min_dist, marker="o", linewidth=2.0, label="minimum obstacle distance")
    ax1.axhline(STOP_DISTANCE_M, linestyle="--", linewidth=1.5, label="stop threshold")
    ax1.axhline(SLOW_DISTANCE_M, linestyle=":", linewidth=1.5, label="slowdown threshold")
    ax1.set_xlabel("Obstacle move start time [s]")
    ax1.set_ylabel("Minimum distance to obstacle center [m]")
    ax1.grid(True, alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(xs, min_scale, marker="s", linewidth=2.0, label="minimum speed scale")
    ax2.set_ylabel("Minimum safety speed scale [-]")
    ax2.set_ylim(-0.05, 1.05)

    # Outcome labels
    for r in rows:
        ax1.annotate(
            r["outcome"].replace("_", "\\n"),
            xy=(r["move_start_s"], r["min_distance_m"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    ax1.set_title("Dynamic Obstacle Timing Sweep")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    mpc_cfg = MPCConfig()
    sup_cfg = SupervisorConfig()

    # Sweep broadly, with denser samples around the transition you observed.
    move_start_values = [
        2.0,
        3.0,
        4.0,
        5.0,
        5.2,
        5.4,
        5.6,
        5.8,
        6.0,
        6.5,
        7.0,
    ]

    rows = []

    for move_start_s in move_start_values:
        print(f"Running move_start_s={move_start_s:.1f}...")
        result = run_case(move_start_s, mpc_cfg, sup_cfg)
        rows.append(result)

        print(
            f"  {result['outcome']} | "
            f"min_dist={result['min_distance_m']:.3f} m | "
            f"min_scale={result['min_speed_scale']:.2f} | "
            f"RMS={result['rms_error_cm']:.1f} cm | "
            f"stop_latched={result['stop_latched']}"
        )

    csv_path = OUTDIR / "dynamic_obstacle_timing_sweep.csv"
    plot_path = OUTDIR / "dynamic_obstacle_timing_sweep.png"

    write_csv(rows, csv_path)
    plot_sweep(rows, plot_path)

    print(f"\\nWrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
