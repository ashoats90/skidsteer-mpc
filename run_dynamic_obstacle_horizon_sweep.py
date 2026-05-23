"""Dynamic obstacle timing x MPC horizon sweep.

This version separates two validation questions:

1. MPC-only horizon sweep:
   - Proximity safety filter disabled.
   - Tests whether longer MPC horizon improves obstacle avoidance behavior.

2. Optional safety-enabled sweep:
   - Proximity safety filter enabled.
   - Tests whether the safety envelope slows/stops near the obstacle.

By default:

    USE_PROXIMITY_SAFETY = False

That is the right setting for testing the horizon effect without the safety
filter dominating the outcome.

Run:
    python run_dynamic_obstacle_horizon_sweep.py
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
from skidsteer_mpc.controller import SolveResult
from skidsteer_mpc.safety import ProximitySafetyConfig, ProximitySpeedLimiter


OUTDIR = Path("out")
OUTDIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment switches
# ---------------------------------------------------------------------------

# For the horizon sweep, leave this False.
# This isolates the MPC obstacle-avoidance behavior.
#
# Set this True only when you want to test the proximity safety layer's effect.
USE_PROXIMITY_SAFETY = False

# This wrapper records raw SLSQP solve time but feeds a represented solve time
# to the supervisor so Python/SLSQP runtime does not dominate behavior.
USE_REPRESENTED_SOLVE_TIME_FOR_SUPERVISOR = True
REPRESENTED_SOLVE_MS = 50.0

# Safety-envelope thresholds.
STOP_DISTANCE_M = 1.0
SLOW_DISTANCE_M = 4.0

# Keep dt fixed so horizon changes are easy to interpret as longer lookahead.
DT = 1.0 / 6.0

# Obstacle path: starts below the nominal path and moves into it.
OBSTACLE_START_XY = (10.0, -4.0)
OBSTACLE_END_XY = (10.0, 0.0)
OBSTACLE_MOVE_DURATION_S = 0.5


# ---------------------------------------------------------------------------
# Outcome labels
# ---------------------------------------------------------------------------

if USE_PROXIMITY_SAFETY:
    OUTCOME_SCORE = {
        "VIOLATED": 0,
        "PROXIMITY_STOPPED": 1,
        "SUPERVISOR_STOPPED": 2,
        "AVOIDED_WITH_SLOWDOWN": 3,
        "AVOIDED_CLEANLY": 4,
    }

    OUTCOME_LABEL = {
        0: "VIOLATED",
        1: "PROX\nSTOP",
        2: "SUP\nSTOP",
        3: "SLOWED",
        4: "CLEAN",
    }
else:
    OUTCOME_SCORE = {
        "VIOLATED": 0,
        "SUPERVISOR_STOPPED": 1,
        "TIGHT_AVOID": 2,
        "CLEAR_AVOID": 3,
    }

    OUTCOME_LABEL = {
        0: "VIOLATED",
        1: "SUP\nSTOP",
        2: "TIGHT",
        3: "CLEAR",
    }


class TimingControllerWrapper:
    """Record raw solve time while feeding represented timing to the supervisor."""

    def __init__(self, controller, represented_solve_ms: float = 50.0):
        self.controller = controller
        self.represented_solve_ms = represented_solve_ms
        self.raw_solve_ms_history: list[float] = []

    def solve(self, state, ref_preview, u_prev):
        result = self.controller.solve(state, ref_preview, u_prev)

        raw_solve_ms = result.solve_time_s * 1000.0
        self.raw_solve_ms_history.append(raw_solve_ms)

        if not USE_REPRESENTED_SOLVE_TIME_FOR_SUPERVISOR:
            return result

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


def build_sim(move_start_s: float, horizon: int, sup_cfg: SupervisorConfig):
    mpc_cfg = make_mpc_config(horizon)

    model = SkidSteerModel()
    ref = SCurveLaneChange(v_target=mpc_cfg.v_target)

    obs = TimedMovingObstacle(
        start_xy=OBSTACLE_START_XY,
        end_xy=OBSTACLE_END_XY,
        move_start_s=move_start_s,
        move_duration_s=OBSTACLE_MOVE_DURATION_S,
    )

    raw_ctrl = ScipyMPC(model, mpc_cfg, obs)
    ctrl = TimingControllerWrapper(raw_ctrl, represented_solve_ms=REPRESENTED_SOLVE_MS)

    gov = CommandGovernor(sup_cfg, v0=mpc_cfg.v_target)

    safety = None
    if USE_PROXIMITY_SAFETY:
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
        SimConfig(nominal_solve_ms=None),
        safety_filter=safety,
    )

    return sim, safety, mpc_cfg, ctrl


def raw_solve_time_metrics_ms(ctrl: TimingControllerWrapper) -> tuple[float, float, float]:
    solve_ms = np.asarray(ctrl.raw_solve_ms_history, dtype=float)

    if solve_ms.size == 0:
        return 0.0, 0.0, 0.0

    return (
        float(np.mean(solve_ms)),
        float(np.max(solve_ms)),
        float(np.percentile(solve_ms, 95)),
    )


def minimum_speed_scale(safety: ProximitySpeedLimiter | None) -> float:
    if safety is None or not safety.scale_history:
        return 1.0
    return float(np.min(safety.scale_history))


def minimum_safety_distance(safety: ProximitySpeedLimiter | None, log) -> float:
    if safety is not None and safety.distance_history:
        return float(np.min(safety.distance_history))
    return float(log.min_clearance())


def classify_outcome_mpc_only(log) -> str:
    """Classify MPC-only obstacle avoidance.

    Uses the obstacle safety radius, not the proximity-slowdown distance.
    """
    min_distance = float(log.min_clearance())
    safe_stop_samples = int(np.sum(log.mode >= 3))

    r_safe = float(log.obstacle.r_safe)
    tol = 0.05
    close_margin = 0.20

    if safe_stop_samples > 0:
        return "SUPERVISOR_STOPPED"

    if min_distance < r_safe - tol:
        return "VIOLATED"

    if min_distance < r_safe + close_margin:
        return "TIGHT_AVOID"

    return "CLEAR_AVOID"


def classify_outcome_with_safety(
    log,
    safety: ProximitySpeedLimiter,
    stop_distance: float = STOP_DISTANCE_M,
    tol: float = 0.05,
) -> str:
    min_distance = minimum_safety_distance(safety, log)
    min_scale = minimum_speed_scale(safety)
    safe_stop_samples = int(np.sum(log.mode >= 3))
    stopped_by_supervisor = safe_stop_samples > 0

    if min_distance < stop_distance - tol and not safety.stop_latched:
        return "VIOLATED"

    if safety.stop_latched:
        return "PROXIMITY_STOPPED"

    if stopped_by_supervisor:
        return "SUPERVISOR_STOPPED"

    if min_scale < 0.99:
        return "AVOIDED_WITH_SLOWDOWN"

    return "AVOIDED_CLEANLY"


def classify_outcome(log, safety: ProximitySpeedLimiter | None) -> str:
    if USE_PROXIMITY_SAFETY:
        if safety is None:
            raise RuntimeError("USE_PROXIMITY_SAFETY=True but safety filter is None")
        return classify_outcome_with_safety(log, safety)

    return classify_outcome_mpc_only(log)


def run_case(move_start_s: float, horizon: int, sup_cfg: SupervisorConfig) -> dict:
    sim, safety, mpc_cfg, ctrl = build_sim(move_start_s, horizon, sup_cfg)
    log = sim.run()

    speed = log.z[:, 3]
    min_distance = minimum_safety_distance(safety, log)
    min_scale = minimum_speed_scale(safety)
    avg_solve_ms, max_solve_ms, p95_solve_ms = raw_solve_time_metrics_ms(ctrl)
    safe_stop_samples = int(np.sum(log.mode >= 3))
    outcome = classify_outcome(log, safety)

    stop_latched = bool(safety.stop_latched) if safety is not None else False
    final_speed_scale = float(safety.last_scale) if safety is not None else 1.0

    return {
        "move_start_s": float(move_start_s),
        "horizon": int(horizon),
        "lookahead_s": float(horizon * mpc_cfg.dt),
        "use_proximity_safety": bool(USE_PROXIMITY_SAFETY),
        "outcome": outcome,
        "outcome_score": OUTCOME_SCORE[outcome],
        "min_distance_m": min_distance,
        "min_margin_to_stop_m": min_distance - STOP_DISTANCE_M,
        "rms_error_cm": log.rms_cm(),
        "peak_error_cm": log.peak_cm(),
        "stop_latched": stop_latched,
        "safe_stop_samples": safe_stop_samples,
        "min_speed_mps": float(np.min(speed)),
        "final_speed_mps": float(speed[-1]),
        "min_speed_scale": min_scale,
        "final_speed_scale": final_speed_scale,
        "avg_raw_solve_time_ms": avg_solve_ms,
        "max_raw_solve_time_ms": max_solve_ms,
        "p95_raw_solve_time_ms": p95_solve_ms,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_grid(
    rows: list[dict],
    move_start_values: list[float],
    horizon_values: list[int],
    key: str,
) -> np.ndarray:
    lookup = {
        (float(r["move_start_s"]), int(r["horizon"])): r
        for r in rows
    }

    grid = np.full((len(move_start_values), len(horizon_values)), np.nan)

    for i, move_start_s in enumerate(move_start_values):
        for j, horizon in enumerate(horizon_values):
            row = lookup[(float(move_start_s), int(horizon))]
            grid[i, j] = row[key]

    return grid


def plot_outcome_grid(
    rows: list[dict],
    move_start_values: list[float],
    horizon_values: list[int],
    path: Path,
) -> None:
    grid = make_grid(rows, move_start_values, horizon_values, "outcome_score")

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    vmax = max(OUTCOME_SCORE.values())
    im = ax.imshow(grid, aspect="auto", origin="lower", vmin=0, vmax=vmax)

    ax.set_xticks(np.arange(len(horizon_values)))
    ax.set_xticklabels([str(h) for h in horizon_values])
    ax.set_yticks(np.arange(len(move_start_values)))
    ax.set_yticklabels([f"{v:.1f}" for v in move_start_values])

    ax.set_xlabel("MPC horizon [steps]")
    ax.set_ylabel("Obstacle move start time [s]")

    mode = "Safety enabled" if USE_PROXIMITY_SAFETY else "MPC only"
    ax.set_title(f"Dynamic Obstacle Sweep: Outcome vs. Horizon ({mode})")

    for i, _move_start_s in enumerate(move_start_values):
        for j, _horizon in enumerate(horizon_values):
            score = int(grid[i, j])
            label = OUTCOME_LABEL[score]
            ax.text(j, i, label, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks(list(OUTCOME_LABEL.keys()))
    cbar.set_ticklabels([OUTCOME_LABEL[k].replace("\n", " ") for k in OUTCOME_LABEL.keys()])

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_metric_grid(
    rows: list[dict],
    move_start_values: list[float],
    horizon_values: list[int],
    key: str,
    title: str,
    cbar_label: str,
    path: Path,
    fmt: str = ".2f",
) -> None:
    grid = make_grid(rows, move_start_values, horizon_values, key)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    im = ax.imshow(grid, aspect="auto", origin="lower")

    ax.set_xticks(np.arange(len(horizon_values)))
    ax.set_xticklabels([str(h) for h in horizon_values])
    ax.set_yticks(np.arange(len(move_start_values)))
    ax.set_yticklabels([f"{v:.1f}" for v in move_start_values])

    ax.set_xlabel("MPC horizon [steps]")
    ax.set_ylabel("Obstacle move start time [s]")
    ax.set_title(title)

    for i in range(len(move_start_values)):
        for j in range(len(horizon_values)):
            ax.text(j, i, format(grid[i, j], fmt), ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_solve_time_summary(
    rows: list[dict],
    horizon_values: list[int],
    path: Path,
) -> None:
    avg_by_horizon = []
    max_by_horizon = []
    p95_by_horizon = []

    for horizon in horizon_values:
        subset = [r for r in rows if int(r["horizon"]) == int(horizon)]
        avg_by_horizon.append(float(np.mean([r["avg_raw_solve_time_ms"] for r in subset])))
        max_by_horizon.append(float(np.max([r["max_raw_solve_time_ms"] for r in subset])))
        p95_by_horizon.append(float(np.mean([r["p95_raw_solve_time_ms"] for r in subset])))

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(horizon_values, avg_by_horizon, marker="o", linewidth=2.0, label="average raw solve time")
    ax.plot(horizon_values, p95_by_horizon, marker="s", linewidth=2.0, label="mean p95 raw solve time")
    ax.plot(horizon_values, max_by_horizon, marker="^", linewidth=2.0, label="max raw solve time")

    ax.set_xlabel("MPC horizon [steps]")
    ax.set_ylabel("Raw solve time [ms]")
    ax.set_title("Raw Solve Time vs. MPC Horizon")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def print_summary(rows: list[dict]) -> None:
    print("\nOutcome counts by horizon:")
    horizons = sorted(set(int(r["horizon"]) for r in rows))

    for horizon in horizons:
        subset = [r for r in rows if int(r["horizon"]) == horizon]
        counts = {}
        for r in subset:
            counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1

        avg_solve = float(np.mean([r["avg_raw_solve_time_ms"] for r in subset]))
        max_solve = float(np.max([r["max_raw_solve_time_ms"] for r in subset]))
        p95_solve = float(np.mean([r["p95_raw_solve_time_ms"] for r in subset]))

        parts = [f"{k}={v}" for k, v in sorted(counts.items())]
        print(
            f"  horizon={horizon:>2}: "
            + ", ".join(parts)
            + f" | raw avg={avg_solve:.1f}ms, raw p95={p95_solve:.1f}ms, raw max={max_solve:.1f}ms"
        )


def main():
    sup_cfg = SupervisorConfig()

    move_start_values = [
        4.5,
        5.0,
        5.2,
        5.4,
        5.6,
        5.8,
        6.0,
        6.2,
        6.5,
    ]

    horizon_values = [20, 30, 40, 50]

    rows = []

    total = len(move_start_values) * len(horizon_values)
    case_num = 0

    mode = "safety-enabled" if USE_PROXIMITY_SAFETY else "MPC-only"
    print(f"Running dynamic obstacle horizon sweep in {mode} mode")

    for horizon in horizon_values:
        for move_start_s in move_start_values:
            case_num += 1
            lookahead_s = horizon * DT

            print(
                f"[{case_num:02d}/{total}] "
                f"horizon={horizon} ({lookahead_s:.2f}s), "
                f"move_start_s={move_start_s:.1f}"
            )

            result = run_case(move_start_s, horizon, sup_cfg)
            rows.append(result)

            print(
                f"    {result['outcome']} | "
                f"min_dist={result['min_distance_m']:.3f}m | "
                f"min_scale={result['min_speed_scale']:.2f} | "
                f"RMS={result['rms_error_cm']:.1f}cm | "
                f"raw_avg_solve={result['avg_raw_solve_time_ms']:.1f}ms | "
                f"raw_max_solve={result['max_raw_solve_time_ms']:.1f}ms | "
                f"stop={result['stop_latched']}"
            )

    suffix = "safety" if USE_PROXIMITY_SAFETY else "mpc_only"

    csv_path = OUTDIR / f"dynamic_obstacle_horizon_sweep_{suffix}.csv"
    outcome_path = OUTDIR / f"dynamic_obstacle_horizon_outcome_grid_{suffix}.png"
    distance_path = OUTDIR / f"dynamic_obstacle_horizon_min_distance_{suffix}.png"
    avg_solve_path = OUTDIR / f"dynamic_obstacle_horizon_avg_raw_solve_time_{suffix}.png"
    max_solve_path = OUTDIR / f"dynamic_obstacle_horizon_max_raw_solve_time_{suffix}.png"
    solve_summary_path = OUTDIR / f"dynamic_obstacle_horizon_solve_time_summary_{suffix}.png"

    write_csv(rows, csv_path)

    plot_outcome_grid(rows, move_start_values, horizon_values, outcome_path)

    plot_metric_grid(
        rows,
        move_start_values,
        horizon_values,
        key="min_distance_m",
        title=f"Dynamic Obstacle Sweep: Minimum Obstacle Distance ({mode})",
        cbar_label="Minimum distance to obstacle center [m]",
        path=distance_path,
        fmt=".2f",
    )

    plot_metric_grid(
        rows,
        move_start_values,
        horizon_values,
        key="avg_raw_solve_time_ms",
        title=f"Dynamic Obstacle Sweep: Average Raw Solve Time ({mode})",
        cbar_label="Average raw solve time [ms]",
        path=avg_solve_path,
        fmt=".1f",
    )

    plot_metric_grid(
        rows,
        move_start_values,
        horizon_values,
        key="max_raw_solve_time_ms",
        title=f"Dynamic Obstacle Sweep: Max Raw Solve Time ({mode})",
        cbar_label="Max raw solve time [ms]",
        path=max_solve_path,
        fmt=".1f",
    )

    plot_solve_time_summary(rows, horizon_values, solve_summary_path)

    print_summary(rows)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {outcome_path}")
    print(f"Wrote {distance_path}")
    print(f"Wrote {avg_solve_path}")
    print(f"Wrote {max_solve_path}")
    print(f"Wrote {solve_summary_path}")


if __name__ == "__main__":
    main()
