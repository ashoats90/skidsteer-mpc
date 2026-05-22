"""Dynamic obstacle validation scenarios.

Scenario A:
    Obstacle moves into the vehicle path early enough that the MPC can plan
    around it.

Scenario B:
    Obstacle moves into the vehicle path late, leaving too little time for a
    comfortable avoidance maneuver. The proximity safety filter slows the
    vehicle and latches a stop at 1.0 m.

Outputs plots and replay GIFs to out/.
"""
from __future__ import annotations

import time

from skidsteer_mpc import (
    SkidSteerModel,
    SCurveLaneChange,
    TimedMovingObstacle,
    ScipyMPC,
    CommandGovernor,
    ClosedLoopSimulator,
    MpcPlotter,
    MpcAnimator,
    MPCConfig,
    SupervisorConfig,
    SimConfig,
)
from skidsteer_mpc.safety import ProximitySafetyConfig, ProximitySpeedLimiter


OUTDIR = "out"


def build(reference, obstacle, mpc_cfg, sup_cfg, safety_filter=None):
    model = SkidSteerModel()
    ctrl = ScipyMPC(model, mpc_cfg, obstacle)
    gov = CommandGovernor(sup_cfg, v0=mpc_cfg.v_target)

    return ClosedLoopSimulator(
        model,
        ctrl,
        gov,
        reference,
        obstacle,
        mpc_cfg,
        SimConfig(),
        safety_filter=safety_filter,
    )


def save_outputs(name, log, mpc_cfg):
    p = MpcPlotter(log, outdir=OUTDIR)
    p.path_tracking(name=f"{name}_path_tracking.png")
    p.control_inputs(mpc_cfg.v_max, mpc_cfg.v_min, name=f"{name}_control_inputs.png")
    p.tracking_error(name=f"{name}_tracking_error.png")
    p.attitude(name=f"{name}_attitude.png")

    MpcAnimator(log, outdir=OUTDIR).save_gif(
        filename=f"{name}_replay.gif",
        fps=12,
        every=2,
        dpi=120,
    )


def summarize(name, log, safety_filter=None):
    stopped = bool((log.mode >= 3).any())
    print(
        f"[{name}] RMS={log.rms_cm():.1f}cm "
        f"peak={log.peak_cm():.1f}cm "
        f"min_distance_to_obstacle_center={log.min_clearance():.3f}m "
        f"stopped={stopped}"
    )

    if safety_filter is not None:
        print(
            f"    safety_stop_latched={safety_filter.stop_latched} "
            f"final_scale={safety_filter.last_scale:.2f} "
            f"final_distance={safety_filter.last_distance:.3f}m"
        )


def main():
    mpc_cfg = MPCConfig()
    sup_cfg = SupervisorConfig()

    # Obstacle ends directly on the nominal early part of the S-curve path.
    # Early appearance gives the MPC time to plan around it.
    early_obs = TimedMovingObstacle(
        start_xy=(10.0, -4.0),
        end_xy=(10.0, 0.0),
        move_start_s=2.0,
        move_duration_s=2.0,
    )
    early_ref = SCurveLaneChange(v_target=mpc_cfg.v_target)
    early_sim = build(early_ref, early_obs, mpc_cfg, sup_cfg)

    t0 = time.time()
    early_log = early_sim.run()
    print(f"dynamic_obstacle_early_avoid completed in {time.time() - t0:.1f}s")
    summarize("dynamic_obstacle_early_avoid", early_log)
    save_outputs("dynamic_obstacle_early_avoid", early_log, mpc_cfg)

    # Late appearance: obstacle moves into the path when the vehicle is already
    # close. Keep the stop threshold at 1.0 m and slow down below 4.0 m.
    late_obs = TimedMovingObstacle(
        start_xy=(10.0, -4.0),
        end_xy=(10.0, 0.0),
        move_start_s=5.4,
        move_duration_s=0.5,
    )
    late_ref = SCurveLaneChange(v_target=mpc_cfg.v_target)
    late_limiter = ProximitySpeedLimiter(
        late_obs,
        ProximitySafetyConfig(
            stop_distance=1.0,
            slow_distance=4.0,
            latch_stop=True,
            use_smoothstep=True,
        ),
    )
    late_sim = build(late_ref, late_obs, mpc_cfg, sup_cfg, safety_filter=late_limiter)

    t0 = time.time()
    late_log = late_sim.run()
    print(f"dynamic_obstacle_late_stop completed in {time.time() - t0:.1f}s")
    summarize("dynamic_obstacle_late_stop", late_log, late_limiter)
    save_outputs("dynamic_obstacle_late_stop", late_log, mpc_cfg)

    print(f"outputs written to {OUTDIR}/")


if __name__ == "__main__":
    main()
