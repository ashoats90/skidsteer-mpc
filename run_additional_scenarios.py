"""Additional validation scenarios.

Runs:
1. Nominal S-curve demo.
2. Sinusoidal curved-track demo.
3. Sinusoidal track with proximity-based speed limiting and stop latch.

Outputs static plots and replay GIFs to out/.
"""
from __future__ import annotations

import time

from skidsteer_mpc import (
    SkidSteerModel,
    SCurveLaneChange,
    SinusoidalTrack,
    Obstacle,
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


def build(reference, mpc_cfg, sup_cfg, safety_filter=None):
    model = SkidSteerModel()
    obs = Obstacle()
    ctrl = ScipyMPC(model, mpc_cfg, obs)
    gov = CommandGovernor(sup_cfg, v0=mpc_cfg.v_target)
    return ClosedLoopSimulator(
        model,
        ctrl,
        gov,
        reference,
        obs,
        mpc_cfg,
        SimConfig(),
        safety_filter=safety_filter,
    )


def summarize(name, log, mpc_cfg):
    print(
        f"[{name}] RMS={log.rms_cm():.1f}cm "
        f"peak={log.peak_cm():.1f}cm "
        f"sat={log.saturated_samples(mpc_cfg.v_max)} "
        f"min_distance_to_obstacle_center={log.min_clearance():.3f}m"
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


def main():
    mpc_cfg = MPCConfig()
    sup_cfg = SupervisorConfig()

    scenarios = []

    scenarios.append((
        "s_curve_nominal",
        SCurveLaneChange(v_target=mpc_cfg.v_target),
        None,
    ))

    scenarios.append((
        "sinusoidal_nominal",
        SinusoidalTrack(
            v_target=mpc_cfg.v_target,
            amplitude=1.2,
            wavelength=12.0,
            length=35.0,
        ),
        None,
    ))

    # Safety scenario:
    # - slow below 4 m from obstacle center
    # - latch stop at 1 m from obstacle center
    safety_obs = Obstacle()
    limiter = ProximitySpeedLimiter(
        safety_obs,
        ProximitySafetyConfig(
            stop_distance=1.5,
            slow_distance=5.0,
            latch_stop=True,
            use_smoothstep=True,
        ),
    )
    scenarios.append((
        "sinusoidal_proximity_stop",
        SinusoidalTrack(
            v_target=mpc_cfg.v_target,
            amplitude=1.2,
            wavelength=12.0,
            length=35.0,
        ),
        limiter,
    ))

    for name, ref, safety_filter in scenarios:
        t0 = time.time()
        sim = build(ref, mpc_cfg, sup_cfg, safety_filter=safety_filter)
        log = sim.run()
        print(f"{name} completed in {time.time() - t0:.1f}s")
        summarize(name, log, mpc_cfg)
        save_outputs(name, log, mpc_cfg)

        if safety_filter is not None:
            print(
                f"  final safety distance={safety_filter.last_distance:.3f}m, "
                f"final speed scale={safety_filter.last_scale:.2f}, "
                f"stop_latched={safety_filter.stop_latched}"
            )

    print(f"outputs written to {OUTDIR}/")


if __name__ == "__main__":
    main()
