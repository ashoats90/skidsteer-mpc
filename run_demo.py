"""Demo / validation entry point for the skidsteer_mpc package.

Runs two closed-loop simulations through the same OO pipeline:
  - nominal (no faults)        -> plots 1-4
  - supervised (injected faults) -> plots 5-6
and prints the headline metrics so the refactor can be checked against the
original procedural results.
"""
import time
from skidsteer_mpc import (SkidSteerModel, SCurveLaneChange, Obstacle, ScipyMPC,
                           CommandGovernor, ClosedLoopSimulator, MpcPlotter,
                           MPCConfig, SupervisorConfig, SimConfig, FaultInjector)

OUTDIR = "out"


def build(mpc_cfg, sup_cfg, faults=None):
    model = SkidSteerModel()
    ref = SCurveLaneChange(v_target=mpc_cfg.v_target)
    obs = Obstacle()
    ctrl = ScipyMPC(model, mpc_cfg, obs)
    gov = CommandGovernor(sup_cfg, v0=mpc_cfg.v_target)
    return ClosedLoopSimulator(model, ctrl, gov, ref, obs, mpc_cfg,
                               SimConfig(), faults)


def main():
    mpc_cfg = MPCConfig()
    sup_cfg = SupervisorConfig()

    # ---- nominal run -> plots 1-4 ----
    t0 = time.time()
    log = build(mpc_cfg, sup_cfg).run()
    print(f"[nominal] {time.time()-t0:.1f}s | RMS={log.rms_cm():.1f}cm "
          f"peak={log.peak_cm():.1f}cm sat={log.saturated_samples(mpc_cfg.v_max)} "
          f"min_clear={log.min_clearance():.3f}m")
    p = MpcPlotter(log, outdir=OUTDIR)
    p.path_tracking(); p.control_inputs(mpc_cfg.v_max, mpc_cfg.v_min)
    p.tracking_error(); p.attitude()

    # ---- supervised run with injected faults -> plots 5-6 ----
    SHORT, LONG = range(60, 64), range(100, 118)
    faults = FaultInjector(timeout_steps=SHORT, fail_steps=LONG)
    t0 = time.time()
    slog = build(mpc_cfg, sup_cfg, faults).run()
    n_deg = int((slog.mode == 2).sum()); n_stop = int((slog.mode >= 3).sum())
    print(f"[supervised] {time.time()-t0:.1f}s | DEGRADED={n_deg} SAFE_STOP/FAULT={n_stop} "
          f"min_clear={slog.min_clearance():.3f}m "
          f"({'MAINTAINED' if slog.min_clearance() >= slog.obstacle.r_safe - 1e-3 else 'VIOLATED'})")
    ps = MpcPlotter(slog, outdir=OUTDIR)
    ps.supervised_trajectory()
    ps.supervisor_timeline(mpc_cfg.v_max, mpc_cfg.v_min,
                           fault_windows=[(list(SHORT), "transient outage"),
                                          (list(LONG), "sustained outage")])
    print("plots written to", OUTDIR)


if __name__ == "__main__":
    main()
