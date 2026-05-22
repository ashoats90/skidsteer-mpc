"""Small example showing how to save a replay GIF.

Run after your normal simulation dependencies are installed:

    python make_replay_gif.py
"""
from skidsteer_mpc import (
    SkidSteerModel,
    SCurveLaneChange,
    Obstacle,
    ScipyMPC,
    CommandGovernor,
    ClosedLoopSimulator,
    MPCConfig,
    SupervisorConfig,
    SimConfig,
    FaultInjector,
    MpcAnimator,
)


def build(mpc_cfg, sup_cfg, faults=None):
    model = SkidSteerModel()
    ref = SCurveLaneChange(v_target=mpc_cfg.v_target)
    obs = Obstacle()
    ctrl = ScipyMPC(model, mpc_cfg, obs)
    gov = CommandGovernor(sup_cfg, v0=mpc_cfg.v_target)
    return ClosedLoopSimulator(model, ctrl, gov, ref, obs, mpc_cfg, SimConfig(), faults)


def main():
    mpc_cfg = MPCConfig()
    sup_cfg = SupervisorConfig()

    # Nominal replay
    log = build(mpc_cfg, sup_cfg).run()
    MpcAnimator(log, outdir="out").save_gif(
        filename="nominal_replay.gif",
        fps=12,
        every=2,
        dpi=120,
    )

    # Supervised replay with injected faults
    short_fault = range(60, 64)
    long_fault = range(100, 118)
    faults = FaultInjector(timeout_steps=short_fault, fail_steps=long_fault)
    slog = build(mpc_cfg, sup_cfg, faults).run()
    MpcAnimator(slog, outdir="out").save_gif(
        filename="supervised_replay.gif",
        fps=12,
        every=2,
        dpi=120,
    )

    print("GIFs written to out/")


if __name__ == "__main__":
    main()
