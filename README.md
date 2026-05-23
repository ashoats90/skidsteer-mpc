# skidsteer_mpc

Object-oriented skid-steer nonlinear MPC with a safety supervisor and a CAN
output layer. Refactor of the original procedural scripts into a small package
with clear seams. **Illustrative simulation — not hardware data.**

## Architecture

```
config.py       dataclasses: ModelParams, MPCConfig, SupervisorConfig,
                ObstacleSpec, CanConfig, SimConfig, PlotStyle   (all tuning here)
model.py        SkidSteerModel — 7-state dynamics written ONCE, evaluated both
                numerically (numpy) and symbolically (CasADi RK4 integrator)
reference.py    ReferencePath ABC + SCurveLaneChange / StraightWithSwerve /
                ConstantRadiusCircle   (arc-length sampling + horizon preview)
obstacle.py     Obstacle — distance helpers usable in numpy and CasADi
controller.py   Controller ABC + ScipyMPC (SLSQP) and CasadiMPC (IPOPT);
                SolveResult / SolveStatus
supervisor.py   CommandGovernor — fault state machine (NOMINAL -> DEGRADED ->
                SAFE_STOP -> FAULT) + finite/saturation/slew validation
can_bus.py      TrackCmdCodec (E2E: CRC8 + rolling counter), CmdReceiver,
                MpcStatusCodec
simulator.py    ClosedLoopSimulator + SimLog (metric helpers) + FaultInjector
plotting.py     MpcPlotter — all six figure types from a SimLog
```

Dependency direction is a DAG (config is the leaf; simulator/plotting at the
top). The simulator and supervisor depend only on the abstract `Controller`
interface, so the SLSQP and IPOPT backends are interchangeable, and any
`ReferencePath` subclass drops in without touching the loop.

## Usage

```python
from skidsteer_mpc import *

model = SkidSteerModel()
ref   = SCurveLaneChange(v_target=1.2)        # or ConstantRadiusCircle(...), etc.
obs   = Obstacle()
ctrl  = ScipyMPC(model, MPCConfig(), obs)     # CasadiMPC(...) on the target
gov   = CommandGovernor(SupervisorConfig(), v0=1.2)

log = ClosedLoopSimulator(model, ctrl, gov, ref, obs, MPCConfig()).run()
print(log.rms_cm(), log.min_clearance())

MpcPlotter(log, outdir="out").path_tracking()
```

Run the full demo (two sims + all six plots):

```
python run_demo.py
```

## Backends

- **ScipyMPC** (single shooting, SLSQP) runs anywhere; used to generate the
  in-sandbox plots.
- **CasadiMPC** (multiple shooting, IPOPT with `max_wall_time`) is the deploy
  target; `pip install casadi`. Both share the same model, cost, and limits.

## Validation

`run_demo.py` reproduces the original procedural results exactly: nominal run
RMS 8.3 cm / peak 40.6 cm / 19 saturated samples / 1.100 m clearance; supervised
run DEGRADED=10 / SAFE_STOP=40 / clearance maintained.

## Animation / replay GIFs

You can turn any `SimLog` into a replay GIF:

```python
from skidsteer_mpc import MpcAnimator

log = ClosedLoopSimulator(model, ctrl, gov, ref, obs, MPCConfig()).run()

MpcAnimator(log, outdir="out").save_gif(
    filename="nominal_replay.gif",
    fps=12,
    every=2,
    dpi=120,
)
```

`every` skips frames to keep the file size reasonable. For example, `every=2`
uses every other simulation sample.

## Additional validation scenarios

This project can also run extra scenarios:

1. A nominal S-curve lane change.
2. A sinusoidal curved-track path.
3. A sinusoidal path with a proximity-based safety envelope.

The proximity safety layer reduces longitudinal velocity as the vehicle gets
closer to the obstacle and latches a stop when the vehicle is within the stop
threshold.

```python
from skidsteer_mpc.safety import ProximitySafetyConfig, ProximitySpeedLimiter

limiter = ProximitySpeedLimiter(
    obs,
    ProximitySafetyConfig(
        stop_distance=1.0,
        slow_distance=4.0,
        latch_stop=True,
    ),
)

sim = ClosedLoopSimulator(
    model, ctrl, gov, ref, obs, mpc_cfg, SimConfig(),
    safety_filter=limiter,
)
```

Run:

```bash
python run_additional_scenarios.py
```
## Dynamic obstacle scenarios

Two dynamic obstacle scenarios are included:

1. `dynamic_obstacle_early_avoid`
   - The obstacle moves into the path well ahead of the vehicle.
   - The MPC has time to plan around it.

2. `dynamic_obstacle_late_stop`
   - The obstacle moves into the path late.
   - The proximity speed limiter slows longitudinal velocity as distance decreases.
   - The vehicle latches a stop at 1.0 m from the obstacle center.

Run:

```bash
python run_dynamic_obstacle_scenarios.py
```

This writes static plots and replay GIFs to `out/`.
