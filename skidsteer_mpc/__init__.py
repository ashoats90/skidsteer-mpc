"""skidsteer_mpc — object-oriented skid-steer NLP-MPC with a safety supervisor
and CAN output layer.

Typical use:
    from skidsteer_mpc import *
    model = SkidSteerModel()
    ref   = SCurveLaneChange(v_target=1.2)
    obs   = Obstacle()
    ctrl  = ScipyMPC(model, MPCConfig(), obs)          # or CasadiMPC(...) on target
    gov   = CommandGovernor(SupervisorConfig(), v0=1.2)
    log   = ClosedLoopSimulator(model, ctrl, gov, ref, obs, MPCConfig()).run()
    MpcPlotter(log).path_tracking()
"""
from .config import (ModelParams, MPCConfig, SupervisorConfig, ObstacleSpec,
                     CanConfig, SimConfig, PlotStyle)
from .model import SkidSteerModel
from .reference import (ReferencePath, SCurveLaneChange, StraightWithSwerve,
                        ConstantRadiusCircle)
from .obstacle import Obstacle
from .controller import Controller, ScipyMPC, CasadiMPC, SolveResult, SolveStatus
from .supervisor import CommandGovernor, CtrlMode, FaultCode, CommandOutput
from .can_bus import (TrackCmdCodec, CmdReceiver, MpcStatusCodec, crc8_autosar,
                      FLAG_SOLVER_OK, FLAG_TIMEOUT, FLAG_BUFFERED, FLAG_CLAMPED)
from .simulator import ClosedLoopSimulator, SimLog, FaultInjector
from .plotting import MpcPlotter

__all__ = [
    "ModelParams", "MPCConfig", "SupervisorConfig", "ObstacleSpec", "CanConfig",
    "SimConfig", "PlotStyle", "SkidSteerModel", "ReferencePath", "SCurveLaneChange",
    "StraightWithSwerve", "ConstantRadiusCircle", "Obstacle", "Controller",
    "ScipyMPC", "CasadiMPC", "SolveResult", "SolveStatus", "CommandGovernor",
    "CtrlMode", "FaultCode", "CommandOutput", "TrackCmdCodec", "CmdReceiver",
    "MpcStatusCodec", "crc8_autosar", "FLAG_SOLVER_OK", "FLAG_TIMEOUT",
    "FLAG_BUFFERED", "FLAG_CLAMPED", "ClosedLoopSimulator", "SimLog",
    "FaultInjector", "MpcPlotter",
]
