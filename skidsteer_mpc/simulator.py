"""Closed-loop orchestration. The simulator wires model + controller + supervisor
+ reference + obstacle and produces a SimLog. It depends only on the abstract
interfaces, so any Controller backend or ReferencePath drops in unchanged."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .config import MPCConfig, SimConfig
from .model import SkidSteerModel
from .controller import Controller, SolveResult, SolveStatus
from .supervisor import CommandGovernor, CtrlMode
from .reference import ReferencePath
from .obstacle import Obstacle


@dataclass
class SimLog:
    t: np.ndarray
    z: np.ndarray
    u: np.ndarray
    ref: np.ndarray
    mode: np.ndarray
    fault: np.ndarray
    buffered: np.ndarray
    solve_ms: np.ndarray
    obstacle: Obstacle
    reference: ReferencePath

    def tracking_error(self) -> np.ndarray:
        return np.hypot(self.z[:, 0] - self.ref[:, 0], self.z[:, 1] - self.ref[:, 1])

    def rms_cm(self) -> float:
        return float(np.sqrt(np.mean(self.tracking_error() ** 2)) * 100)

    def peak_cm(self) -> float:
        return float(np.max(self.tracking_error()) * 100)

    def min_clearance(self) -> float:
        return float(np.min(self.obstacle.distance(self.z[:, 0], self.z[:, 1])))

    def saturated_samples(self, v_max: float) -> int:
        return int(np.sum(np.isclose(self.u, v_max, atol=1e-3)))


class FaultInjector:
    """Forces solver outcomes on specific steps to exercise the supervisor."""
    def __init__(self, timeout_steps=None, fail_steps=None):
        self.timeout_steps = set(timeout_steps or [])
        self.fail_steps = set(fail_steps or [])

    def forced(self, step: int):
        if step in self.timeout_steps:
            return SolveResult(SolveStatus.TIMEOUT, None, 0.20)
        if step in self.fail_steps:
            return SolveResult(SolveStatus.FAIL, None, 0.0)
        return None


class ClosedLoopSimulator:
    def __init__(self, model: SkidSteerModel, controller: Controller,
                 supervisor: CommandGovernor, reference: ReferencePath,
                 obstacle: Obstacle, mpc_cfg: MPCConfig,
                 sim_cfg: SimConfig | None = None,
                 fault_injector: FaultInjector | None = None):
        self.model, self.controller, self.sup = model, controller, supervisor
        self.ref, self.obs = reference, obstacle
        self.mpc, self.sim = mpc_cfg, sim_cfg or SimConfig()
        self.faults = fault_injector

    def run(self) -> SimLog:
        m, c = self.model, self.mpc
        steps = int(self.sim.t_sim / c.dt)
        z = np.array([0.0, 0.0, 0.0, c.v_target, 0.0, 0.0, 0.0])
        u_prev = np.array([c.v_target, c.v_target])

        cols = {k: [] for k in ("t", "z", "u", "ref", "mode", "fault", "buffered", "solve_ms")}
        for i in range(steps):
            ref_preview = self.ref.preview(i * c.dt, c.horizon, c.dt)

            forced = self.faults.forced(i) if self.faults else None
            res = forced if forced is not None else \
                self.controller.solve(z, ref_preview, u_prev)
            if res.status == SolveStatus.OK and self.sim.nominal_solve_ms is not None:
                res.solve_time_s = self.sim.nominal_solve_ms / 1000.0   # represented IPOPT time

            out = self.sup.step(res)
            u = np.array([out.v_left, out.v_right])

            cols["t"].append(i * c.dt); cols["z"].append(z.copy()); cols["u"].append(u)
            cols["ref"].append(ref_preview[0]); cols["mode"].append(int(out.mode))
            cols["fault"].append(int(out.fault)); cols["buffered"].append(out.buffered)
            cols["solve_ms"].append(res.solve_time_s * 1000)

            z = m.rk4_np(z, u, c.dt)
            u_prev = u

        return SimLog(**{k: np.array(v) for k, v in cols.items()},
                      obstacle=self.obs, reference=self.ref)
