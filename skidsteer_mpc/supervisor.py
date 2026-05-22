"""Safety supervisor / command governor. Consumes a SolveResult each cycle and
guarantees a valid, on-time command via graceful degradation:
NOMINAL -> DEGRADED (buffered plan) -> SAFE_STOP (decel ramp) -> FAULT (latched).
Every output passes finite-check -> saturation -> slew-rate limit."""
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import math

from .config import SupervisorConfig
from .controller import SolveResult, SolveStatus


class CtrlMode(IntEnum):
    DISABLE = 0
    NOMINAL = 1
    DEGRADED = 2
    SAFE_STOP = 3
    FAULT = 4


class FaultCode(IntEnum):
    NONE = 0
    SOLVE_FAIL = 1
    TIMEOUT = 2
    NONFINITE = 3
    BUFFER_EXHAUSTED = 4
    ESTOP = 5


@dataclass
class CommandOutput:
    v_left: float
    v_right: float
    mode: CtrlMode
    fault: FaultCode
    buffered: bool
    clamped: bool
    solve_time_s: float


def _finite(x) -> bool:
    return x is not None and math.isfinite(float(x))


class CommandGovernor:
    def __init__(self, cfg: SupervisorConfig | None = None, v0: float = 0.0):
        self.cfg = cfg or SupervisorConfig()
        self.mode = CtrlMode.NOMINAL
        self.fault = FaultCode.NONE
        self.prev_cmd = (v0, v0)
        self.last_plan = None
        self.buf_offset = 0
        self.consec_fail = 0
        self.good_streak = 0
        self.estop = False

    def reset(self):
        self.mode, self.fault = CtrlMode.NOMINAL, FaultCode.NONE
        self.buf_offset = self.consec_fail = self.good_streak = 0

    def set_estop(self, engaged: bool):
        self.estop = engaged

    def step(self, result: SolveResult) -> CommandOutput:
        cfg = self.cfg
        buffered = False

        if self.estop:
            self.mode, self.fault = CtrlMode.FAULT, FaultCode.ESTOP
            return self._emit(0.0, 0.0, buffered, result.solve_time_s)
        if self.mode == CtrlMode.FAULT:
            return self._emit(0.0, 0.0, buffered, result.solve_time_s)

        usable = (result.status == SolveStatus.OK
                  and result.solve_time_s <= cfg.solve_budget_s
                  and self._plan_ok(result.plan))
        overrun = (result.status == SolveStatus.OK
                   and result.solve_time_s > cfg.solve_budget_s)

        if usable and self.mode in (CtrlMode.NOMINAL, CtrlMode.DEGRADED):
            self.last_plan = list(result.plan)
            self.buf_offset = 1
            self.consec_fail = 0
            self.good_streak += 1
            if self.mode == CtrlMode.DEGRADED and self.good_streak >= cfg.recover_count:
                self.mode, self.fault = CtrlMode.NOMINAL, FaultCode.NONE
            vl, vr = self.last_plan[0], self.last_plan[1]

        elif self.mode == CtrlMode.SAFE_STOP:
            vl, vr = self._safe_stop()

        else:
            self.good_streak = 0
            self.consec_fail += 1
            nonfinite_plan = (result.status == SolveStatus.OK
                              and result.plan is not None
                              and not self._plan_ok(result.plan))
            if nonfinite_plan:
                self.fault = FaultCode.NONFINITE
            elif overrun or result.status == SolveStatus.TIMEOUT:
                self.fault = FaultCode.TIMEOUT
            else:
                self.fault = FaultCode.SOLVE_FAIL

            if nonfinite_plan:
                self.mode = CtrlMode.SAFE_STOP
                vl, vr = self._safe_stop()
            elif (self.last_plan is not None
                    and self.buf_offset < cfg.max_buffered
                    and 2 * self.buf_offset + 1 < len(self.last_plan)):
                self.mode = CtrlMode.DEGRADED
                vl = self.last_plan[2 * self.buf_offset]
                vr = self.last_plan[2 * self.buf_offset + 1]
                self.buf_offset += 1
                buffered = True
            else:
                self.mode = CtrlMode.SAFE_STOP
                self.fault = FaultCode.BUFFER_EXHAUSTED
                vl, vr = self._safe_stop()

        return self._emit(vl, vr, buffered, result.solve_time_s)

    def _plan_ok(self, plan) -> bool:
        return plan is not None and len(plan) >= 2 and _finite(plan[0]) and _finite(plan[1])

    def _safe_stop(self):
        v_prev = 0.5 * (self.prev_cmd[0] + self.prev_cmd[1])
        v_next = max(0.0, v_prev - self.cfg.a_stop * self.cfg.dt)
        return v_next, v_next

    def _emit(self, vl, vr, buffered, solve_time_s) -> CommandOutput:
        cfg = self.cfg
        clamped = False
        if not (_finite(vl) and _finite(vr)):
            self.mode, self.fault = CtrlMode.FAULT, FaultCode.NONFINITE
            vl = vr = 0.0
        cl = min(max(vl, cfg.v_min), cfg.v_max)
        cr = min(max(vr, cfg.v_min), cfg.v_max)
        clamped |= (cl != vl) or (cr != vr)
        pl, pr = self.prev_cmd
        rl = min(max(cl, pl - cfg.dv_max), pl + cfg.dv_max)
        rr = min(max(cr, pr - cfg.dv_max), pr + cfg.dv_max)
        clamped |= (rl != cl) or (rr != cr)
        self.prev_cmd = (rl, rr)
        if self.mode == CtrlMode.SAFE_STOP and 0.5 * (rl + rr) <= cfg.stop_eps:
            self.mode = CtrlMode.FAULT
        return CommandOutput(rl, rr, self.mode, self.fault, buffered, clamped, solve_time_s)
