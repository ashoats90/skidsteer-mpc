"""MPC controllers behind one interface. The simulator and supervisor depend only
on Controller.solve(...) -> SolveResult, so the SLSQP and IPOPT backends are
interchangeable. SolveResult / SolveStatus live here because the controller
produces them; the supervisor imports them."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
import time
import numpy as np

from .config import MPCConfig
from .model import SkidSteerModel
from .obstacle import Obstacle


class SolveStatus(IntEnum):
    OK = 0
    FAIL = 1        # infeasible / exception / no acceptable point
    TIMEOUT = 2     # exceeded wall-clock budget


@dataclass
class SolveResult:
    status: SolveStatus
    plan: object = None             # flat [VL0,VR0,VL1,VR1,...] or None
    solve_time_s: float = 0.0


class Controller(ABC):
    @abstractmethod
    def solve(self, state: np.ndarray, ref_preview: np.ndarray,
              u_prev: np.ndarray) -> SolveResult:
        """ref_preview is (horizon+1, 3) = [x_ref, y_ref, yaw_ref]."""


# ---------------------------------------------------------------------------
# scipy SLSQP, single shooting (runs anywhere; used for the in-sandbox plots)
# ---------------------------------------------------------------------------
class ScipyMPC(Controller):
    def __init__(self, model: SkidSteerModel, cfg: MPCConfig, obstacle: Obstacle):
        self.model, self.cfg, self.obs = model, cfg, obstacle
        self._U = np.tile([cfg.v_target, cfg.v_target], cfg.horizon).astype(float)

    @staticmethod
    def _wrap(a):
        return np.arctan2(np.sin(a), np.cos(a))

    def _rollout(self, z0, U):
        N = self.cfg.horizon
        Z = np.empty((N + 1, self.model.NX))
        Z[0] = z0
        for k in range(N):
            Z[k + 1] = self.model.rk4_np(Z[k], U[2 * k:2 * k + 2], self.cfg.dt)
        return Z

    def _cost(self, U, z0, ref, u_prev):
        c, o = self.cfg, self.obs
        Z = self._rollout(z0, U)
        J = 0.0
        for k in range(1, c.horizon + 1):
            xr, yr, yawr = ref[k]
            w = c.term_mult if k == c.horizon else 1.0
            J += w * (c.q_x * (Z[k, 0] - xr) ** 2 + c.q_y * (Z[k, 1] - yr) ** 2)
            J += c.q_psi * self._wrap(Z[k, 2] - yawr) ** 2
            J += c.w_time * (c.v_target - Z[k, 3]) ** 2
            slack = o.r_safe - np.sqrt(o.dist2(Z[k, 0], Z[k, 1]) + 1e-9)
            if slack > 0:
                J += c.w_obs * slack ** 2
        prev = u_prev
        for k in range(c.horizon):
            uk = U[2 * k:2 * k + 2]
            J += c.r_rate * np.sum((uk - prev) ** 2)
            prev = uk
        return J

    def _obstacle_ineq(self, U, z0):
        c, o = self.cfg, self.obs
        Z = self._rollout(z0, U)
        return np.array([o.dist2(Z[k, 0], Z[k, 1]) - o.r_safe ** 2
                         for k in range(1, c.horizon + 1)])

    def solve(self, state, ref_preview, u_prev):
        from scipy.optimize import minimize
        c = self.cfg
        bounds = [(c.v_min, c.v_max)] * (2 * c.horizon)
        cons = [{"type": "ineq", "fun": lambda U: self._obstacle_ineq(U, state)}]
        t0 = time.perf_counter()
        res = minimize(lambda U: self._cost(U, state, ref_preview, u_prev),
                       self._U, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 80, "ftol": 1e-4})
        dt_solve = time.perf_counter() - t0
        U = res.x
        self._U = np.concatenate([U[2:], U[-2:]])          # warm-start shift
        status = SolveStatus.OK if res.success or res.status == 9 else SolveStatus.FAIL
        return SolveResult(status, plan=U.tolist(), solve_time_s=dt_solve)


# ---------------------------------------------------------------------------
# CasADi multiple shooting + IPOPT (deploy target; needs `casadi`)
# ---------------------------------------------------------------------------
class CasadiMPC(Controller):
    def __init__(self, model: SkidSteerModel, cfg: MPCConfig, obstacle: Obstacle,
                 solve_budget_s: float = 0.10):
        import casadi as ca
        self.ca = ca
        self.model, self.cfg, self.obs = model, cfg, obstacle
        N, nx, nu = cfg.horizon, model.NX, model.NU
        F = model.casadi_integrator(ca, cfg.dt)

        opti = ca.Opti()
        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)
        x0p = opti.parameter(nx)
        refp = opti.parameter(3, N + 1)
        prevp = opti.parameter(nu)

        J = 0
        opti.subject_to(X[:, 0] == x0p)
        for k in range(N):
            opti.subject_to(X[:, k + 1] == F(X[:, k], U[:, k]))
        for k in range(1, N + 1):
            w = cfg.term_mult if k == N else 1.0
            ex, ey = X[0, k] - refp[0, k], X[1, k] - refp[1, k]
            eyaw = ca.atan2(ca.sin(X[2, k] - refp[2, k]), ca.cos(X[2, k] - refp[2, k]))
            J += w * (cfg.q_x * ex ** 2 + cfg.q_y * ey ** 2) + cfg.q_psi * eyaw ** 2
            J += cfg.w_time * (cfg.v_target - X[3, k]) ** 2
            d2 = self.obs.dist2(X[0, k], X[1, k])
            J += cfg.w_obs * ca.fmax(0.0, self.obs.r_safe - ca.sqrt(d2 + 1e-9)) ** 2
            opti.subject_to(d2 >= self.obs.r_safe ** 2)
        puse = prevp
        for k in range(N):
            J += cfg.r_rate * ca.sumsqr(U[:, k] - puse)
            puse = U[:, k]
        opti.minimize(J)
        opti.subject_to(opti.bounded(cfg.v_min, ca.vec(U), cfg.v_max))
        opti.solver("ipopt", {"expand": True},
                    {"print_level": 0, "sb": "yes", "max_iter": 300,
                     "max_wall_time": solve_budget_s, "acceptable_tol": 1e-6})

        self.opti, self.X, self.U = opti, X, U
        self.x0p, self.refp, self.prevp = x0p, refp, prevp
        self._X_ws = None
        self._U_ws = np.zeros((nu, N))

    def solve(self, state, ref_preview, u_prev):
        ca, N = self.ca, self.cfg.horizon
        if self._X_ws is None:
            self._X_ws = np.tile(np.asarray(state).reshape(-1, 1), (1, N + 1))
        self.opti.set_value(self.x0p, state)
        self.opti.set_value(self.refp, ref_preview.T)
        self.opti.set_value(self.prevp, u_prev)
        self.opti.set_initial(self.X, self._X_ws)
        self.opti.set_initial(self.U, self._U_ws)

        t0 = time.perf_counter()
        try:
            sol = self.opti.solve()
            U_opt = np.atleast_2d(sol.value(self.U))
            X_opt = sol.value(self.X)
            status = SolveStatus.OK
        except RuntimeError:
            ret = self.opti.stats().get("return_status", "")
            if ret in ("Maximum_WallTime_Exceeded", "Maximum_Iterations_Exceeded",
                       "Solved_To_Acceptable_Level"):
                U_opt = np.atleast_2d(self.opti.debug.value(self.U))
                X_opt = self.opti.debug.value(self.X)
                status = SolveStatus.TIMEOUT
            else:
                return SolveResult(SolveStatus.FAIL, None, time.perf_counter() - t0)
        dt_solve = time.perf_counter() - t0
        self._X_ws = np.hstack([X_opt[:, 1:], X_opt[:, -1:]])
        self._U_ws = np.hstack([U_opt[:, 1:], U_opt[:, -1:]])
        return SolveResult(status, plan=U_opt.T.reshape(-1).tolist(),
                           solve_time_s=dt_solve)
