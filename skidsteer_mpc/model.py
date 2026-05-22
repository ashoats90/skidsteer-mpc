"""The 7-state skid-steer model. Dynamics are written ONCE in a backend-agnostic
form and reused both numerically (numpy, for simulation/SLSQP) and symbolically
(CasADi, for the IPOPT NLP) — so there is no chance of the two drifting apart."""
from __future__ import annotations
import numpy as np
from .config import ModelParams


class SkidSteerModel:
    NX = 7   # [x, y, yaw, v, pitch, roll, yawrate]
    NU = 2   # [V_L, V_R]

    def __init__(self, params: ModelParams | None = None):
        self.p = params or ModelParams()

    # -- backend-agnostic continuous dynamics --------------------------------
    def _f(self, z, u, m):
        """m is the math backend (numpy or casadi); only cos/sin differ."""
        x, y, yaw, v, pitch, roll, r = (z[i] for i in range(self.NX))
        VL, VR = u[0], u[1]
        v_cmd = 0.5 * (VL + VR)
        r_cmd = (VR - VL) / self.p.track_width

        a_lon = (v_cmd - v) / self.p.tau_v          # modeled longitudinal accel
        a_lat = v * r                               # lateral (centripetal) accel
        pitch_ss = -a_lon / self.p.g                # small-angle quasi-static attitude
        roll_ss = a_lat / self.p.g
        return [
            v * m.cos(yaw),
            v * m.sin(yaw),
            r,
            a_lon,
            (pitch_ss - pitch) / self.p.tau_pitch,
            (roll_ss - roll) / self.p.tau_roll,
            (r_cmd - r) / self.p.tau_r,
        ]

    # -- numeric (simulation / SLSQP) ----------------------------------------
    def f_np(self, z, u) -> np.ndarray:
        return np.asarray(self._f(z, u, np), dtype=float)

    def rk4_np(self, z, u, dt) -> np.ndarray:
        k1 = self.f_np(z, u)
        k2 = self.f_np(z + 0.5 * dt * k1, u)
        k3 = self.f_np(z + 0.5 * dt * k2, u)
        k4 = self.f_np(z + dt * k3, u)
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # -- symbolic (CasADi IPOPT) ---------------------------------------------
    def casadi_integrator(self, ca, dt):
        """Return a CasADi Function F(z, u) -> z_next implementing one RK4 step."""
        z = ca.SX.sym("z", self.NX)
        u = ca.SX.sym("u", self.NU)
        f = ca.Function("f", [z, u], [ca.vertcat(*self._f(z, u, ca))])
        k1 = f(z, u)
        k2 = f(z + 0.5 * dt * k1, u)
        k3 = f(z + 0.5 * dt * k2, u)
        k4 = f(z + dt * k3, u)
        return ca.Function("F", [z, u], [z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)])
