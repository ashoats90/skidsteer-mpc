"""Centralized configuration. All tuning lives here as dataclasses instead of
module-level globals, so an experiment is one config object, not edits in five files."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ModelParams:
    track_width: float = 0.60      # m
    tau_v: float = 0.30            # track-speed first-order lag [s]
    tau_r: float = 0.25            # yaw-rate lag [s]
    tau_pitch: float = 0.40        # pitch settling [s]
    tau_roll: float = 0.40         # roll settling [s]
    g: float = 9.81


@dataclass
class MPCConfig:
    horizon: int = 30
    dt: float = 1.0 / 6.0          # ~167 ms
    v_target: float = 1.20
    v_max: float = 1.40
    v_min: float = 0.00
    q_x: float = 8.0               # position tracking
    q_y: float = 8.0
    q_psi: float = 0.5             # heading tracking
    w_time: float = 1.5            # time / progress penalty
    w_obs: float = 60.0            # obstacle soft weight
    r_rate: float = 0.20           # input-rate regularization
    term_mult: float = 3.0         # terminal-stage multiplier

    @property
    def lookahead_s(self) -> float:
        return self.horizon * self.dt


@dataclass
class ObstacleSpec:
    x: float = 12.5
    y: float = 2.0
    radius: float = 0.50           # physical [m]
    clearance: float = 0.60        # required margin [m]

    @property
    def r_safe(self) -> float:
        return self.radius + self.clearance


@dataclass
class SupervisorConfig:
    dt: float = 1.0 / 6.0
    v_min: float = 0.0
    v_max: float = 1.40
    dv_max: float = 0.40           # slew limit per cycle per track [m/s]
    solve_budget_s: float = 0.10   # production IPOPT max_wall_time
    max_buffered: int = 5          # buffered-plan steps before SAFE_STOP
    recover_count: int = 3         # good solves to clear DEGRADED
    a_stop: float = 1.2            # safe-stop decel [m/s^2]
    stop_eps: float = 0.03


@dataclass
class CanConfig:
    bitrate: int = 500_000
    track_cmd_id: int = 0x120
    status_id: int = 0x121
    data_id: int = 0x1A            # E2E Data-ID folded into CRC
    v_scale: float = 0.001         # m/s per bit
    tx_period_s: float = 0.020     # 50 Hz
    rx_timeout_s: float = 0.060    # drive-side freshness watchdog


@dataclass
class SimConfig:
    t_sim: float = 24.0
    nominal_solve_ms: float | None = 50.0   # represented IPOPT solve time fed to
                                             # the supervisor (the in-sandbox SLSQP
                                             # wall time is not representative)


@dataclass
class PlotStyle:
    teal: str = "#0E8C82"
    teal_dk: str = "#13414E"
    gray: str = "#8A8F98"
    coral: str = "#E07A5F"
    amber: str = "#E0A458"
    red: str = "#C9544D"
    ink: str = "#23272E"
    grid: str = "#E7E9EC"
    footnote: str = "Illustrative simulation — not hardware data"
    dpi: int = 200
