"""All figures in one place, driven by a SimLog. Replaces the styling/plot code
that was duplicated across the old scripts."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D

from .config import PlotStyle
from .simulator import SimLog
from .supervisor import CtrlMode


class MpcPlotter:
    def __init__(self, log: SimLog, style: PlotStyle | None = None, outdir: str = "."):
        self.log = log
        self.s = style or PlotStyle()
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        plt.rcParams.update({
            "figure.facecolor": "white", "axes.facecolor": "white",
            "savefig.facecolor": "white", "font.family": "DejaVu Sans",
            "font.size": 11, "axes.edgecolor": "#C4C8CE", "axes.linewidth": 1.0,
            "axes.labelcolor": self.s.ink, "text.color": self.s.ink,
            "xtick.color": self.s.ink, "ytick.color": self.s.ink,
            "axes.grid": True, "grid.color": self.s.grid, "grid.linewidth": 1.0})

    # -- helpers ------------------------------------------------------------
    def _foot(self, fig):
        fig.text(0.012, 0.012, self.s.footnote, fontsize=8.5, style="italic", color="#9AA0A6")

    def _save(self, fig, name):
        path = os.path.join(self.outdir, name)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(path, dpi=self.s.dpi)
        plt.close(fig)
        return path

    def _obstacle(self, ax):
        o = self.log.obstacle.spec
        ax.add_patch(Circle((o.x, o.y), o.radius, facecolor=self.s.coral,
                            edgecolor="#B5503A", lw=1.2, alpha=0.92, zorder=5))
        ax.add_patch(Circle((o.x, o.y), o.r_safe, facecolor="none", edgecolor=self.s.coral,
                            lw=1.4, ls=(0, (5, 4)), alpha=0.85, zorder=5))

    # -- 1. path tracking ---------------------------------------------------
    def path_tracking(self, name="plot1_path_tracking.png"):
        s, L = self.s, self.log
        x, y = L.z[:, 0], L.z[:, 1]
        fig, ax = plt.subplots(figsize=(8.4, 4.4))

        def draw(a):
            a.plot(L.ref[:, 0], L.ref[:, 1], color=s.gray, lw=2.2, ls="--", zorder=3)
            a.plot(x, y, color=s.teal, lw=2.8, zorder=4)
            self._obstacle(a)
        ax.plot([], [], color=s.gray, lw=2.2, ls="--", label="Reference path")
        ax.plot([], [], color=s.teal, lw=2.8, label="MPC-followed trajectory")
        draw(ax)
        o = L.obstacle.spec
        ax.annotate("Obstacle", (o.x, o.y), (o.x - 3.0, o.y + 1.7), color="#B5503A",
                    fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#B5503A", lw=1))
        ax.scatter([x[0]], [y[0]], s=55, color=s.teal, zorder=6, edgecolor="white", lw=1.2)
        ax.text(x[0], y[0] - 0.55, "start", color=s.teal, fontsize=9, ha="center")
        ax.scatter([x[-1]], [y[-1]], s=55, marker="s", color=s.teal, zorder=6,
                   edgecolor="white", lw=1.2)
        ax.set_xlabel("x  [m]"); ax.set_ylabel("y  [m]")
        ax.set_title("Path Tracking — Reference vs. NLP-MPC Trajectory",
                     fontsize=13, fontweight="bold", pad=10)
        ax.set_aspect("equal", adjustable="box"); ax.set_xlim(-1, 29); ax.set_ylim(-1.4, 4.2)
        ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#D7DBDF")
        axin = ax.inset_axes([0.05, 0.50, 0.34, 0.46]); draw(axin)
        axin.set_xlim(o.x - 1.9, o.x + 2.1); axin.set_ylim(0.45, 2.75)
        axin.set_aspect("equal", adjustable="box")
        axin.set_xticklabels([]); axin.set_yticklabels([]); axin.tick_params(length=0); axin.grid(False)
        for sp in axin.spines.values(): sp.set_edgecolor("#B9BEC4")
        axin.set_title("avoidance detail", fontsize=8.5, color="#6B7178", pad=3)
        ax.indicate_inset_zoom(axin, edgecolor="#B9BEC4", alpha=0.7)
        self._foot(fig)
        return self._save(fig, name)

    # -- 2. control inputs --------------------------------------------------
    def control_inputs(self, v_max, v_min, name="plot2_control_inputs.png"):
        s, L = self.s, self.log
        t, VL, VR = L.t, L.u[:, 0], L.u[:, 1]
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        ax.axhline(v_max, color=s.red, lw=1.3, ls="--"); ax.axhline(v_min, color=s.red, lw=1.3, ls="--")
        ax.text(0.3, v_max + 0.02, "track velocity limit  V$_{max}$", color=s.red, fontsize=8.8, va="bottom")
        ax.text(0.3, v_min + 0.02, "V$_{min}$", color=s.red, fontsize=8.8, va="bottom")
        sat = np.isclose(VR, v_max, atol=1e-3) | np.isclose(VL, v_max, atol=1e-3)
        ax.fill_between(t, 0, 1, where=sat, transform=ax.get_xaxis_transform(),
                        color=s.red, alpha=0.08, label="saturation interval")
        ax.plot(t, VL, color=s.teal, lw=2.4, label="V$_L$ (left track)")
        ax.plot(t, VR, color=s.teal_dk, lw=2.4, label="V$_R$ (right track)")
        ax.set_xlabel("time  [s]"); ax.set_ylabel("track velocity  [m/s]")
        ax.set_title("Control Inputs — Track Velocity Commands", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(v_min - 0.12, v_max + 0.18)
        ax.legend(loc="lower left", frameon=True, framealpha=0.95, edgecolor="#D7DBDF")
        self._foot(fig)
        return self._save(fig, name)

    # -- 3. tracking error --------------------------------------------------
    def tracking_error(self, name="plot3_tracking_error.png"):
        s, L = self.s, self.log
        t = L.t; err = L.tracking_error() * 100; rms = L.rms_cm(); ipk = int(np.argmax(err))
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        ax.plot(t, err, color=s.teal, lw=2.6, label="position error  $\\|p-p_{ref}\\|$")
        ax.axhline(rms, color=s.teal_dk, lw=1.4, ls="--", label=f"RMS = {rms:.1f} cm")
        ax.scatter([t[ipk]], [err[ipk]], s=45, color=s.coral, zorder=5, edgecolor="white", lw=1.2)
        ax.annotate(f"obstacle-avoidance\nexcursion  ({err[ipk]:.0f} cm)", (t[ipk], err[ipk]),
                    (t[ipk] + 1.3, err[ipk] - 3), color="#B5503A", fontsize=9.5,
                    arrowprops=dict(arrowstyle="->", color="#B5503A", lw=1.1))
        mask = (t < t[ipk] - 2.2) | (t > t[ipk] + 2.2)
        nom = float(np.sqrt(np.mean((err[mask]) ** 2)))
        ax.text(0.985, 0.40, f"off-maneuver tracking ≈ {nom:.1f} cm", transform=ax.transAxes,
                ha="right", fontsize=9, color="#6B7178")
        ax.set_xlabel("time  [s]"); ax.set_ylabel("tracking error  [cm]")
        ax.set_title("Tracking Error vs. Time", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, max(err) * 1.18)
        ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#D7DBDF")
        self._foot(fig)
        return self._save(fig, name)

    # -- 4. attitude --------------------------------------------------------
    def attitude(self, name="plot4_attitude.png"):
        s, L = self.s, self.log
        t, z = L.t, L.z
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        l1, = ax.plot(t, np.degrees(z[:, 4]), color=s.teal, lw=2.3, label="pitch  [deg]")
        l2, = ax.plot(t, np.degrees(z[:, 5]), color=s.coral, lw=2.3, label="roll  [deg]")
        ax.axhline(0, color="#C4C8CE", lw=1.0); ax.set_ylim(-7, 7)
        ax.set_xlabel("time  [s]"); ax.set_ylabel("body attitude  [deg]")
        ax2 = ax.twinx()
        l3, = ax2.plot(t, np.degrees(z[:, 6]), color=s.teal_dk, lw=2.3, label="yaw rate  [deg/s]")
        ax2.set_ylabel("yaw rate  [deg/s]", color=s.teal_dk); ax2.tick_params(colors=s.teal_dk)
        ax2.grid(False); ax2.set_ylim(-70, 70)
        ax.set_title("Body Attitude & Yaw-Rate States  (supplementary)", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlim(t[0], t[-1])
        ax.legend(handles=[l1, l2, l3], loc="upper right", frameon=True,
                  framealpha=0.95, edgecolor="#D7DBDF", ncol=3)
        self._foot(fig)
        return self._save(fig, name)

    # -- 5. supervised trajectory (mode-colored) ----------------------------
    def supervised_trajectory(self, name="plot5_supervised_trajectory.png"):
        s, L = self.s, self.log
        x, y, mode = L.z[:, 0], L.z[:, 1], L.mode
        cmap = {int(CtrlMode.NOMINAL): s.teal, int(CtrlMode.DEGRADED): s.amber,
                int(CtrlMode.SAFE_STOP): s.red, int(CtrlMode.FAULT): s.red,
                int(CtrlMode.DISABLE): s.gray}
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        ax.plot(L.ref[:, 0], L.ref[:, 1], color=s.gray, lw=2.0, ls="--", zorder=2)
        i = 0
        while i < len(x) - 1:
            j = i
            while j < len(x) - 1 and mode[j + 1] == mode[i]:
                j += 1
            ax.plot(x[i:j + 2], y[i:j + 2], color=cmap.get(mode[i], s.gray), lw=2.8,
                    solid_capstyle="round", zorder=4)
            i = j + 1
        self._obstacle(ax)
        if np.any(mode >= int(CtrlMode.SAFE_STOP)):
            ax.scatter([x[-1]], [y[-1]], s=70, marker="X", color=s.red, zorder=7, edgecolor="white", lw=1.3)
            ax.annotate("safe-stop\n(latched)", (x[-1], y[-1]), (x[-1] - 0.5, y[-1] - 1.7),
                        color=s.red, fontsize=9, ha="center",
                        arrowprops=dict(arrowstyle="->", color=s.red, lw=1.1))
        ax.scatter([x[0]], [y[0]], s=50, color=s.teal, zorder=6, edgecolor="white", lw=1.2)
        ax.text(x[0], y[0] - 0.55, "start", color=s.teal, fontsize=9, ha="center")
        handles = [Line2D([], [], color=s.gray, ls="--", lw=2, label="Reference"),
                   Line2D([], [], color=s.teal, lw=3, label="NOMINAL"),
                   Line2D([], [], color=s.amber, lw=3, label="DEGRADED (buffered plan)"),
                   Line2D([], [], color=s.red, lw=3, label="SAFE-STOP / FAULT")]
        ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.95,
                  edgecolor="#D7DBDF", fontsize=9)
        ax.set_xlabel("x  [m]"); ax.set_ylabel("y  [m]"); ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1, 29); ax.set_ylim(-1.4, 4.2)
        ax.set_title("Supervised Trajectory — Graceful Degradation Under Solver Faults",
                     fontsize=12.5, fontweight="bold", pad=10)
        self._foot(fig)
        return self._save(fig, name)

    # -- 6. supervisor timeline ---------------------------------------------
    def supervisor_timeline(self, v_max, v_min, fault_windows=None,
                            name="plot6_supervisor_timeline.png"):
        s, L = self.s, self.log
        t, VL, VR, v, mode = L.t, L.u[:, 0], L.u[:, 1], L.z[:, 3], L.mode
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.4, 6.0), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1.3]})

        def shade(ax):
            for st, c in [(int(CtrlMode.DEGRADED), s.amber), (int(CtrlMode.SAFE_STOP), s.red),
                          (int(CtrlMode.FAULT), s.red)]:
                ax.fill_between(t, 0, 1, where=(mode == st), transform=ax.get_xaxis_transform(),
                                color=c, alpha=0.12, zorder=0, step="mid")
        shade(a1)
        a1.axhline(v_max, color=s.red, lw=1.2, ls="--"); a1.axhline(v_min, color=s.red, lw=1.2, ls="--")
        a1.text(0.2, v_max + 0.02, "V$_{max}$", color=s.red, fontsize=8.5, va="bottom")
        a1.plot(t, VL, color=s.teal, lw=2.3, label="V$_L$ (left)")
        a1.plot(t, VR, color=s.teal_dk, lw=2.3, label="V$_R$ (right)")
        for w, lab in (fault_windows or []):
            t0, t1 = w[0] * L.t[1], (w[-1] + 1) * L.t[1]
            a1.annotate(lab, ((t0 + t1) / 2, v_max + 0.16), ha="center", fontsize=8.5, color="#6B7178")
            a1.plot([t0, t1], [v_max + 0.10] * 2, color="#6B7178", lw=1.2)
        a1.set_ylabel("track velocity  [m/s]"); a1.set_ylim(v_min - 0.12, v_max + 0.30)
        a1.legend(loc="center right", frameon=True, framealpha=0.95, edgecolor="#D7DBDF", ncol=2, fontsize=9)
        a1.set_title("Supervisor Command Output Under Injected Solver Faults",
                     fontsize=12.5, fontweight="bold", pad=10)
        a1.add_artist(a1.legend(handles=[Patch(color=s.amber, alpha=0.3, label="DEGRADED"),
                      Patch(color=s.red, alpha=0.3, label="SAFE-STOP / FAULT")],
                      loc="lower right", frameon=True, framealpha=0.95, edgecolor="#D7DBDF", fontsize=8.5))
        shade(a2)
        a2.plot(t, v, color=s.teal, lw=2.4, label="forward speed v")
        a2.set_ylabel("v  [m/s]"); a2.set_xlabel("time  [s]"); a2.set_ylim(-0.1, 1.5)
        a2.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#D7DBDF", fontsize=9)
        self._foot(fig)
        return self._save(fig, name)
