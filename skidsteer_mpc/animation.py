"""Animation / replay utilities for skidsteer_mpc.

This module turns a SimLog into replayable trajectory animations and GIFs.
It does not rerun the simulation; it visualizes the time history already stored
in SimLog.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D

from .simulator import SimLog
from .supervisor import CtrlMode


class MpcAnimator:
    """Create replay animations from a SimLog.

    Parameters
    ----------
    log:
        SimLog returned by ClosedLoopSimulator.run().
    outdir:
        Directory where GIFs are written.
    robot_length:
        Visual body length in meters.
    robot_width:
        Visual body width in meters.
    trail_seconds:
        Length of recent trajectory trail to emphasize.
    """

    def __init__(
        self,
        log: SimLog,
        outdir: str | Path = "out",
        robot_length: float = 0.9,
        robot_width: float = 0.6,
        trail_seconds: float = 4.0,
    ):
        self.log = log
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.robot_length = robot_length
        self.robot_width = robot_width
        self.trail_seconds = trail_seconds

    def save_gif(
        self,
        filename: str = "replay.gif",
        fps: int = 12,
        every: int = 1,
        dpi: int = 120,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        show_preview: bool = True,
        show_modes: bool = True,
    ) -> Path:
        """Save a trajectory replay GIF.

        Parameters
        ----------
        filename:
            Output GIF filename.
        fps:
            Frames per second in the exported GIF.
        every:
            Use every Nth simulation sample. Increase this to reduce file size.
        dpi:
            GIF resolution.
        xlim, ylim:
            Optional plot limits. If omitted, limits are inferred from trajectory,
            reference, and obstacle.
        show_preview:
            Draw a short future reference preview.
        show_modes:
            Color the robot body by supervisor mode.

        Returns
        -------
        Path
            Path to the written GIF.
        """
        if every < 1:
            raise ValueError("every must be >= 1")
        if fps < 1:
            raise ValueError("fps must be >= 1")

        L = self.log
        idx = np.arange(0, len(L.t), every)
        if len(idx) == 0:
            raise ValueError("SimLog contains no samples")

        x = L.z[:, 0]
        y = L.z[:, 1]
        yaw = L.z[:, 2]
        ref = L.ref

        if xlim is None or ylim is None:
            pad = 1.5
            ox = L.obstacle.spec.x
            oy = L.obstacle.spec.y
            rs = L.obstacle.spec.r_safe
            xmin = min(np.min(x), np.min(ref[:, 0]), ox - rs) - pad
            xmax = max(np.max(x), np.max(ref[:, 0]), ox + rs) + pad
            ymin = min(np.min(y), np.min(ref[:, 1]), oy - rs) - pad
            ymax = max(np.max(y), np.max(ref[:, 1]), oy + rs) + pad
            xlim = xlim or (float(xmin), float(xmax))
            ylim = ylim or (float(ymin), float(ymax))

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Skid-Steer MPC Replay")
        ax.grid(True, alpha=0.35)

        # Reference and obstacle
        ax.plot(ref[:, 0], ref[:, 1], "--", linewidth=2.0, label="reference")
        ax.plot(x, y, linewidth=1.2, alpha=0.25, label="full trajectory")

        obs = L.obstacle.spec
        ax.add_patch(Circle((obs.x, obs.y), obs.radius, alpha=0.35, label="obstacle"))
        ax.add_patch(Circle((obs.x, obs.y), obs.r_safe, fill=False, linestyle="--", linewidth=1.5))

        # Dynamic artists
        recent_line, = ax.plot([], [], linewidth=3.0, label="recent trail")
        point, = ax.plot([], [], "o", markersize=5)
        preview_line, = ax.plot([], [], ":", linewidth=2.0, label="reference preview")

        body = Rectangle(
            (-self.robot_length / 2, -self.robot_width / 2),
            self.robot_length,
            self.robot_width,
            fill=True,
            alpha=0.75,
        )
        ax.add_patch(body)

        mode_text = ax.text(
            0.02,
            0.96,
            "",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85),
        )

        time_text = ax.text(
            0.98,
            0.96,
            "",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85),
        )

        mode_colors = {
            int(CtrlMode.DISABLE): "0.5",
            int(CtrlMode.NOMINAL): "C0",
            int(CtrlMode.DEGRADED): "C1",
            int(CtrlMode.SAFE_STOP): "C3",
            int(CtrlMode.FAULT): "C3",
        }

        trail_n = max(2, int(round(self.trail_seconds / (L.t[1] - L.t[0])))) if len(L.t) > 1 else 2
        preview_n = max(2, int(round(2.0 / (L.t[1] - L.t[0])))) if len(L.t) > 1 else 2

        def init():
            recent_line.set_data([], [])
            point.set_data([], [])
            preview_line.set_data([], [])
            mode_text.set_text("")
            time_text.set_text("")
            return recent_line, point, preview_line, body, mode_text, time_text

        def update(frame_idx):
            i = int(idx[frame_idx])

            lo = max(0, i - trail_n)
            recent_line.set_data(x[lo:i + 1], y[lo:i + 1])
            point.set_data([x[i]], [y[i]])

            if show_preview:
                hi = min(len(ref), i + preview_n)
                preview_line.set_data(ref[i:hi, 0], ref[i:hi, 1])
            else:
                preview_line.set_data([], [])

            transform = (
                Affine2D()
                .rotate_around(0.0, 0.0, yaw[i])
                .translate(x[i], y[i])
                + ax.transData
            )
            body.set_transform(transform)

            mode = int(L.mode[i])
            if show_modes:
                body.set_facecolor(mode_colors.get(mode, "C0"))

            mode_name = CtrlMode(mode).name if mode in [int(m) for m in CtrlMode] else str(mode)
            mode_text.set_text(f"mode: {mode_name}")
            time_text.set_text(f"t = {L.t[i]:.1f} s")

            return recent_line, point, preview_line, body, mode_text, time_text

        anim = FuncAnimation(
            fig,
            update,
            frames=len(idx),
            init_func=init,
            interval=1000 / fps,
            blit=True,
        )

        path = self.outdir / filename
        anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)
        return path
