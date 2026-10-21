"""Constant-velocity Kalman filter for a single bounding box.

State vector (8-D)::

    [cx, cy, w, h, vcx, vcy, vw, vh]

Process and measurement noise are scaled to the box size, which keeps the
filter equally well behaved for a person close to the camera and for one at
the far end of a corridor.  Written with plain numpy so the module stays
CPU-only and dependency-light.
"""

from __future__ import annotations

import numpy as np

from ..types import Box

_DIM_X = 8
_DIM_Z = 4


class BoxKalman:
    """Kalman filter tracking ``(cx, cy, w, h)`` with constant velocity."""

    def __init__(self, box: Box, measurement_noise: float = 10.0) -> None:
        w = max(box.width, 1.0)
        h = max(box.height, 1.0)
        self.x = np.array(
            [box.cx, box.cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self.F = np.eye(_DIM_X)
        for i in range(4):
            self.F[i, i + 4] = 1.0
        self.H = np.zeros((_DIM_Z, _DIM_X))
        for i in range(4):
            self.H[i, i] = 1.0

        self.P = np.eye(_DIM_X) * 10.0
        self.P[4:, 4:] *= 100.0  # high uncertainty on the initial velocity
        self.measurement_noise = measurement_noise

    # ------------------------------------------------------------------ noise
    def _std(self) -> tuple[np.ndarray, np.ndarray]:
        w = max(abs(self.x[2]), 1.0)
        h = max(abs(self.x[3]), 1.0)
        std_pos = np.array([w / 20.0, h / 20.0, w / 20.0, h / 20.0])
        std_vel = np.array([w / 160.0, h / 160.0, w / 160.0, h / 160.0])
        return std_pos, std_vel

    # ---------------------------------------------------------------- predict
    def predict(self, dt: float = 1.0) -> None:
        """Advance the state by ``dt`` frames (dt=1 is one video frame)."""
        std_pos, std_vel = self._std()
        q = np.concatenate([std_pos, std_vel]) ** 2
        if dt != 1.0:
            f = np.eye(_DIM_X)
            for i in range(4):
                f[i, i + 4] = dt
            self.x = f @ self.x
            self.P = f @ self.P @ f.T + np.diag(q) * dt
        else:
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + np.diag(q)

        # Never let a box invert itself.
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)

    # ----------------------------------------------------------------- update
    def update(self, box: Box) -> None:
        std_pos, _ = self._std()
        r = np.diag(std_pos ** 2) * self.measurement_noise
        z = np.array([box.cx, box.cy, max(box.width, 1.0), max(box.height, 1.0)])

        s = self.H @ self.P @ self.H.T + r
        k = self.P @ self.H.T @ np.linalg.inv(s)
        innovation = z - self.H @ self.x
        self.x = self.x + k @ innovation
        i_kh = np.eye(_DIM_X) - k @ self.H
        self.P = i_kh @ self.P @ i_kh.T + k @ r @ k.T  # Joseph form
        self.P = 0.5 * (self.P + self.P.T)             # keep symmetric
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)

    # --------------------------------------------------------------- getters
    @property
    def box(self) -> Box:
        return Box.from_cxcyw(self.x[0], self.x[1], self.x[2], self.x[3])

    @property
    def velocity(self) -> tuple[float, float, float, float]:
        return (float(self.x[4]), float(self.x[5]), float(self.x[6]), float(self.x[7]))
