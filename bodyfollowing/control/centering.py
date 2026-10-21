"""Target centring logic.

Turns "where the target is in the frame" into a normalised pan/tilt command
that a camera gimbal (or a simulated one) can consume.

Design choices that matter in practice:

* **Aim at the face when the identity is confirmed**, otherwise at the upper
  body.  Centring on the feet makes the framing lurch when the detector box
  grows during an approach.
* **Dead-zone** so a stationary person does not produce servo jitter.
* **Smoothing + slew limit** so a detection glitch does not whip the camera.
* **Refuse to chase an UNKNOWN target.**  When the identity model is not sure,
  holding still is honest; chasing a look-alike is how robots lose people.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import CentringConfig
from ..types import BodyTrack, CentringCommand, FaceTrack, IdentityClaim, IdentityState
from ..utils.geometry import clamp, ema


@dataclass
class CentringController:
    """Converts target geometry into a normalised camera command."""

    config: CentringConfig = field(default_factory=CentringConfig)
    _last_pan: float = 0.0
    _last_tilt: float = 0.0
    _smooth_x: float = 0.0
    _smooth_y: float = 0.0
    _initialised: bool = False

    def reset(self) -> None:
        self._last_pan = 0.0
        self._last_tilt = 0.0
        self._smooth_x = 0.0
        self._smooth_y = 0.0
        self._initialised = False

    # --------------------------------------------------------------- aim point
    def aim_point(
        self,
        body: BodyTrack | None,
        face: FaceTrack | None,
        claim: IdentityClaim,
    ) -> tuple[float, float] | None:
        """Where the camera should look, in frame pixels."""
        if claim.state is IdentityState.CONFIRMED and face is not None:
            return face.box.center
        if body is not None:
            y = body.box.y1 + self.config.fallback_aim_fraction * body.box.height
            return (body.box.cx, y)
        if face is not None:
            return face.box.center
        return None

    # ------------------------------------------------------------------ update
    def update(
        self,
        body: BodyTrack | None,
        face: FaceTrack | None,
        claim: IdentityClaim,
        frame_size: tuple[int, int],
    ) -> CentringCommand:
        """Compute the command for one frame."""
        width, height = frame_size
        half_w = max(width / 2.0, 1.0)
        half_h = max(height / 2.0, 1.0)

        if claim.state is IdentityState.UNKNOWN:
            return self._hold("target identity unknown - holding position")

        aim = self.aim_point(body, face, claim)
        if aim is None:
            return self._hold("no target geometry this frame")

        ex = aim[0] - half_w
        ey = aim[1] - half_h
        nx = clamp(ex / half_w, -1.0, 1.0)
        ny = clamp(ey / half_h, -1.0, 1.0)

        if not self._initialised:
            self._smooth_x, self._smooth_y = nx, ny
            self._initialised = True
        else:
            self._smooth_x = ema(self._smooth_x, nx, 1.0 - self.config.error_smoothing)
            self._smooth_y = ema(self._smooth_y, ny, 1.0 - self.config.error_smoothing)

        sx = 0.0 if abs(self._smooth_x) <= self.config.deadzone_x else self._smooth_x
        sy = 0.0 if abs(self._smooth_y) <= self.config.deadzone_y else self._smooth_y

        # +pan = pan right, +tilt = tilt up (screen y grows downward).
        target_pan = clamp(self.config.gain_x * sx, -1.0, 1.0)
        target_tilt = clamp(-self.config.gain_y * sy, -1.0, 1.0)

        pan = self._slew(self._last_pan, target_pan)
        tilt = self._slew(self._last_tilt, target_tilt)
        self._last_pan, self._last_tilt = pan, tilt

        return CentringCommand(
            pan=pan,
            tilt=tilt,
            error_x=ex,
            error_y=ey,
            active=True,
            reason=f"centring on {claim.state.value} target",
        )

    def _slew(self, previous: float, target: float) -> float:
        delta = clamp(target - previous, -self.config.max_slew, self.config.max_slew)
        return previous + delta

    def _hold(self, reason: str) -> CentringCommand:
        # Decay the command to rest instead of snapping, so a real gimbal eases.
        self._last_pan = self._slew(self._last_pan, 0.0)
        self._last_tilt = self._slew(self._last_tilt, 0.0)
        return CentringCommand(
            pan=self._last_pan,
            tilt=self._last_tilt,
            error_x=0.0,
            error_y=0.0,
            active=False,
            reason=reason,
        )
