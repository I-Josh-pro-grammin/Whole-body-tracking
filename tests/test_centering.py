"""Tests for the target centring controller."""

from __future__ import annotations

import pytest

from bodyfollowing.config import CentringConfig
from bodyfollowing.control.centering import CentringController
from bodyfollowing.types import (
    BodyTrack,
    Box,
    FaceTrack,
    IdentityClaim,
    IdentityState,
)

FRAME = (640, 480)


def body_at(cx: float, cy: float = 300.0, h: float = 250.0, w: float = 100.0) -> BodyTrack:
    return BodyTrack(track_id=1, box=Box.from_cxcyw(cx, cy, w, h), confirmed=True)


def claim(state: IdentityState = IdentityState.CONFIRMED) -> IdentityClaim:
    return IdentityClaim(identity_id=1, state=state, confidence=0.9, body_track_id=1)


def controller(**kw) -> CentringController:
    return CentringController(CentringConfig(**kw))


def test_centred_target_produces_no_command():
    ctrl = controller()
    b = body_at(320)
    cmd = ctrl.update(b, None, claim(), FRAME)
    assert cmd.active
    assert cmd.pan == pytest.approx(0.0, abs=1e-6)
    assert abs(cmd.error_x) < 1.0


def test_target_right_of_centre_pans_right():
    ctrl = controller()
    # cy chosen so the aim point (upper body) lands exactly on the centre line.
    b = body_at(560, cy=320)
    cmd = ctrl.update(b, None, claim(), FRAME)
    assert cmd.error_x > 0
    assert cmd.pan > 0
    assert cmd.tilt == pytest.approx(0.0, abs=1e-6)


def test_target_left_of_centre_pans_left():
    ctrl = controller()
    cmd = ctrl.update(body_at(80), None, claim(), FRAME)
    assert cmd.pan < 0


def test_target_below_centre_tilts_down():
    ctrl = controller()
    b = body_at(320, cy=430, h=200)
    cmd = ctrl.update(b, None, claim(), FRAME)
    assert cmd.error_y > 0
    assert cmd.tilt < 0  # screen y grows downward; tilt-down is negative


def test_deadzone_suppresses_small_errors():
    ctrl = controller(deadzone_x=0.15, deadzone_y=0.15)
    # 8% off centre: inside the 15% dead-zone.
    cmd = ctrl.update(body_at(320 + 0.08 * 320), None, claim(), FRAME)
    assert cmd.pan == pytest.approx(0.0, abs=1e-6)
    assert cmd.error_x > 0  # the error is still reported


def test_slew_limit_ramps_up():
    ctrl = controller(max_slew=0.25, error_smoothing=0.0)
    b = body_at(620)
    first = ctrl.update(b, None, claim(), FRAME)
    assert first.pan == pytest.approx(0.25, abs=1e-6)
    second = ctrl.update(b, None, claim(), FRAME)
    assert second.pan == pytest.approx(0.50, abs=1e-6)
    assert second.pan > first.pan


def test_command_is_saturated():
    ctrl = controller(gain_x=10.0, max_slew=5.0, error_smoothing=0.0)
    cmd = ctrl.update(body_at(639), None, claim(), FRAME)
    assert -1.0 <= cmd.pan <= 1.0


def test_unknown_identity_does_not_chase():
    ctrl = controller()
    cmd = ctrl.update(body_at(600), None, claim(IdentityState.UNKNOWN), FRAME)
    assert not cmd.active
    assert cmd.pan == pytest.approx(0.0)
    assert "unknown" in cmd.reason.lower()


def test_temporary_identity_still_centres():
    ctrl = controller()
    cmd = ctrl.update(body_at(560), None, claim(IdentityState.TEMPORARY), FRAME)
    assert cmd.active
    assert cmd.pan > 0


def test_aims_at_face_when_confirmed():
    ctrl = controller()
    b = body_at(320)
    face = FaceTrack(track_id=9, box=Box.from_cxcyw(500, b.box.y1 + 20, 40, 50))
    cmd = ctrl.update(b, face, claim(IdentityState.CONFIRMED), FRAME)
    assert cmd.error_x == pytest.approx(500 - 320)


def test_falls_back_to_body_when_face_missing():
    ctrl = controller()
    b = body_at(320)
    cmd = ctrl.update(b, None, claim(IdentityState.CONFIRMED), FRAME)
    aim_y = b.box.y1 + ctrl.config.fallback_aim_fraction * b.box.height
    assert cmd.error_y == pytest.approx(aim_y - 240)


def test_hold_decays_command_to_rest():
    ctrl = controller(max_slew=0.3)
    ctrl.update(body_at(620), None, claim(), FRAME)
    held = ctrl.update(None, None, claim(IdentityState.UNKNOWN), FRAME)
    assert not held.active
    assert abs(held.pan) <= 0.3
