"""Tests for the exponential decay model and the three-state transitions."""

from __future__ import annotations

import math

import pytest

from bodyfollowing.identity.states import (
    confidence_after,
    decay_confidence,
    half_life,
    state_for,
    time_to_threshold,
)
from bodyfollowing.types import IdentityState


def test_decay_follows_exponential_curve():
    tau = 2.0
    assert decay_confidence(1.0, 0.0, tau) == 1.0
    assert decay_confidence(1.0, tau, tau) == pytest.approx(math.exp(-1), rel=1e-9)
    assert decay_confidence(1.0, 2 * tau, tau) == pytest.approx(math.exp(-2), rel=1e-9)
    assert decay_confidence(1.0, 10 * tau, tau) < 1e-4


def test_decay_is_multiplicative():
    """Decaying in two steps equals decaying in one step (semigroup property)."""
    tau = 3.0
    one_step = decay_confidence(0.9, 4.0, tau)
    two_step = decay_confidence(decay_confidence(0.9, 1.5, tau), 2.5, tau)
    assert one_step == pytest.approx(two_step, rel=1e-12)


def test_decay_half_life_matches_definition():
    tau = 2.0
    assert decay_confidence(1.0, half_life(tau), tau) == pytest.approx(0.5, rel=1e-9)


def test_decay_guards():
    assert decay_confidence(0.5, -1.0, 2.0) == 0.5   # time never runs backwards
    assert decay_confidence(0.5, 1.0, 0.0) == 0.0    # degenerate tau
    assert decay_confidence(0.0, 5.0, 2.0) == 0.0


def test_confidence_after_uses_elapsed_time():
    assert confidence_after(0.8, 10.0, 12.0, 2.0) == pytest.approx(
        0.8 * math.exp(-1.0), rel=1e-9
    )


def test_time_to_threshold_matches_simulation():
    tau, conf, thr = 2.0, 0.95, 0.25
    predicted = time_to_threshold(conf, thr, tau)
    assert decay_confidence(conf, predicted, tau) == pytest.approx(thr, rel=1e-9)
    # Already below the threshold -> no waiting.
    assert time_to_threshold(0.2, 0.25, tau) == 0.0
    assert time_to_threshold(0.0, 0.1, tau) == math.inf


# --------------------------------------------------------------------- states
def _state(conf, has_body=True, since_face=0.0, since_body=0.0, observed=True):
    return state_for(
        confidence=conf,
        has_live_body=has_body,
        seconds_since_face=since_face,
        seconds_since_body=since_body,
        confirm_threshold=0.70,
        unknown_threshold=0.25,
        freshness_window=1.0,
        body_grace_seconds=1.5,
        enough_observations=observed,
    )


def test_confirmed_requires_fresh_face_and_body():
    assert _state(0.9, has_body=True, since_face=0.2) is IdentityState.CONFIRMED
    # No fresh face -> cannot be confirmed even with high confidence.
    assert _state(0.9, has_body=True, since_face=2.0) is IdentityState.TEMPORARY
    # No body -> cannot be confirmed.
    assert _state(0.9, has_body=False, since_face=0.2) is IdentityState.TEMPORARY


def test_temporary_when_body_alive_but_face_missing():
    assert _state(0.6, has_body=True, since_face=3.0) is IdentityState.TEMPORARY


def test_unknown_below_floor():
    assert _state(0.1, has_body=True, since_face=3.0) is IdentityState.UNKNOWN
    assert _state(0.1, has_body=False, since_face=9.0, since_body=9.0) is IdentityState.UNKNOWN


def test_body_grace_then_unknown():
    # Body just lost: still temporary thanks to the grace period.
    assert _state(0.8, has_body=False, since_face=0.5, since_body=0.4) is IdentityState.TEMPORARY
    # Grace expired and confidence low -> unknown.
    assert _state(0.4, has_body=False, since_face=5.0, since_body=5.0) is IdentityState.UNKNOWN


def test_confirm_threshold_is_a_hard_boundary():
    assert _state(0.699, since_face=0.0) is not IdentityState.CONFIRMED
    assert _state(0.70, since_face=0.0) is IdentityState.CONFIRMED


def test_confirm_needs_enough_observations():
    """One observation is not enough when ``min_confirmations`` is higher."""
    assert _state(0.95, observed=False) is IdentityState.TEMPORARY
    assert _state(0.95, observed=True) is IdentityState.CONFIRMED


def test_full_decay_trajectory_passes_through_all_three_states():
    """A single exponential decay must visit CONFIRMED -> TEMPORARY -> UNKNOWN."""
    tau, conf = 2.0, 0.95
    seen = []
    for step in range(0, 200):
        t = step * 0.05
        c = decay_confidence(conf, t, tau)
        state = _state(c, has_body=True, since_face=t)
        if not seen or seen[-1] is not state:
            seen.append(state)
    assert seen == [
        IdentityState.CONFIRMED,
        IdentityState.TEMPORARY,
        IdentityState.UNKNOWN,
    ]
