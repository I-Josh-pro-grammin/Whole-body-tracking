"""The honest three-state identity model.

The model deliberately separates *what we observed* from *what we believe*:

* A body track is a continuous motion hypothesis about a person.
* A face observation is the only thing that can **assert** an identity.
* Between face observations the belief is carried by an exponentially decaying
  confidence and is reported as ``TEMPORARY`` -- a hypothesis, never a fact.
* When the belief decays below a floor it becomes ``UNKNOWN`` and the module
  stops making any identity claim at all.

Confidence decay is time-based (not frame-based) so behaviour is identical at
different frame rates and real-time jitter does not distort the model::

    c(t) = c0 * exp(-(t - t0) / tau)
"""

from __future__ import annotations

import math

from ..types import IdentityState


def decay_confidence(confidence: float, elapsed: float, tau: float) -> float:
    """Exponentially decay ``confidence`` by ``elapsed`` seconds.

    ``tau`` is the e-folding time constant.  ``elapsed <= 0`` returns the
    confidence unchanged; ``tau <= 0`` collapses straight to zero.
    """
    if confidence <= 0.0:
        return 0.0
    if elapsed <= 0.0:
        return float(confidence)
    if tau <= 0.0:
        return 0.0
    return float(confidence * math.exp(-elapsed / tau))


def confidence_after(
    confidence: float,
    last_time: float,
    now: float,
    tau: float,
) -> float:
    """Confidence at ``now`` given a snapshot taken at ``last_time``."""
    return decay_confidence(confidence, now - last_time, tau)


def half_life(tau: float) -> float:
    """Seconds for confidence to halve (``tau * ln 2``)."""
    return tau * math.log(2.0)


def time_to_threshold(confidence: float, threshold: float, tau: float) -> float:
    """Seconds until ``confidence`` decays to ``threshold`` (inf if never)."""
    if confidence <= 0.0:
        return math.inf
    if threshold >= confidence:
        return 0.0
    if threshold <= 0.0:
        return math.inf
    return tau * math.log(confidence / threshold)


def state_for(
    confidence: float,
    has_live_body: bool,
    seconds_since_face: float,
    seconds_since_body: float,
    confirm_threshold: float,
    unknown_threshold: float,
    freshness_window: float,
    body_grace_seconds: float,
    enough_observations: bool = True,
) -> IdentityState:
    """Decide the state from the current evidence.

    ``CONFIRMED`` requires all of: a live body track, a *fresh* face
    observation, confidence at or above ``confirm_threshold``, and at least
    ``min_confirmations`` face observations.  A body alone can never confirm an
    identity -- that is the whole point of the three-state model.
    """
    if (
        has_live_body
        and seconds_since_face <= freshness_window
        and confidence >= confirm_threshold
        and enough_observations
    ):
        return IdentityState.CONFIRMED

    body_ok = has_live_body or seconds_since_body <= body_grace_seconds
    if body_ok and confidence >= unknown_threshold:
        return IdentityState.TEMPORARY

    return IdentityState.UNKNOWN


def describe(state: IdentityState, confidence: float) -> str:
    """Human-readable, non-overclaiming description of the current belief."""
    pct = round(confidence * 100)
    if state is IdentityState.CONFIRMED:
        return f"identity confirmed ({pct}%)"
    if state is IdentityState.TEMPORARY:
        return f"probably same person, unverified ({pct}%)"
    return "identity unknown"
