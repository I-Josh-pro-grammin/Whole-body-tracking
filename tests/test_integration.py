"""End-to-end tests: the pipeline over scripted scenarios and stub detectors."""

from __future__ import annotations

import numpy as np
import pytest

from bodyfollowing.config import PipelineConfig
from bodyfollowing.evaluation.metrics import (
    EXPECTATIONS,
    check_expectation,
    evaluate_scene,
)
from bodyfollowing.perception.appearance import DetectionBatch
from bodyfollowing.pipeline import BodyFollowingPipeline
from bodyfollowing.sources.simulated import SCENARIOS, build_scenario
from bodyfollowing.types import Box, Detection, IdentityState


class StubDetector:
    """Feeds pre-baked detections so the pixel path can be tested without ML."""

    def __init__(self, batches: list[DetectionBatch]) -> None:
        self.batches = batches
        self.index = 0

    def detect(self, frame: np.ndarray) -> DetectionBatch:
        batch = self.batches[min(self.index, len(self.batches) - 1)]
        self.index += 1
        return batch


def person_frame_detection(cx: float = 320.0) -> DetectionBatch:
    body = Box.from_cxcyw(cx, 300, 100, 250)
    face = Box.from_cxcyw(cx, body.y1 + 19, 33, 42)
    return DetectionBatch(
        persons=[Detection(box=body, score=0.9)],
        faces=[Detection(box=face, score=0.95)],
        face_descriptors=[np.ones(40) / np.sqrt(40)],
        face_yaws=[0.0],
    )


def test_pipeline_runs_on_pixel_frames_with_a_detector():
    pipe = BodyFollowingPipeline(PipelineConfig(), detector=StubDetector([DetectionBatch()]))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipe.step(frame, now=0.0)
    assert result.frame_size == (640, 480)
    assert result.claim.state in set(IdentityState)


def test_pipeline_without_detector_or_detections_raises():
    pipe = BodyFollowingPipeline(PipelineConfig())
    with pytest.raises(RuntimeError):
        pipe.step(np.zeros((48, 64, 3), dtype=np.uint8), now=0.0)


def test_upward_face_mapping_confirms():
    batch = person_frame_detection()
    pipe = BodyFollowingPipeline(PipelineConfig(), detector=StubDetector([batch]))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # A body track must be confirmed (min_hits) before it can carry identity.
    for step in range(4):
        result = pipe.step(frame, now=0.1 * step)
    assert result.face_body_pairs
    assert result.identities
    assert result.claim.state is IdentityState.CONFIRMED


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_meets_its_expectations(name):
    result = evaluate_scene(build_scenario(name))
    checks = check_expectation(result, EXPECTATIONS[name])
    failures = [f"{c.name}: {c.detail}" for c in checks if not c.ok]
    assert not failures, f"{name} failed: {failures}"


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_never_confirms_a_wrong_person(name):
    metrics = evaluate_scene(build_scenario(name)).metrics
    assert metrics["confirmed_wrong"] == 0
    assert metrics["wrong_person_frames"] == 0


def test_long_absence_reaches_unknown_then_recovers():
    result = evaluate_scene(build_scenario("long_absence"))
    states = [r.state for r in result.records]
    assert "unknown" in states
    # The identity id used before and after the absence must be the same.
    assert result.metrics["identities_claimed"] == 1
    assert states[-1] == "confirmed"


def test_decay_tau_changes_how_long_temporary_survives():
    from bodyfollowing.sources.simulated import long_absence_scene

    def first_unknown(tau: float) -> float:
        cfg = PipelineConfig()
        cfg.auto_lock = False
        cfg.identity.decay_tau = tau
        result = evaluate_scene(long_absence_scene(), config=cfg)
        return next(
            r.timestamp
            for r in result.records
            if r.timestamp > 2.0 and r.state == "unknown"
        )

    fast = first_unknown(1.0)
    slow = first_unknown(4.0)
    assert slow > fast
    # Roughly a tau-driven scaling: 4x the constant should delay the floor.
    assert (slow - 2.0) / (fast - 2.0) > 2.0
