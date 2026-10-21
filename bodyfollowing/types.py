"""Core data types for the body-following / face-association module.

All geometric quantities are in **pixel coordinates** of the source frame,
with the origin at the top-left corner.  Boxes are stored as ``x1, y1, x2, y2``
(float, x2/y2 exclusive of the last pixel but treated as continuous).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Box:
    """An axis-aligned bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float

    # ---------------------------------------------------------------- ctors
    @staticmethod
    def from_xywh(x: float, y: float, w: float, h: float) -> "Box":
        return Box(float(x), float(y), float(x + w), float(y + h))

    @staticmethod
    def from_cxcyw(cx: float, cy: float, w: float, h: float) -> "Box":
        return Box.from_xywh(cx - w / 2.0, cy - h / 2.0, w, h)

    # ------------------------------------------------------------ properties
    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return (self.cx, self.cy)

    @property
    def area_ratio(self) -> float:
        return self.area / float(self.width * self.height) if self.width else 0.0

    # ---------------------------------------------------------------- ops
    def as_int(self) -> tuple[int, int, int, int]:
        return (
            int(round(self.x1)),
            int(round(self.y1)),
            int(round(self.x2)),
            int(round(self.y2)),
        )

    def as_xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    def as_tlwh(self) -> tuple[int, int, int, int]:
        x, y, x2, y2 = self.as_int()
        return (x, y, x2 - x, y2 - y)

    def iou(self, other: "Box") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def intersection_area(self, other: "Box") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    def contains(self, px: float, py: float) -> bool:
        return self.x1 <= px <= self.x2 and self.y1 <= py <= self.y2

    def contains_box(self, other: "Box", min_ratio: float = 0.0) -> bool:
        """True if at least ``min_ratio`` of *other* lies inside this box."""
        if other.area <= 0:
            return False
        return self.intersection_area(other) / other.area >= min_ratio

    def expand(self, factor: float) -> "Box":
        w, h = self.width, self.height
        dx, dy = w * (factor - 1.0) / 2.0, h * (factor - 1.0) / 2.0
        return Box(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy)

    def clip(self, w: float, h: float) -> "Box":
        return Box(
            max(0.0, min(self.x1, w)),
            max(0.0, min(self.y1, h)),
            max(0.0, min(self.x2, w)),
            max(0.0, min(self.y2, h)),
        )

    def top_fraction(self, py: float) -> float:
        """Vertical position of ``py`` inside the box as a fraction of height."""
        if self.height <= 0:
            return 0.0
        return (py - self.y1) / self.height

    def to_dict(self) -> dict[str, Any]:
        return {
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
        }


@dataclass
class Detection:
    """A single detector output (person box, face box, ...)."""

    box: Box
    score: float
    class_id: int = 0
    class_name: str = "person"
    landmarks: Optional[list[tuple[float, float]]] = None
    track_id: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": self.box.to_dict(),
            "score": round(self.score, 4),
            "class": self.class_name,
            "track_id": self.track_id,
        }


class IdentityState(str, enum.Enum):
    """The honest three-state identity model.

    ``CONFIRMED``
        A face was positively observed on the tracked body *recently* and the
        link confidence is high.  We know who this is.

    ``TEMPORARY``
        The body is still tracked but the face is momentarily not visible
        (head turn, occlusion, detection miss).  The identity is *carried* with
        an exponentially decaying confidence.  This is a hypothesis, not a fact.

    ``UNKNOWN``
        Confidence decayed below the floor, the body track was lost, or the
        association was ambiguous.  The module makes **no** identity claim.
    """

    CONFIRMED = "confirmed"
    TEMPORARY = "temporary"
    UNKNOWN = "unknown"

    @property
    def is_assertive(self) -> bool:
        """True only for states that assert a positive identity."""
        return self is IdentityState.CONFIRMED

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass
class BodyTrack:
    """State of one tracked person body."""

    track_id: int
    box: Box
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    score: float = 0.0
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    confirmed: bool = False

    @property
    def head_box(self) -> Box:
        """Heuristic head region: top slice of the body box."""
        h = self.box.height
        return Box(self.box.x1, self.box.y1, self.box.x2, self.box.y1 + 0.30 * h)

    @property
    def head_center(self) -> tuple[float, float]:
        hb = self.head_box
        return (hb.cx, hb.cy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "box": self.box.to_dict(),
            "score": round(self.score, 4),
            "hits": self.hits,
            "age": self.age,
            "confirmed": self.confirmed,
        }


@dataclass
class FaceTrack:
    """State of one tracked face."""

    track_id: int
    box: Box
    score: float = 0.0
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    yaw: float = 0.0  # estimated head yaw in degrees (0 = frontal)
    descriptor: Optional[Any] = None  # appearance signature for re-id

    @property
    def is_frontal(self) -> bool:
        return abs(self.yaw) <= 35.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "box": self.box.to_dict(),
            "score": round(self.score, 4),
            "yaw": round(self.yaw, 3),
        }


@dataclass
class Identity:
    """A tracked person with an explicit, decaying identity claim.

    The identity id is stable for as long as the module keeps believing the
    same physical person is being followed.  The *state* is allowed to move
    between CONFIRMED / TEMPORARY / UNKNOWN; the id is only retired when the
    person has been missing for longer than ``retire_after``.
    """

    identity_id: int
    state: IdentityState = IdentityState.UNKNOWN
    confidence: float = 0.0
    #: Enrolled name, set when the face matched a known person.  ``None`` means
    #: "tracked but not recognized" -- the module never invents a name.
    name: Optional[str] = None
    body_track_id: Optional[int] = None
    face_track_id: Optional[int] = None
    last_face_time: float = -1.0
    last_update: float = 0.0
    body_lost_time: float = -1.0
    association_score: float = 0.0
    face_observations: int = 0
    total_confirmed_time: float = 0.0
    total_temporary_time: float = 0.0
    total_unknown_time: float = 0.0
    state_changes: list[tuple[float, IdentityState]] = field(default_factory=list)
    descriptor: Optional[Any] = None

    # ------------------------------------------------------------- helpers
    @property
    def seconds_since_face(self) -> float:
        if self.last_face_time < 0:
            return math.inf
        return max(0.0, self.last_update - self.last_face_time)

    @property
    def seconds_since_body(self) -> float:
        if self.body_lost_time < 0:
            return 0.0
        return max(0.0, self.last_update - self.body_lost_time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "state": self.state.value,
            "confidence": round(self.confidence, 4),
            "name": self.name,
            "body_track_id": self.body_track_id,
            "face_track_id": self.face_track_id,
            "face_observations": self.face_observations,
            "association_score": round(self.association_score, 4),
        }


@dataclass
class IdentityClaim:
    """The public, honest output of the identity module for one frame."""

    identity_id: Optional[int]
    state: IdentityState
    confidence: float
    #: Enrolled name of the target, when recognized.  ``None`` = no name known.
    name: Optional[str] = None
    body_track_id: Optional[int] = None
    face_track_id: Optional[int] = None
    reason: str = ""

    @property
    def asserts_identity(self) -> bool:
        return self.identity_id is not None and self.state.is_assertive

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "state": self.state.value,
            "confidence": round(self.confidence, 4),
            "name": self.name,
            "body_track_id": self.body_track_id,
            "face_track_id": self.face_track_id,
            "asserts_identity": self.asserts_identity,
            "reason": self.reason,
        }


@dataclass
class CentringCommand:
    """Camera-centring command produced for the locked target."""

    pan: float = 0.0
    tilt: float = 0.0
    error_x: float = 0.0
    error_y: float = 0.0
    active: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pan": round(self.pan, 4),
            "tilt": round(self.tilt, 4),
            "error_x": round(self.error_x, 1),
            "error_y": round(self.error_y, 1),
            "active": self.active,
            "reason": self.reason,
        }


@dataclass
class FrameResult:
    """Everything the pipeline produced for a single frame."""

    frame_index: int
    timestamp: float
    frame_size: tuple[int, int]
    bodies: list[BodyTrack] = field(default_factory=list)
    faces: list[FaceTrack] = field(default_factory=list)
    identities: list[Identity] = field(default_factory=list)
    claim: IdentityClaim = field(
        default_factory=lambda: IdentityClaim(None, IdentityState.UNKNOWN, 0.0)
    )
    command: CentringCommand = field(default_factory=CentringCommand)
    face_body_pairs: list[tuple[int, int, float]] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    def identity_by_id(self, identity_id: int) -> Optional[Identity]:
        for ident in self.identities:
            if ident.identity_id == identity_id:
                return ident
        return None
