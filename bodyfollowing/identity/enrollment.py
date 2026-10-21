"""Persistent enrollment: remember a person's appearance across runs.

An ``EnrollmentBook`` is a small, JSON-backed registry of named appearance
descriptors.  It is what lets the module *know* somebody rather than merely
follow them for the length of one process:

* the operator teaches it a name once (``press e`` in the demo),
* the next run loads the same file and recognizes that person on sight,
* a recognized person is locked immediately instead of waiting for a
  click or for the first CONFIRMED identity to appear.

Honest limits
-------------
The descriptor is the colour-layout signature produced by
:mod:`bodyfollowing.perception.appearance` -- a per-channel HSV histogram plus
a coarse luminance grid.  It is **not** a biometric and it is **not** a face
embedding: it separates people who differ in hair, clothing or skin tone
under the lighting it was recorded in, and it will happily confuse two
similarly dressed strangers.  Matching therefore requires a high similarity
*and* a clear margin over the runner-up, and a failed match yields no claim
at all rather than a guess.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..utils.geometry import cosine_similarity

#: Schema version written into (and validated against) the JSON file.
SCHEMA_VERSION = 1


def _normalize(vector: Sequence[float] | np.ndarray) -> np.ndarray | None:
    """Return a unit-length float64 vector, or ``None`` if unusable."""
    arr = np.asarray(vector, dtype=np.float64).ravel()
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        return None
    return arr / norm


def _mean_descriptor(samples: Iterable[np.ndarray]) -> np.ndarray | None:
    """Average several observations into one unit vector."""
    stacked = [s for s in (_normalize(v) for v in samples) if s is not None]
    if not stacked:
        return None
    return _normalize(np.mean(np.stack(stacked, axis=0), axis=0))


@dataclass
class Match:
    """The outcome of comparing a descriptor against the book."""

    name: str
    similarity: float
    margin: float

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Match(name={self.name!r}, similarity={self.similarity:.3f})"


@dataclass
class Person:
    """One enrolled person: a name plus their recorded observations."""

    name: str
    samples: list[np.ndarray] = field(default_factory=list)
    created: float = 0.0
    updated: float = 0.0

    @property
    def descriptor(self) -> np.ndarray | None:
        """The averaged signature used for matching."""
        return _mean_descriptor(self.samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "created": self.created,
            "updated": self.updated,
            "samples": [np.asarray(s, dtype=float).tolist() for s in self.samples],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Person":
        samples = []
        for raw in data.get("samples", []):
            arr = np.asarray(raw, dtype=np.float64).ravel()
            if arr.size:
                samples.append(arr)
        return cls(
            name=str(data.get("name", "")),
            samples=samples,
            created=float(data.get("created", 0.0)),
            updated=float(data.get("updated", 0.0)),
        )


class EnrollmentBook:
    """A named registry of appearance descriptors, loadable from disk.

    Parameters
    ----------
    path:
        JSON file used for :meth:`save` / :meth:`load`.  ``None`` makes the
        book purely in-memory (handy in tests).
    similarity_threshold:
        Minimum cosine similarity for a match to be reported.
    margin:
        Minimum gap between the best and second-best person.  Without it two
        look-alikes are resolved by a coin flip, which is exactly the kind of
        silent wrong-person claim this module refuses to make.
    max_samples:
        Cap on stored observations per person, so a long session cannot grow
        the file without bound.  Oldest samples are dropped first.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        similarity_threshold: float = 0.80,
        margin: float = 0.05,
        max_samples: int = 12,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.similarity_threshold = float(similarity_threshold)
        self.margin = float(margin)
        self.max_samples = max(1, int(max_samples))
        self.people: dict[str, Person] = {}

    # ----------------------------------------------------------------- misc
    def __len__(self) -> int:
        return len(self.people)

    def __contains__(self, name: object) -> bool:
        return name in self.people

    def names(self) -> list[str]:
        return sorted(self.people)

    def forget(self, name: str) -> bool:
        """Remove one person.  Returns ``True`` if they existed."""
        return self.people.pop(name, None) is not None

    def clear(self) -> None:
        self.people.clear()

    # ---------------------------------------------------------------- enroll
    def enroll(
        self,
        name: str,
        descriptor: Sequence[float] | np.ndarray | None,
        *,
        now: float = 0.0,
    ) -> bool:
        """Record an observation of ``name``.

        Returns ``False`` (and records nothing) when the descriptor is empty,
        non-finite, or all-zero -- a silent bad enrollment would poison every
        later match for that person.
        """
        name = (name or "").strip()
        normalized = _normalize(descriptor) if descriptor is not None else None
        if not name or normalized is None:
            return False

        person = self.people.get(name)
        if person is None:
            person = Person(name=name, created=now)
            self.people[name] = person
        person.samples.append(normalized)
        if len(person.samples) > self.max_samples:
            del person.samples[: len(person.samples) - self.max_samples]
        person.updated = now
        return True

    def descriptor_for(self, name: str) -> np.ndarray | None:
        person = self.people.get(name)
        return person.descriptor if person is not None else None

    # ----------------------------------------------------------------- match
    def match(self, descriptor: Sequence[float] | np.ndarray | None) -> Match | None:
        """Find the enrolled person ``descriptor`` most resembles.

        Returns ``None`` when nothing clears the threshold, when a single
        person is enrolled and the similarity is still short of the bar, or
        when the two best candidates are too close to tell apart.  Ambiguity
        is reported as "no answer", never as a choice.
        """
        probe = _normalize(descriptor) if descriptor is not None else None
        if probe is None:
            return None

        scored: list[tuple[str, float]] = []
        for name, person in self.people.items():
            ref = person.descriptor
            if ref is None or ref.size != probe.size:
                continue
            scored.append((name, cosine_similarity(probe, ref)))
        if not scored:
            return None

        scored.sort(key=lambda item: item[1], reverse=True)
        best_name, best_sim = scored[0]
        runner_up = scored[1][1] if len(scored) > 1 else -1.0
        margin = best_sim - runner_up

        if best_sim < self.similarity_threshold:
            return None
        if runner_up >= 0.0 and margin < self.margin:
            return None
        return Match(name=best_name, similarity=float(best_sim), margin=float(margin))

    # ---------------------------------------------------------------- persist
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "people": [p.to_dict() for p in self.people.values()],
        }

    def save(self, path: str | Path | None = None) -> Path | None:
        """Write the book to JSON.  Returns the path written, if any."""
        target = Path(path) if path is not None else self.path
        if target is None:
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(target)  # atomic: never leave a half-written registry
        return target

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        similarity_threshold: float = 0.80,
        margin: float = 0.05,
        max_samples: int = 12,
    ) -> "EnrollmentBook":
        """Read a book from disk, tolerating a missing or corrupt file."""
        book = cls(
            path,
            similarity_threshold=similarity_threshold,
            margin=margin,
            max_samples=max_samples,
        )
        if book.path is None or not book.path.exists():
            return book
        try:
            data = json.loads(book.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return book  # a damaged registry must not stop the demo booting
        if not isinstance(data, dict):
            return book
        for raw in data.get("people", []):
            if not isinstance(raw, dict):
                continue
            person = Person.from_dict(raw)
            if person.name and person.samples:
                book.people[person.name] = person
        return book
