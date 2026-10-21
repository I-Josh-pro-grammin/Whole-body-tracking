"""Tests for persistent enrollment: matching, persistence and honesty guards."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bodyfollowing.association.face_body import AssociationResult, FaceBodyMatch
from bodyfollowing.config import AssociationConfig, IdentityConfig
from bodyfollowing.identity.enrollment import EnrollmentBook, Match, Person
from bodyfollowing.identity.manager import IdentityManager
from bodyfollowing.types import BodyTrack, Box, FaceTrack, IdentityState


def descriptor(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=40)
    return vec / np.linalg.norm(vec)


def make_body(track_id: int = 1, x: float = 300.0) -> BodyTrack:
    return BodyTrack(track_id=track_id, box=Box.from_cxcyw(x, 300, 100, 250), confirmed=True)


def make_face(track_id: int, body: BodyTrack, desc=None, tsu: int = 0) -> FaceTrack:
    fh = 0.17 * body.box.height
    fw = 0.78 * fh
    return FaceTrack(
        track_id=track_id,
        box=Box.from_cxcyw(body.box.cx, body.box.y1 + 0.075 * body.box.height, fw, fh),
        score=0.95,
        time_since_update=tsu,
        descriptor=desc,
    )


def make_assoc(face: FaceTrack | None, body: BodyTrack, score: float = 0.9, **kw):
    if face is None:
        return AssociationResult()
    return AssociationResult(
        matches=[
            FaceBodyMatch(face_id=face.track_id, body_id=body.track_id, score=score, **kw)
        ]
    )


# ------------------------------------------------------------------- matching
def test_enrolled_person_is_matched_by_similarity():
    book = EnrollmentBook(None)
    ref = descriptor(7)
    assert book.enroll("josh", ref, now=0.0) is True

    match = book.match(ref + 0.02 * np.random.default_rng(0).normal(size=ref.size))
    assert isinstance(match, Match)
    assert match.name == "josh"
    assert match.similarity > 0.9


def test_unknown_person_yields_no_match():
    book = EnrollmentBook(None)
    book.enroll("josh", descriptor(7))
    assert book.match(descriptor(99)) is None


def test_ambiguous_entrant_is_refused_not_guessed():
    """Two equally good candidates must produce no answer at all."""
    book = EnrollmentBook(None, similarity_threshold=0.5, margin=0.05)
    ref = descriptor(3)
    book.enroll("twin_a", ref)
    book.enroll("twin_b", ref)
    assert book.match(ref) is None


def test_match_below_threshold_is_none():
    book = EnrollmentBook(None, similarity_threshold=0.95)
    book.enroll("josh", descriptor(7))
    assert book.match(descriptor(8)) is None


def test_enroll_rejects_bad_input_and_leaves_book_unchanged():
    book = EnrollmentBook(None)
    assert book.enroll("", descriptor(1)) is False
    assert book.enroll("   ", descriptor(1)) is False
    assert book.enroll("x", None) is False
    assert book.enroll("x", np.zeros(0)) is False
    assert book.enroll("x", np.zeros(40)) is False  # zero vector is degenerate
    assert len(book) == 0
    assert book.match(None) is None


def test_samples_are_capped_per_person():
    book = EnrollmentBook(None, max_samples=3)
    ref = descriptor(1)
    for i in range(10):
        book.enroll("josh", ref + 0.01 * i, now=float(i))
    assert len(book.people["josh"].samples) == 3


# ---------------------------------------------------------------- persistence
def test_book_survives_a_round_trip(tmp_path):
    path = tmp_path / "enrollments.json"
    book = EnrollmentBook(path)
    book.enroll("josh", descriptor(7), now=1.0)
    book.enroll("josh", descriptor(7) * 0.99, now=2.0)
    book.enroll("ada", descriptor(21), now=3.0)
    book.save()
    assert path.exists()

    reloaded = EnrollmentBook.load(path)
    assert reloaded.names() == ["ada", "josh"]
    assert reloaded.match(descriptor(7)).name == "josh"
    assert len(reloaded.people["josh"].samples) == 2


def test_load_tolerates_missing_and_corrupt_files(tmp_path):
    missing = EnrollmentBook.load(tmp_path / "nope.json")
    assert len(missing) == 0

    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    assert len(EnrollmentBook.load(bad)) == 0

    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert len(EnrollmentBook.load(wrong)) == 0


def test_forget_and_clear(tmp_path):
    book = EnrollmentBook(tmp_path / "e.json")
    book.enroll("josh", descriptor(7))
    assert "josh" in book
    assert book.forget("josh") is True
    assert book.forget("josh") is False
    book.enroll("ada", descriptor(8))
    book.clear()
    assert len(book) == 0


def test_person_round_trips_through_dict():
    person = Person(name="josh", samples=[descriptor(5)], created=1.0, updated=2.0)
    again = Person.from_dict(person.to_dict())
    assert again.name == "josh"
    assert again.created == 1.0 and again.updated == 2.0
    assert again.descriptor is not None


# ------------------------------------------------- recognition in the manager
def manager_with_book(**book_kw) -> IdentityManager:
    book = EnrollmentBook(None, **book_kw)
    return IdentityManager(
        IdentityConfig(), AssociationConfig(), auto_lock=False, enrollment=book
    )


def test_recognized_face_labels_the_identity():
    mgr = manager_with_book()
    ref = descriptor(7)
    mgr.enrollment.enroll("josh", ref)

    body = make_body()
    face = make_face(11, body, ref)
    identities = mgr.update([body], [face], make_assoc(face, body), now=0.0)
    assert identities[0].name == "josh"

    mgr.lock(identities[0].identity_id)
    assert mgr.claim(0.0).name == "josh"
    assert "josh" in mgr.claim(0.0).reason


def test_unrecognized_face_gets_no_name():
    """Never invent a name: an unknown visitor stays nameless."""
    mgr = manager_with_book()
    mgr.enrollment.enroll("josh", descriptor(7))

    body = make_body()
    face = make_face(11, body, descriptor(55))
    identities = mgr.update([body], [face], make_assoc(face, body), now=0.0)
    assert identities[0].name is None
    assert mgr.claim(0.0).name is None


def test_name_is_not_overwritten_by_a_later_marginal_frame():
    """Once committed to a name we do not silently relabel the identity."""
    mgr = manager_with_book(similarity_threshold=0.5, margin=0.05)
    mgr.enrollment.enroll("josh", descriptor(7))
    mgr.enrollment.enroll("ada", descriptor(21))

    body = make_body()
    face = make_face(11, body, descriptor(7))
    mgr.update([body], [face], make_assoc(face, body), now=0.0)
    assert mgr.identities[1].name == "josh"

    later = make_face(11, body, descriptor(21))
    mgr.update([body], [later], make_assoc(later, body), now=0.2)
    assert mgr.identities[1].name == "josh"
    assert any(e[2] == "recognition_conflict" for e in mgr.events)


def test_enroll_target_records_the_locked_identity():
    mgr = manager_with_book()
    body = make_body()
    face = make_face(11, body, descriptor(7))
    mgr.update([body], [face], make_assoc(face, body), now=0.0)
    mgr.lock(1)

    assert mgr.enroll_target("josh", now=0.1) == "josh"
    assert mgr.last_enrollment == "josh"
    assert "josh" in mgr.enrollment
    assert mgr.identities[1].name == "josh"


def test_enroll_target_without_a_face_signature_is_refused():
    """A premature enroll must not create a bogus record."""
    mgr = manager_with_book()
    assert mgr.enroll_target("josh") is None
    assert len(mgr.enrollment) == 0


def test_enroll_target_without_a_book_is_refused():
    mgr = IdentityManager(IdentityConfig(), AssociationConfig(), auto_lock=False)
    assert mgr.enroll_target("josh") is None


def test_auto_lock_prefers_a_recognized_person_over_a_stranger():
    """With a name in the room, lock the person we know, not the loudest face."""
    book = EnrollmentBook(None)
    book.enroll("josh", descriptor(7))
    mgr = IdentityManager(
        IdentityConfig(), AssociationConfig(), auto_lock=True, enrollment=book
    )

    # Both people appear in the same first frame: a high-scoring stranger and
    # the enrolled person, who scores slightly lower.  The name must win.
    stranger_body = make_body(track_id=1, x=120.0)
    stranger_face = make_face(11, stranger_body, descriptor(80))
    known_body = make_body(track_id=2, x=520.0)
    known_face = make_face(12, known_body, descriptor(7))

    mgr.update(
        [stranger_body, known_body],
        [stranger_face, known_face],
        AssociationResult(
            matches=[
                FaceBodyMatch(face_id=11, body_id=1, score=0.99),
                FaceBodyMatch(face_id=12, body_id=2, score=0.9),
            ]
        ),
        now=0.0,
    )

    assert mgr.target is not None
    assert mgr.target.name == "josh"
    assert mgr.target.body_track_id == 2
