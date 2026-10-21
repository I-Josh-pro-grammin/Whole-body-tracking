"""Overlay rendering for demos and for the scenario contact sheets."""

from __future__ import annotations

import numpy as np

from ..types import FrameResult, IdentityState

STATE_COLORS = {
    IdentityState.CONFIRMED: (60, 200, 60),
    IdentityState.TEMPORARY: (40, 170, 255),
    IdentityState.UNKNOWN: (60, 60, 230),
}
STATE_LABELS = {
    IdentityState.CONFIRMED: "CONFIRMED",
    IdentityState.TEMPORARY: "TEMPORARY",
    IdentityState.UNKNOWN: "UNKNOWN",
}


def _cv2():
    import cv2

    return cv2


def _put(img, text, org, color, scale=0.5, thickness=1) -> None:
    cv2 = _cv2()
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


def _rect(img, box, color, thickness=2) -> None:
    cv2 = _cv2()
    x1, y1, x2, y2 = box.as_int()
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def _confidence_bar(img, x: int, y: int, w: int, h: int, conf: float, color) -> None:
    cv2 = _cv2()
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), 1)
    filled = int(round(w * max(0.0, min(1.0, conf))))
    if filled > 0:
        cv2.rectangle(img, (x + 1, y + 1), (x + filled - 1, y + h - 1), color, -1)


def draw_overlay(
    image: np.ndarray,
    result: FrameResult,
    show_all_tracks: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    """Draw the whole pipeline state onto a copy of ``image``."""
    img = image.copy()
    h, w = img.shape[:2]
    color = STATE_COLORS[result.claim.state]
    target_body_id = result.claim.body_track_id
    target_face_id = result.claim.face_track_id

    for body in result.bodies:
        is_target = body.track_id == target_body_id
        if not is_target:
            continue
        _rect(img, body.box, color, thickness)
        _rect(img, body.head_box, (90, 210, 240), 1)

    for face in result.faces:
        is_target = face.track_id == target_face_id
        if is_target:
            _rect(img, face.box, (255, 220, 60), 2)

    # Draw the target label last so face and body boxes cannot cover it.
    target = next((b for b in result.bodies if b.track_id == target_body_id), None)
    if target is not None:
        label = result.claim.name or "Unrecognized person"
        label += f" | {STATE_LABELS[result.claim.state]}"
        cv2 = _cv2()
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
        )
        label_x = max(0, min(w - text_w - 10, int(target.box.x1)))
        label_y = max(text_h + baseline + 6, int(target.box.y1))
        cv2.rectangle(
            img,
            (label_x, label_y - text_h - baseline - 6),
            (label_x + text_w + 10, label_y),
            color,
            -1,
        )
        _put(img, label, (label_x + 5, label_y - baseline - 3), (255, 255, 255), 0.52, 1)

    # Frame centre marker.
    cx, cy = w // 2, h // 2
    cv2 = _cv2()
    cv2.drawMarker(img, (cx, cy), (120, 120, 120), cv2.MARKER_CROSS, 18, 1)

    # HUD: one item per line so nothing can collide as text lengths change.
    panel = img.copy()
    cv2.rectangle(panel, (6, 6), (min(w - 6, 480), 178), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.55, img, 0.45, 0, img)

    state_text = STATE_LABELS[result.claim.state]
    _put(img, f"{state_text}", (16, 30), color, 0.7, 2)
    person = result.claim.name or "Unrecognized person"
    _put(img, f"locked person: {person}", (16, 52), (235, 235, 235), 0.5)
    _put(
        img,
        f"identity {result.claim.identity_id if result.claim.identity_id is not None else '-'}"
        f"   confidence {result.claim.confidence:.2f}",
        (16, 74),
        (235, 235, 235),
        0.48,
    )
    _confidence_bar(img, 16, 82, 260, 10, result.claim.confidence, color)
    _put(img, result.claim.reason[:58], (16, 108), (200, 200, 200), 0.42)
    cmd = result.command
    _put(
        img,
        f"pan {cmd.pan:+.2f}  tilt {cmd.tilt:+.2f}  "
        f"err ({cmd.error_x:+.0f},{cmd.error_y:+.0f})px",
        (16, 132),
        (200, 220, 255),
        0.42,
    )
    _put(
        img,
        f"bodies {len(result.bodies)}   faces {len(result.faces)}   "
        f"pairs {len(result.face_body_pairs)}",
        (16, 156),
        (200, 200, 200),
        0.42,
    )
    return img


def render_sim_frame(
    scene,
    sim_frame,
    result: FrameResult,
    show_detections: bool = True,
) -> np.ndarray:
    """Render a kinematic-scene frame with ground truth and pipeline state."""
    cv2 = _cv2()
    w, h = sim_frame.frame_size
    img = np.full((h, w, 3), 24, dtype=np.uint8)

    # Ground truth: solid dark boxes with the gt id.
    for gt_id, actor in sim_frame.ground_truth.items():
        if not actor.in_frame:
            continue
        gt_color = (80, 80, 80)
        _rect(img, actor.body_box, gt_color, 1)
        if actor.face_box is not None and actor.face_visible:
            _rect(img, actor.face_box, (110, 110, 60), 1)
        _put(
            img,
            f"gt{gt_id}{'' if actor.face_visible else ' (no face)'}",
            (int(actor.body_box.x1) + 3, int(actor.body_box.y2) - 6),
            (140, 140, 140),
            0.38,
        )

    # Detections.
    if show_detections:
        for det in sim_frame.detections.persons:
            _rect(img, det.box, (90, 60, 40), 1)
        for det in sim_frame.detections.faces:
            _rect(img, det.box, (140, 90, 40), 1)

    color = STATE_COLORS[result.claim.state]
    for body in result.bodies:
        is_target = body.track_id == result.claim.body_track_id
        _rect(img, body.box, color if is_target else (110, 110, 110), 2 if is_target else 1)
        if is_target:
            _rect(img, body.head_box, (90, 210, 240), 1)
    for face in result.faces:
        is_target = face.track_id == result.claim.face_track_id
        _rect(img, face.box, (255, 220, 60) if is_target else (120, 120, 120), 2 if is_target else 1)

    # HUD.
    _put(img, scene.name, (10, 22), (220, 220, 220), 0.6, 2)
    _put(
        img,
        f"t={sim_frame.timestamp:5.2f}s  {STATE_LABELS[result.claim.state]}"
        f"  conf={result.claim.confidence:.2f}",
        (10, 46),
        color,
        0.52,
    )
    _confidence_bar(img, 10, 54, 220, 9, result.claim.confidence, color)
    _put(img, result.claim.reason[:58], (10, 82), (200, 200, 200), 0.4)
    _put(
        img,
        f"pan {result.command.pan:+.2f} tilt {result.command.tilt:+.2f} "
        f"active={result.command.active}",
        (10, h - 14),
        (200, 220, 255),
        0.42,
    )
    return img


def contact_sheet(images: list[np.ndarray], cols: int = 3, scale: float = 0.5) -> np.ndarray:
    """Tile images into one sheet (used for the report figures)."""
    cv2 = _cv2()
    if not images:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    resized = [
        cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for im in images
    ]
    h, w = resized[0].shape[:2]
    rows = (len(resized) + cols - 1) // cols
    sheet = np.full((rows * h, cols * w, 3), 18, dtype=np.uint8)
    for idx, im in enumerate(resized):
        r, c = divmod(idx, cols)
        sheet[r * h : r * h + im.shape[0], c * w : c * w + im.shape[1]] = im
    return sheet
