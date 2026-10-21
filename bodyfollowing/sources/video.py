"""Frame sources: webcam, video file, and image sequence.

All sources yield ``(frame, timestamp)`` pairs.  The timestamp is *wall clock*
where possible so the identity model's exponential decay is exercised with
real timing rather than an idealised frame counter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class Frame:
    index: int
    timestamp: float
    image: np.ndarray


class VideoSource:
    """Reads frames from a file or a camera index."""

    def __init__(
        self,
        source: int | str | Path,
        width: int | None = None,
        height: int | None = None,
        real_time: bool = False,
        loop: bool = False,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.real_time = real_time
        self.loop = loop
        self._cap = None

    # ------------------------------------------------------------------ open/close
    def open(self) -> None:
        import cv2

        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video source: {self.source!r}")
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- properties
    @property
    def fps(self) -> float:
        import cv2

        if self._cap is None:
            return 0.0
        value = self._cap.get(cv2.CAP_PROP_FPS)
        return float(value) if value and value > 0 else 30.0

    @property
    def size(self) -> tuple[int, int]:
        import cv2

        if self._cap is None:
            return (0, 0)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    # -------------------------------------------------------------------- frames
    def frames(self, max_frames: int | None = None) -> Iterator[Frame]:
        if self._cap is None:
            self.open()
        assert self._cap is not None

        index = 0
        start = time.monotonic()
        frame_period = 1.0 / max(self.fps, 1e-6)

        while True:
            ok, image = self._cap.read()
            if not ok:
                if self.loop:
                    import cv2

                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, image = self._cap.read()
                if not ok:
                    break

            if self.real_time:
                # Present frames at the source rate so decay sees real gaps.
                target = start + index * frame_period
                delay = target - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                timestamp = time.monotonic() - start
            else:
                timestamp = index * frame_period

            yield Frame(index=index, timestamp=timestamp, image=image)
            index += 1
            if max_frames is not None and index >= max_frames:
                break


def list_cameras(max_index: int = 5) -> list[int]:
    """Probe for available camera indices (no hardware required to import)."""
    import cv2

    found: list[int] = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            found.append(idx)
        cap.release()
    return found
