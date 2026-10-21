"""Optimal assignment (Hungarian / Jonker-Volgenant) with a numpy fallback.

``scipy`` is used when available; otherwise a compact O(n^3) implementation
of the shortest-augmenting-path algorithm (the same algorithm scipy uses)
takes over.  The fallback is unit-tested against brute force so the package
never silently degrades to greedy matching.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:  # pragma: no cover - exercised only when scipy is installed
    from scipy.optimize import linear_sum_assignment as _scipy_lsa

    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _scipy_lsa = None
    HAVE_SCIPY = False


def _lsa_numpy(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-cost assignment for a rectangular matrix (pure numpy)."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost matrix must be 2-D")
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    transposed = n > m
    if transposed:
        cost = cost.T
        n, m = m, n

    # 1-indexed potentials / matching (JV shortest augmenting paths).
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)   # p[j] = row matched to column j (1-based)
    way = np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    col_of_row = np.full(n, -1, dtype=int)
    for j in range(1, m + 1):
        if p[j] != 0:
            col_of_row[p[j] - 1] = j - 1
    rows = np.arange(n, dtype=int)
    ok = col_of_row >= 0
    rows, cols = rows[ok], col_of_row[ok]

    if transposed:
        # Solving on the transpose swapped the roles of rows and columns.
        return cols, rows
    return rows, cols


def linear_sum_assignment(cost: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Return row/column indices minimising the total assignment cost."""
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    if not np.all(np.isfinite(matrix)):
        matrix = np.where(np.isfinite(matrix), matrix, 1e9)
    if HAVE_SCIPY:
        rows, cols = _scipy_lsa(matrix)
        return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)
    return _lsa_numpy(matrix)


def match_scores(
    scores: np.ndarray,
    threshold: float,
    reject_cost: float | None = None,
) -> list[tuple[int, int, float]]:
    """Assign rows to columns maximising score, refusing weak pairs.

    The cost matrix is padded with one dummy column per row so that *not*
    matching is always an option; a pair is only returned when its score is
    strictly above ``threshold``.  This is what keeps the pipeline honest:
    an unconvincing face is never forced onto a body.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n, m = scores.shape
    if n == 0 or m == 0:
        return []

    if reject_cost is None:
        reject_cost = 1.0 - threshold
    big = 1e6
    cost = np.full((n, m + n), big, dtype=np.float64)
    cost[:, :m] = 1.0 - np.clip(scores, 0.0, 1.0)
    # Dummy column i can absorb row i at the reject cost.
    for i in range(n):
        cost[i, m + i] = reject_cost

    rows, cols = linear_sum_assignment(cost)
    matches: list[tuple[int, int, float]] = []
    for r, c in zip(rows, cols):
        if c < m:
            s = float(scores[r, c])
            if s > threshold:
                matches.append((int(r), int(c), s))
    matches.sort(key=lambda t: t[2], reverse=True)
    return matches


def unmatched(n: int, used: Sequence[int]) -> list[int]:
    used_set = set(int(u) for u in used)
    return [i for i in range(n) if i not in used_set]
