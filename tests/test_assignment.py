"""Tests for the assignment solver used by both trackers and the associator."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from bodyfollowing.utils.assignment import (
    _lsa_numpy,
    linear_sum_assignment,
    match_scores,
    unmatched,
)


def brute_force(cost: np.ndarray) -> float:
    """Optimal assignment cost by exhaustive search (small matrices only)."""
    n, m = cost.shape
    best = np.inf
    if n <= m:
        for cols in itertools.permutations(range(m), n):
            best = min(best, sum(cost[i, cols[i]] for i in range(n)))
    else:
        for rows in itertools.permutations(range(n), m):
            best = min(best, sum(cost[rows[j], j] for j in range(m)))
    return best


@pytest.mark.parametrize("shape", [(3, 3), (4, 2), (2, 5), (5, 5), (6, 3), (3, 6)])
def test_numpy_solver_matches_brute_force(shape):
    rng = np.random.default_rng(11)
    cost = rng.random(shape)
    rows, cols = _lsa_numpy(cost)
    assert len(rows) == min(shape)
    assert len(set(rows)) == len(rows)
    assert len(set(cols)) == len(cols)
    assert cost[rows, cols].sum() == pytest.approx(brute_force(cost))


def test_solver_handles_negative_and_empty():
    rows, cols = linear_sum_assignment([])
    assert len(rows) == 0 and len(cols) == 0
    cost = np.array([[-5.0, 1.0], [2.0, -3.0]])
    rows, cols = linear_sum_assignment(cost)
    assert cost[rows, cols].sum() == pytest.approx(-8.0)


def test_solver_prefers_obvious_diagonal():
    cost = np.array([[0.1, 5.0, 5.0], [5.0, 0.2, 5.0], [5.0, 5.0, 0.3]])
    rows, cols = _lsa_numpy(cost)
    assert list(zip(rows, cols)) == [(0, 0), (1, 1), (2, 2)]


# ------------------------------------------------------------------ matching
def test_match_scores_refuses_weak_pairs():
    scores = np.array([[0.9, 0.1], [0.2, 0.3]])
    matches = match_scores(scores, threshold=0.5)
    assert matches == [(0, 0, pytest.approx(0.9))]


def test_match_scores_is_one_to_one():
    scores = np.array([[0.9, 0.85], [0.88, 0.1]])
    matches = match_scores(scores, threshold=0.5)
    used_rows = [m[0] for m in matches]
    used_cols = [m[1] for m in matches]
    assert len(used_rows) == len(set(used_rows))
    assert len(used_cols) == len(set(used_cols))
    # Optimal: row0-row... the best total is (0,1) + (1,0).
    assert {(m[0], m[1]) for m in matches} == {(0, 1), (1, 0)}


def test_match_scores_can_return_nothing():
    scores = np.zeros((2, 2))
    assert match_scores(scores, threshold=0.5) == []


def test_match_scores_more_rows_than_columns():
    scores = np.array([[0.9], [0.8], [0.2]])
    matches = match_scores(scores, threshold=0.5)
    assert len(matches) == 1 and matches[0][0] == 0
    assert unmatched(3, [matches[0][0]]) == [1, 2]
