"""Generic rank of intrinsic planar-polygon constraints."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


DEFAULT_RANK_TOLERANCE = 1e-9
_DEGENERACY_TOLERANCE = 1e-10
_GENERIC_SEEDS = (1729, 3253, 6421, 9013, 12011)


def _generic_polygon(n_sides: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = 2.0 * math.pi * np.arange(n_sides) / n_sides
    angles += rng.uniform(-0.12, 0.12, n_sides)
    radii = rng.uniform(0.78, 1.24, n_sides)
    return np.column_stack(
        (
            1.13 * radii * np.cos(angles) + 0.09 * np.sin(2.0 * angles),
            0.87 * radii * np.sin(angles) + 0.07 * np.cos(3.0 * angles),
        )
    )


def _validate_polygon(points: np.ndarray) -> None:
    edges = np.roll(points, -1, axis=0) - points
    if np.any(np.linalg.norm(edges, axis=1) <= _DEGENERACY_TOLERANCE):
        raise ValueError("constraint rank is undefined for a zero-length edge")


def _turn_angle(points: np.ndarray, index: int) -> float:
    previous = points[index] - points[index - 1]
    following = points[(index + 1) % len(points)] - points[index]
    return math.atan2(
        float(previous[0] * following[1] - previous[1] * following[0]),
        float(np.dot(previous, following)),
    )


def _constraint_jacobian(
    points: np.ndarray,
    length_known: Sequence[bool],
    angle_known: Sequence[bool],
) -> np.ndarray:
    """Build length and signed-angle Jacobian rows at one configuration."""

    _validate_polygon(points)
    n_sides = len(points)
    rows: list[np.ndarray] = []
    for index, known in enumerate(length_known):
        if not known:
            continue
        next_index = (index + 1) % n_sides
        edge = points[next_index] - points[index]
        row = np.zeros(2 * n_sides)
        row[2 * index:2 * index + 2] = -2.0 * edge
        row[2 * next_index:2 * next_index + 2] = 2.0 * edge
        rows.append(row)

    for index, known in enumerate(angle_known):
        if not known:
            continue
        previous_index = (index - 1) % n_sides
        next_index = (index + 1) % n_sides
        incoming = points[index] - points[previous_index]
        outgoing = points[next_index] - points[index]
        turn = _turn_angle(points, index)

        cosine = math.cos(turn)
        sine = math.sin(turn)
        cross_du = np.array((outgoing[1], -outgoing[0]))
        cross_dv = np.array((-incoming[1], incoming[0]))
        grad_incoming = cosine * cross_du - sine * outgoing
        grad_outgoing = cosine * cross_dv - sine * incoming

        row = np.zeros(2 * n_sides)
        row[2 * previous_index:2 * previous_index + 2] = -grad_incoming
        row[2 * index:2 * index + 2] = grad_incoming - grad_outgoing
        row[2 * next_index:2 * next_index + 2] = grad_outgoing
        rows.append(row)

    if not rows:
        return np.empty((0, 2 * n_sides))
    jacobian = np.vstack(rows)
    norms = np.linalg.norm(jacobian, axis=1)
    if np.any(norms <= _DEGENERACY_TOLERANCE):
        raise ValueError("degenerate intrinsic constraint has a zero Jacobian row")
    return jacobian / norms[:, None]


def _relative_rank(jacobian: np.ndarray, relative_tolerance: float) -> int:
    if jacobian.size == 0:
        return 0
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return int(np.count_nonzero(singular_values > relative_tolerance * singular_values[0]))


def generic_geometric_constraint_rank(
    n_sides: int,
    length_known: Sequence[bool],
    angle_known: Sequence[bool],
    *,
    relative_tolerance: float = DEFAULT_RANK_TOLERANCE,
) -> int:
    """Estimate generic joint rank from fixed nondegenerate configurations.

    Every Jacobian row is normalized before SVD.  A singular value is retained
    when it exceeds ``relative_tolerance`` times the largest singular value.
    The maximum over fixed generic polygons is deterministic and is capped by
    the intrinsic planar-polygon dimension ``2*n_sides - 3``.
    """

    if n_sides < 3:
        raise ValueError("a polygon requires at least three sides")
    if len(length_known) != n_sides or len(angle_known) != n_sides:
        raise ValueError("constraint masks must contain one entry per edge/vertex")
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative rank tolerance must lie in (0, 1)")

    ranks = (
        _relative_rank(
            _constraint_jacobian(
                _generic_polygon(n_sides, seed),
                length_known,
                angle_known,
            ),
            relative_tolerance,
        )
        for seed in _GENERIC_SEEDS
    )
    return min(2 * n_sides - 3, max(ranks, default=0))


def geometric_constraint_rank(
    knowledge,
    *,
    relative_tolerance: float = DEFAULT_RANK_TOLERANCE,
) -> int:
    """Return generic intrinsic rank for the currently known metric constraints."""

    length_known = [value is not None for value in knowledge.lengths]
    for index, value in enumerate(knowledge.lengths):
        if value is not None and (not math.isfinite(float(value)) or float(value) <= 0.0):
            raise ValueError(f"edge length {index} must be finite and positive")
    angle_known = [value is not None for value in knowledge.corner_angles]
    for index, value in enumerate(knowledge.corner_angles):
        if value is None:
            continue
        angle = float(value)
        if not math.isfinite(angle) or not 0.0 < angle < 360.0:
            raise ValueError(f"corner angle {index} must lie in (0, 360) degrees")
        if math.isclose(angle, 180.0, abs_tol=1e-9):
            raise ValueError(f"corner angle {index} is degenerate at 180 degrees")
    return generic_geometric_constraint_rank(
        knowledge.n_sides,
        length_known,
        angle_known,
        relative_tolerance=relative_tolerance,
    )
