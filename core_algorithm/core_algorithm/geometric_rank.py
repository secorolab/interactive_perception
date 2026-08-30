"""Deterministic generic rank of resolved planar-polygon constraints."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


DEFAULT_RANK_TOLERANCE = 1e-9
_DEGENERACY_TOLERANCE = 1e-10
_GENERIC_SEEDS = (1729, 3253, 6421, 9013, 12011)


def _generic_polygon(n_sides: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spacing = 2.0 * math.pi / n_sides
    jitter = min(0.12, 0.18 * spacing)
    angles = spacing * np.arange(n_sides) + rng.uniform(-jitter, jitter, n_sides)
    radii = rng.uniform(0.78, 1.24, n_sides)
    return np.column_stack((
        1.13 * radii * np.cos(angles) + 0.09 * np.sin(2.0 * angles),
        0.87 * radii * np.sin(angles) + 0.07 * np.cos(3.0 * angles),
    ))


def _validate_polygon(points: np.ndarray) -> None:
    edges = np.roll(points, -1, axis=0) - points
    if np.any(np.linalg.norm(edges, axis=1) <= _DEGENERACY_TOLERANCE):
        raise ValueError("constraint rank is undefined for a zero-length edge")


def _turn_angle(points: np.ndarray, index: int) -> float:
    incoming = points[index] - points[index - 1]
    outgoing = points[(index + 1) % len(points)] - points[index]
    return math.atan2(
        float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0]),
        float(np.dot(incoming, outgoing)),
    )


def _constraint_jacobian(
    points: np.ndarray,
    corner_known: Sequence[bool],
    direction_known: Sequence[bool],
    length_known: Sequence[bool],
    angle_known: Sequence[bool],
    edge_point_counts: Sequence[int],
) -> np.ndarray:
    """Build all active rows at one consistent generic realization."""

    _validate_polygon(points)
    n_sides = len(points)
    rows: list[np.ndarray] = []

    for index, known in enumerate(corner_known):
        if not known:
            continue
        row_x = np.zeros(2 * n_sides)
        row_y = np.zeros(2 * n_sides)
        row_x[2 * index] = 1.0
        row_y[2 * index + 1] = 1.0
        rows.extend((row_x, row_y))

    for index, known in enumerate(direction_known):
        if not known:
            continue
        next_index = (index + 1) % n_sides
        edge = points[next_index] - points[index]
        direction = edge / np.linalg.norm(edge)
        grad_edge = np.array((direction[1], -direction[0]))
        row = np.zeros(2 * n_sides)
        row[2 * index:2 * index + 2] = -grad_edge
        row[2 * next_index:2 * next_index + 2] = grad_edge
        rows.append(row)

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
        grad_incoming = cosine * np.array((outgoing[1], -outgoing[0])) - sine * outgoing
        grad_outgoing = cosine * np.array((-incoming[1], incoming[0])) - sine * incoming
        row = np.zeros(2 * n_sides)
        row[2 * previous_index:2 * previous_index + 2] = -grad_incoming
        row[2 * index:2 * index + 2] = grad_incoming - grad_outgoing
        row[2 * next_index:2 * next_index + 2] = grad_outgoing
        rows.append(row)

    for index, count in enumerate(edge_point_counts):
        next_index = (index + 1) % n_sides
        edge = points[next_index] - points[index]
        independent_count = min(int(count), 2)
        for point_index in range(independent_count):
            fraction = (point_index + 1) / (independent_count + 1)
            offset = fraction * edge
            grad_edge = np.array((offset[1], -offset[0]))
            grad_offset = np.array((-edge[1], edge[0]))
            row = np.zeros(2 * n_sides)
            row[2 * index:2 * index + 2] = -grad_edge - grad_offset
            row[2 * next_index:2 * next_index + 2] = grad_edge
            rows.append(row)

    if not rows:
        return np.empty((0, 2 * n_sides))
    jacobian = np.vstack(rows)
    norms = np.linalg.norm(jacobian, axis=1)
    if np.any(norms <= _DEGENERACY_TOLERANCE):
        raise ValueError("degenerate continuous constraint has a zero Jacobian row")
    return jacobian / norms[:, None]


def _relative_rank(jacobian: np.ndarray, relative_tolerance: float) -> int:
    if jacobian.size == 0:
        return 0
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return int(np.count_nonzero(singular_values > relative_tolerance * singular_values[0]))


def generic_joint_constraint_rank(
    n_sides: int,
    corner_known: Sequence[bool],
    direction_known: Sequence[bool],
    length_known: Sequence[bool],
    angle_known: Sequence[bool],
    edge_point_counts: Sequence[int] | None = None,
    *,
    relative_tolerance: float = DEFAULT_RANK_TOLERANCE,
) -> int:
    """Return the maximum joint rank over fixed nondegenerate polygons."""

    if n_sides < 3:
        raise ValueError("a polygon requires at least three sides")
    if any(len(mask) != n_sides for mask in (
        corner_known, direction_known, length_known, angle_known
    )):
        raise ValueError("constraint masks must contain one entry per edge/vertex")
    if edge_point_counts is None:
        edge_point_counts = [0] * n_sides
    if len(edge_point_counts) != n_sides or any(
        not isinstance(count, (int, np.integer)) or count < 0
        for count in edge_point_counts
    ):
        raise ValueError("edge point counts must be nonnegative integers, one per edge")
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative rank tolerance must lie in (0, 1)")

    ranks = (
        _relative_rank(
            _constraint_jacobian(
                _generic_polygon(n_sides, seed),
                corner_known,
                direction_known,
                length_known,
                angle_known,
                edge_point_counts,
            ),
            relative_tolerance,
        )
        for seed in _GENERIC_SEEDS
    )
    return min(2 * n_sides, max(ranks, default=0))


def _validate_resolved_values(knowledge) -> None:
    n_sides = knowledge.n_sides
    fields = (
        "slopes", "lengths", "edge_unit_vectors", "corners",
        "corner_angles", "internal_points_on_edge",
    )
    if any(len(getattr(knowledge, field)) != n_sides for field in fields):
        raise ValueError("knowledge arrays must agree with n_sides")

    for index, corner in enumerate(knowledge.corners):
        if corner is not None:
            value = np.asarray(corner, dtype=float)
            if value.shape != (2,) or not np.all(np.isfinite(value)):
                raise ValueError(f"corner {index} must contain two finite coordinates")
    for index, length in enumerate(knowledge.lengths):
        if length is not None and (
            not math.isfinite(float(length)) or float(length) <= 0.0
        ):
            raise ValueError(f"edge length {index} must be finite and positive")
    for index, angle in enumerate(knowledge.corner_angles):
        if angle is None:
            continue
        value = float(angle)
        if not math.isfinite(value) or not 0.0 < value < 360.0:
            raise ValueError(f"corner angle {index} must lie in (0, 360) degrees")
        if math.isclose(value, 180.0, abs_tol=1e-9):
            raise ValueError(f"corner angle {index} is degenerate at 180 degrees")
    for index, vector in enumerate(knowledge.edge_unit_vectors):
        if vector is not None:
            value = np.asarray(vector, dtype=float)
            if (
                value.shape != (2,)
                or not np.all(np.isfinite(value))
                or np.linalg.norm(value) <= _DEGENERACY_TOLERANCE
            ):
                raise ValueError(f"edge direction {index} must be finite and nonzero")
    for index, slope in enumerate(knowledge.slopes):
        if slope is not None and math.isnan(float(slope)):
            raise ValueError(f"edge slope {index} must not be NaN")
    for edge, points in enumerate(knowledge.internal_points_on_edge):
        for point in points:
            value = np.asarray(point, dtype=float)
            if value.shape != (2,) or not np.all(np.isfinite(value)):
                raise ValueError(f"edge point on edge {edge} must contain two finite coordinates")


def _unique_edge_point_count(points, tolerance: float = 1e-9) -> int:
    unique: list[np.ndarray] = []
    for point in points:
        value = np.asarray(point, dtype=float)
        if not any(np.linalg.norm(value - existing) <= tolerance for existing in unique):
            unique.append(value)
            if len(unique) == 2:
                return 2
    return len(unique)


def edge_point_constraint_counts(knowledge) -> tuple[int, ...]:
    """Return distinct committed edge-point counts, capped at two per edge."""

    return tuple(_unique_edge_point_count(points) for points in knowledge.internal_points_on_edge)


def geometric_joint_constraint_rank(
    knowledge,
    *,
    relative_tolerance: float = DEFAULT_RANK_TOLERANCE,
) -> int:
    """Return generic joint rank of all resolved continuous attributes."""

    _validate_resolved_values(knowledge)
    n_sides = knowledge.n_sides
    return generic_joint_constraint_rank(
        n_sides,
        [value is not None for value in knowledge.corners],
        [
            knowledge.edge_unit_vectors[index] is not None
            or knowledge.slopes[index] is not None
            for index in range(n_sides)
        ],
        [value is not None for value in knowledge.lengths],
        [value is not None for value in knowledge.corner_angles],
        edge_point_constraint_counts(knowledge),
        relative_tolerance=relative_tolerance,
    )


# Compatibility wrappers for callers that still request intrinsic-only rank.
def generic_geometric_constraint_rank(
    n_sides: int,
    length_known: Sequence[bool],
    angle_known: Sequence[bool],
    *,
    relative_tolerance: float = DEFAULT_RANK_TOLERANCE,
) -> int:
    return generic_joint_constraint_rank(
        n_sides,
        [False] * n_sides,
        [False] * n_sides,
        length_known,
        angle_known,
        relative_tolerance=relative_tolerance,
    )


def geometric_constraint_rank(
    knowledge,
    *,
    relative_tolerance: float = DEFAULT_RANK_TOLERANCE,
) -> int:
    return geometric_joint_constraint_rank(
        knowledge,
        relative_tolerance=relative_tolerance,
    )
