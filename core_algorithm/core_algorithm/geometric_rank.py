"""Exact generic rank of resolved planar-polygon constraints."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix


@dataclass(frozen=True)
class RankCertificate:
    rank: int
    lower_bound: int
    upper_bound: int
    proven: bool
    method: str


def _bool_tuple(values: Sequence[bool], n_sides: int, name: str) -> tuple[bool, ...]:
    if len(values) != n_sides:
        raise ValueError(f"{name} must contain one entry per edge/vertex")
    return tuple(bool(value) for value in values)


def _point_count_tuple(
    values: Sequence[int] | None,
    n_sides: int,
) -> tuple[int, ...]:
    if values is None:
        return (0,) * n_sides
    if len(values) != n_sides or any(
        not isinstance(value, (int, np.integer)) or value < 0 for value in values
    ):
        raise ValueError("edge point counts must be nonnegative integers, one per edge")
    return tuple(min(2, int(value)) for value in values)


def _polynomial_jacobian(
    n_sides: int,
    corner_known: tuple[bool, ...],
    direction_known: tuple[bool, ...],
    length_known: tuple[bool, ...],
    angle_known: tuple[bool, ...],
    point_counts: tuple[int, ...],
) -> tuple[sp.Matrix, tuple[sp.Symbol, ...]]:
    """Build the joint Jacobian after polynomial row scaling."""

    xs = sp.symbols(f"x0:{n_sides}")
    ys = sp.symbols(f"y0:{n_sides}")
    rows: list[list[sp.Expr]] = []
    point_symbols: list[sp.Symbol] = []

    def empty_row() -> list[sp.Expr]:
        return [sp.Integer(0)] * (2 * n_sides)

    for index, known in enumerate(corner_known):
        if not known:
            continue
        row_x = empty_row()
        row_y = empty_row()
        row_x[2 * index] = sp.Integer(1)
        row_y[2 * index + 1] = sp.Integer(1)
        rows.extend((row_x, row_y))

    for index, known in enumerate(direction_known):
        if not known:
            continue
        next_index = (index + 1) % n_sides
        edge_x = xs[next_index] - xs[index]
        edge_y = ys[next_index] - ys[index]
        row = empty_row()
        row[2 * index] = -edge_y
        row[2 * index + 1] = edge_x
        row[2 * next_index] = edge_y
        row[2 * next_index + 1] = -edge_x
        rows.append(row)

    for index, known in enumerate(length_known):
        if not known:
            continue
        next_index = (index + 1) % n_sides
        edge_x = xs[next_index] - xs[index]
        edge_y = ys[next_index] - ys[index]
        row = empty_row()
        row[2 * index] = -2 * edge_x
        row[2 * index + 1] = -2 * edge_y
        row[2 * next_index] = 2 * edge_x
        row[2 * next_index + 1] = 2 * edge_y
        rows.append(row)

    # Scaling each turning-angle derivative by the nonzero squared lengths of
    # its adjacent edges removes denominators without changing generic rank.
    for index, known in enumerate(angle_known):
        if not known:
            continue
        previous_index = (index - 1) % n_sides
        next_index = (index + 1) % n_sides
        incoming_x = xs[index] - xs[previous_index]
        incoming_y = ys[index] - ys[previous_index]
        outgoing_x = xs[next_index] - xs[index]
        outgoing_y = ys[next_index] - ys[index]
        incoming_norm_sq = incoming_x**2 + incoming_y**2
        outgoing_norm_sq = outgoing_x**2 + outgoing_y**2
        row = empty_row()
        row[2 * previous_index] = -incoming_y * outgoing_norm_sq
        row[2 * previous_index + 1] = incoming_x * outgoing_norm_sq
        row[2 * index] = (
            incoming_y * outgoing_norm_sq + outgoing_y * incoming_norm_sq
        )
        row[2 * index + 1] = (
            -incoming_x * outgoing_norm_sq - outgoing_x * incoming_norm_sq
        )
        row[2 * next_index] = -outgoing_y * incoming_norm_sq
        row[2 * next_index + 1] = outgoing_x * incoming_norm_sq
        rows.append(row)

    # A fixed point q on edge i gives cross(v_{i+1}-v_i, q-v_i)=0. At a
    # feasible generic realization q=v_i+t(v_{i+1}-v_i); at most two distinct
    # point incidences on one edge are independent.
    for index, count in enumerate(point_counts):
        next_index = (index + 1) % n_sides
        edge_x = xs[next_index] - xs[index]
        edge_y = ys[next_index] - ys[index]
        for point_index in range(count):
            fraction = sp.Symbol(f"t_{index}_{point_index}")
            point_symbols.append(fraction)
            row = empty_row()
            row[2 * index] = (1 - fraction) * edge_y
            row[2 * index + 1] = -(1 - fraction) * edge_x
            row[2 * next_index] = fraction * edge_y
            row[2 * next_index + 1] = -fraction * edge_x
            rows.append(row)

    matrix = sp.Matrix(rows) if rows else sp.zeros(0, 2 * n_sides)
    return matrix, tuple(xs) + tuple(ys) + tuple(point_symbols)


def _remove_implied_constraints(
    n_sides: int,
    corner_known: tuple[bool, ...],
    direction_known: tuple[bool, ...],
    length_known: tuple[bool, ...],
    angle_known: tuple[bool, ...],
    point_counts: tuple[int, ...],
) -> tuple[tuple[bool, ...], tuple[bool, ...], tuple[bool, ...], tuple[int, ...]]:
    """Drop rows that are provably in the span of other active rows."""

    directions: list[bool] = []
    lengths: list[bool] = []
    angles: list[bool] = []
    points: list[int] = []
    for index in range(n_sides):
        previous_index = (index - 1) % n_sides
        next_index = (index + 1) % n_sides
        endpoints_fixed = corner_known[index] and corner_known[next_index]
        one_endpoint_fixed = corner_known[index] or corner_known[next_index]
        directions.append(
            direction_known[index]
            and not endpoints_fixed
            and not (point_counts[index] >= 2 and not one_endpoint_fixed)
        )
        lengths.append(length_known[index] and not endpoints_fixed)
        angles.append(
            angle_known[index]
            and not (
                corner_known[previous_index]
                and corner_known[index]
                and corner_known[next_index]
            )
            and not (direction_known[previous_index] and direction_known[index])
        )
        points.append(
            0
            if endpoints_fixed or (one_endpoint_fixed and direction_known[index])
            else min(point_counts[index], 1)
            if one_endpoint_fixed
            else point_counts[index]
        )
    return tuple(directions), tuple(lengths), tuple(angles), tuple(points)


def _witness_substitutions(
    symbols: Sequence[sp.Symbol],
    n_sides: int,
    witness_index: int,
) -> dict[sp.Symbol, sp.Rational]:
    substitutions: dict[sp.Symbol, sp.Rational] = {}
    for index in range(n_sides):
        substitutions[sp.Symbol(f"x{index}")] = sp.Integer(
            (index + 1) * (witness_index + 2) + (index * index + 3) % 7
        )
        substitutions[sp.Symbol(f"y{index}")] = sp.Integer(
            (index + 2) * (index + witness_index + 3)
            + ((2 * index + witness_index + 1) ** 2 % 11)
        )
    for symbol in symbols:
        name = str(symbol)
        if not name.startswith("t_"):
            continue
        local_index = int(name.rsplit("_", 1)[1])
        substitutions[symbol] = (
            sp.Rational(1 + witness_index % 2, 4)
            if local_index == 0
            else sp.Rational(3 - witness_index % 2, 4)
        )
    return substitutions


def _witness_lower_bound(
    matrix: sp.Matrix,
    symbols: Sequence[sp.Symbol],
    n_sides: int,
) -> int:
    best = 0
    for witness_index in range(5):
        evaluated = matrix.subs(
            _witness_substitutions(symbols, n_sides, witness_index)
        )
        best = max(best, int(evaluated.rank()))
        if best == min(matrix.rows, matrix.cols):
            break
    return best


def _structural_upper_bound(
    n_sides: int,
    corner_known: tuple[bool, ...],
    direction_known: tuple[bool, ...],
    length_known: tuple[bool, ...],
    angle_known: tuple[bool, ...],
    point_counts: tuple[int, ...],
) -> int:
    """Return a conservative upper bound from row count and gauge freedom."""

    scalar_rows = (
        2 * sum(corner_known)
        + sum(direction_known)
        + sum(length_known)
        + sum(angle_known)
        + sum(point_counts)
    )
    if all(angle_known):
        scalar_rows -= 1  # differential of the total turning sum
    upper = min(2 * n_sides, max(0, scalar_rows))

    corner_count = sum(corner_known)
    has_points = any(point_counts)
    has_direction = any(direction_known)
    has_length = any(length_known)
    if not has_points:
        if corner_count == 0:
            upper = min(upper, 2 * n_sides - 2)  # translation gauge
            if not has_direction:
                upper = min(
                    upper,
                    2 * n_sides - 3 if has_length else 2 * n_sides - 4,
                )
            elif not has_length:
                upper = min(upper, 2 * n_sides - 3)  # scale gauge
        elif corner_count == 1:
            if not has_direction:
                upper = min(
                    upper,
                    2 * n_sides - 1 if has_length else 2 * n_sides - 2,
                )
            elif not has_length:
                upper = min(upper, 2 * n_sides - 1)  # scale gauge
    return max(0, upper)


@lru_cache(maxsize=4096)
def _exact_rank_cached(
    n_sides: int,
    corner_known: tuple[bool, ...],
    direction_known: tuple[bool, ...],
    length_known: tuple[bool, ...],
    angle_known: tuple[bool, ...],
    point_counts: tuple[int, ...],
) -> RankCertificate:
    reduced_directions, reduced_lengths, reduced_angles, reduced_points = (
        _remove_implied_constraints(
            n_sides,
            corner_known,
            direction_known,
            length_known,
            angle_known,
            point_counts,
        )
    )
    matrix, symbols = _polynomial_jacobian(
        n_sides,
        corner_known,
        reduced_directions,
        reduced_lengths,
        reduced_angles,
        reduced_points,
    )
    lower = _witness_lower_bound(matrix, symbols, n_sides)
    upper = _structural_upper_bound(
        n_sides,
        corner_known,
        reduced_directions,
        reduced_lengths,
        reduced_angles,
        reduced_points,
    )
    if lower > upper:
        raise RuntimeError(
            f"internal rank-certificate error: lower bound {lower} exceeds {upper}"
        )
    if lower == upper:
        return RankCertificate(
            lower,
            lower,
            upper,
            True,
            "exact rational witness and structural upper bound",
        )

    if matrix.rows == 0:
        symbolic_rank = 0
    else:
        field = sp.QQ.frac_field(*symbols)
        symbolic_rank = int(
            DomainMatrix.from_Matrix(matrix).convert_to(field).rank()
        )
    if not lower <= symbolic_rank <= upper:
        raise RuntimeError("exact symbolic rank lies outside certified bounds")
    return RankCertificate(
        symbolic_rank,
        symbolic_rank,
        symbolic_rank,
        True,
        "exact rational-function rank",
    )


def exact_generic_joint_constraint_rank(
    n_sides: int,
    corner_known: Sequence[bool],
    direction_known: Sequence[bool],
    length_known: Sequence[bool],
    angle_known: Sequence[bool],
    edge_point_counts: Sequence[int] | None = None,
) -> RankCertificate:
    """Return a certificate for the exact generic joint rank."""

    if n_sides < 3:
        raise ValueError("a polygon requires at least three sides")
    return _exact_rank_cached(
        n_sides,
        _bool_tuple(corner_known, n_sides, "corner_known"),
        _bool_tuple(direction_known, n_sides, "direction_known"),
        _bool_tuple(length_known, n_sides, "length_known"),
        _bool_tuple(angle_known, n_sides, "angle_known"),
        _point_count_tuple(edge_point_counts, n_sides),
    )


def generic_joint_constraint_rank(
    n_sides: int,
    corner_known: Sequence[bool],
    direction_known: Sequence[bool],
    length_known: Sequence[bool],
    angle_known: Sequence[bool],
    edge_point_counts: Sequence[int] | None = None,
) -> int:
    """Compatibility API returning only the certified rank."""

    return exact_generic_joint_constraint_rank(
        n_sides,
        corner_known,
        direction_known,
        length_known,
        angle_known,
        edge_point_counts,
    ).rank


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
                or np.linalg.norm(value) <= 1e-12
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


def geometric_joint_constraint_rank(knowledge) -> int:
    """Return exact generic joint rank of all resolved continuous attributes."""

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
    )


def generic_geometric_constraint_rank(
    n_sides: int,
    length_known: Sequence[bool],
    angle_known: Sequence[bool],
) -> int:
    """Compatibility wrapper for intrinsic-only constraint rank."""

    return generic_joint_constraint_rank(
        n_sides,
        [False] * n_sides,
        [False] * n_sides,
        length_known,
        angle_known,
    )


def geometric_constraint_rank(knowledge) -> int:
    """Compatibility wrapper for the joint rank."""

    return geometric_joint_constraint_rank(knowledge)
