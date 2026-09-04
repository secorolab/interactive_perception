import logging
import math
from typing import Optional, Tuple

import numpy as np
import sympy as sp
from scipy.optimize import linprog

from .helper import (
    are_list_elements_uniform,
    best_fit_unit_vector,
    edge_unit_vector_and_atleast_one_point_known,
    get_angle_between_vector,
    get_rotated_edge_slope,
    is_close,
    line_intersection,
    point_in_list,
    pre_process_polygon_knowledge,
    rotate_vector_2d,
)
from .polygon_knowledge import PolygonKnowledge

logger = logging.getLogger(__name__)


def feasible_bounded_lengths(edge_unit_vectors: list,
                             seed: int = 47,
                             lower_bound: float = 0.5,
                             upper_bound: float = 3.0) -> list:
    """
    Find feasible edge lengths within given bounds that satisfy closure.
    """
    U = np.asarray(edge_unit_vectors, float)
    n = U.shape[0]

    rng = np.random.default_rng(seed)
    c = rng.standard_normal(n)

    res = linprog(
        c,
        A_eq=U.T,
        b_eq=np.zeros(2),
        bounds=[(lower_bound, upper_bound)] * n,
        method="highs",
    )
    if not res.success:
        raise ValueError("No feasible lengths within bounds.")
    return res.x.tolist()


def propagate_parameters(polygon_knowledge: PolygonKnowledge,
                         min_points_to_remove_outlers: int = 4,
                         inlier_distance_threshold: int = 0.02) -> bool:
    """
    Apply geometric inference rules until no rule changes the knowledge state.

    Returns True if any rule updated `polygon_knowledge`.
    """
    changed = True
    knowledge_updated = False
    know = polygon_knowledge
    num_sides = know.n_sides

    while changed:
        changed = False

        changed = bool(
            pre_process_polygon_knowledge(
                polygon_knowledge,
                min_points_to_remove_outlers,
                inlier_distance_threshold,
            )
        )

        if not changed:
            for i in range(num_sides):
                points_on_edge = know.get_all_points_on_edge(i)
                if len(points_on_edge) >= 2 and know.edge_unit_vectors[i] is None:
                    know.edge_unit_vectors[i] = best_fit_unit_vector(points_on_edge)
                    changed = True
                    print(f" => Computed edge_unit_vector of edge {i} as {know.edge_unit_vectors[i]} using two points on the edge")

        if not changed:
            for i in range(num_sides):
                if know.slopes[i] is None and know.edge_unit_vectors[i] is not None:
                    if abs(know.edge_unit_vectors[i][0]) < 1e-5:
                        slope = np.inf
                    else:
                        slope = know.edge_unit_vectors[i][1] / know.edge_unit_vectors[i][0]
                    know.slopes[i] = slope
                    changed = True
                    print(f" => Computed slope of edge {i} as {know.slopes[i]}")

        if not changed:
            for i in range(num_sides):
                prev_i = (i - 1) % num_sides
                if (
                    know.edge_unit_vectors[prev_i] is not None
                    and know.corner_angles[i] is not None
                    and know.edge_unit_vectors[i] is None
                ):
                    angle_between_edges_rad = np.pi - np.deg2rad(know.corner_angles[i])
                    know.edge_unit_vectors[i] = rotate_vector_2d(
                        know.edge_unit_vectors[prev_i],
                        angle_between_edges_rad,
                    )
                    changed = True
                    print(f" => Computed edge_unit_vector of edge {i} as {know.edge_unit_vectors[i]} using previous edge_unit_vector and corner {know.corner_angles[i]} angle at {i}")
                elif (
                    know.edge_unit_vectors[prev_i] is None
                    and know.corner_angles[i] is not None
                    and know.edge_unit_vectors[i] is not None
                ):
                    angle_between_edges_rad = -(np.pi - np.deg2rad(know.corner_angles[i]))
                    know.edge_unit_vectors[prev_i] = rotate_vector_2d(
                        know.edge_unit_vectors[i],
                        angle_between_edges_rad,
                    )
                    changed = True
                    print(f" => Computed edge_unit_vector of edge {prev_i} as {know.edge_unit_vectors[prev_i]} using edge_unit_vector and corner angle {know.corner_angles[i]} at {i}")

        if not changed:
            for i in range(num_sides):
                prev_edge_idx = (i - 1) % num_sides
                if know.slopes[prev_edge_idx] is None or know.slopes[i] is None or know.corners[i] is not None:
                    continue
                p1 = know.internal_points_on_edge[prev_edge_idx][0] if know.internal_points_on_edge[prev_edge_idx] else None
                p2 = know.internal_points_on_edge[i][0] if know.internal_points_on_edge[i] else None
                if p1 is None or p2 is None:
                    continue
                corner = line_intersection(p1, know.slopes[prev_edge_idx], p2, know.slopes[i])
                if corner is not None:
                    corner_tuple = tuple(float(x) for x in corner)
                    know.corners[i] = corner_tuple
                    changed = True
                    print(f" => Computed corner {i} as {corner_tuple} using slopes and points of edges {prev_edge_idx} and {i}")

        if not changed:
            for i in range(num_sides):
                prev_edge_idx = (i - 1) % num_sides
                corner_angle_deg = know.corner_angles[i]
                if corner_angle_deg is None:
                    continue
                corner_angle_rad = np.deg2rad(corner_angle_deg)
                if know.slopes[prev_edge_idx] is not None and know.slopes[i] is None:
                    angle_between_edges_rad = np.pi - corner_angle_rad
                    know.slopes[i] = get_rotated_edge_slope(know.slopes[prev_edge_idx], angle_between_edges_rad)
                    changed = True
                    print(f" => Computed slope of edge {i} as {know.slopes[i]} using corner angle at {i} and slope of previous edge {prev_edge_idx}")
                if know.slopes[prev_edge_idx] is None and know.slopes[i] is not None:
                    angle_between_edges_rad = -(np.pi - corner_angle_rad)
                    know.slopes[prev_edge_idx] = get_rotated_edge_slope(know.slopes[i], angle_between_edges_rad)
                    changed = True
                    print(f" => Computed slope of edge {prev_edge_idx} as {know.slopes[prev_edge_idx]} using corner angle at {i} and slope of edge {i}")

        if not changed:
            for i in range(num_sides):
                a, b = i, (i + 1) % num_sides
                if know.corners[a] is not None and know.corners[b] is not None and know.lengths[i] is None:
                    dx = know.corners[b][0] - know.corners[a][0]
                    dy = know.corners[b][1] - know.corners[a][1]
                    know.lengths[i] = math.sqrt(dx**2 + dy**2)
                    changed = True
                    print(f" => Computed length of edge {i} as {know.lengths[i]} using corners {a} and {b}")

        if not changed:
            known_angles = [a for a in know.corner_angles if a is not None]
            if len(known_angles) == num_sides - 1:
                missing_angle = (num_sides - 2) * 180 - sum(known_angles)
                for i in range(num_sides):
                    if know.corner_angles[i] is None:
                        know.corner_angles[i] = missing_angle
                        changed = True
                        print(f" => Computed missing corner angle at {i} as {missing_angle} using sum of known angles")

        if not changed:
            for i in range(num_sides):
                prev_edge_idx = (i - 1) % num_sides
                prev_edge_unit_vector = know.edge_unit_vectors[prev_edge_idx]
                edge_unit_vector = know.edge_unit_vectors[i]
                if (
                    prev_edge_unit_vector is not None
                    and edge_unit_vector is not None
                    and know.corner_angles[i] is None
                ):
                    theta = get_angle_between_vector(prev_edge_unit_vector, edge_unit_vector)
                    know.corner_angles[i] = 180 - theta
                    changed = True
                    print(f" => Computed corner angle at {i} as {know.corner_angles[i]} using edge vectors of edges {prev_edge_idx} and {i}")

        if not changed:
            for i in range(num_sides):
                if know.corner_angles[i] is not None and know.is_reflexive_angle[i] is None:
                    if know.corner_angles[i] > 180.0:
                        know.is_reflexive_angle[i] = True
                        changed = True
                        print(f" => Corner index {i} is used to fill out that it is reflexive ")
                    elif know.corner_angles[i] < 180.0:
                        know.is_reflexive_angle[i] = False
                        changed = True
                        print(f" => Corner index {i} is used to fill out that it is not reflexive ")

        if not changed:
            for i in range(num_sides):
                next_i = (i + 1) % num_sides
                prev_i = (i - 1) % num_sides

                edge_unit_vector = know.edge_unit_vectors[i]
                prev_edge_unit_vector = know.edge_unit_vectors[prev_i]
                edge_length = know.lengths[i]
                prev_edge_length = know.lengths[prev_i]
                corner = know.corners[i]

                if (
                    edge_unit_vector is not None
                    and edge_length is not None
                    and know.corners[next_i] is None
                    and corner is not None
                ):
                    dx = edge_length * edge_unit_vector[0]
                    dy = edge_length * edge_unit_vector[1]
                    know.corners[next_i] = corner[0] + dx, corner[1] + dy
                    changed = True
                    print(f" => Computed corner {next_i} as {know.corners[next_i]} using corner {i}, edge unit vector {edge_unit_vector}, and length {edge_length}")
                elif (
                    prev_edge_unit_vector is not None
                    and prev_edge_length is not None
                    and know.corners[prev_i] is None
                    and corner is not None
                ):
                    prev_edge_unit_vector = np.array(prev_edge_unit_vector)
                    dx = prev_edge_length * prev_edge_unit_vector[0]
                    dy = prev_edge_length * prev_edge_unit_vector[1]
                    know.corners[prev_i] = corner[0] - dx, corner[1] - dy
                    changed = True
                    print(f" => Computed corner {prev_i} as {know.corners[prev_i]} using corner {i}, previous edge unit vector {prev_edge_unit_vector}, and length {prev_edge_length}")

        if not changed:
            for i in range(num_sides):
                next_i = (i + 1) % num_sides
                prev_i = (i - 1) % num_sides
                if not (
                    know.lengths[i] is not None
                    and len(know.get_all_points_on_edge(i)) == 0
                    and know.edge_unit_vectors[i] is not None
                    and edge_unit_vector_and_atleast_one_point_known(know, prev_i)
                    and edge_unit_vector_and_atleast_one_point_known(know, next_i)
                ):
                    continue

                u_prev = sp.Matrix(know.edge_unit_vectors[prev_i])
                p_prev = sp.Matrix(know.get_all_points_on_edge(prev_i)[0])
                u_next = sp.Matrix(know.edge_unit_vectors[next_i])
                p_next = sp.Matrix(know.get_all_points_on_edge(next_i)[0])
                u_curr = sp.Matrix(know.edge_unit_vectors[i])
                length_curr = know.lengths[i]
                s, t = sp.symbols("s t", real=True)

                corner_start = p_prev + s * u_prev
                corner_end = p_next + t * u_next
                equation = corner_end - (corner_start + length_curr * u_curr)
                solution = sp.solve(equation, (s, t), dict=True)
                if not solution:
                    logger.warning(
                        "Could not solve corner constraints for edge %s; skipping this propagation rule.",
                        i,
                    )
                    continue

                corner_start_sol = corner_start.subs(solution[0]).evalf()
                corner_end_sol = corner_end.subs(solution[0]).evalf()
                corner_start_tuple = tuple(float(v) for v in corner_start_sol)
                corner_end_tuple = tuple(float(v) for v in corner_end_sol)

                know.corners[i] = corner_start_tuple
                know.corners[next_i] = corner_end_tuple
                changed = True
                print(f" => Computed corners {corner_start_tuple} and {corner_end_tuple} at index {i} using adjacent edge constraints")

        if changed:
            knowledge_updated = True

    return knowledge_updated


def find_unique_pattern(polygon_knowledge: PolygonKnowledge,
                        tol: float = 1e-9) -> bool:
    """
    Check whether coupled feature sequences are unique under cyclic rotation.
    """
    pk = polygon_knowledge
    sequences = [pk.dihedrals, pk.corner_angles, pk.slopes, pk.lengths]
    n = len(sequences[0])

    def rotation_preserves(seq, k):
        for i in range(n):
            a = seq[i]
            b = seq[(i + k) % n]
            if a is not None and b is not None and not is_close(a, b, tol=tol):
                return False
        return True

    per_sequence_rotations = []
    for idx, seq in enumerate(sequences):
        preserved = {k for k in range(n) if rotation_preserves(seq, k)}
        per_sequence_rotations.append(preserved)

        if preserved == {0}:
            logger.info("Sequence %d is individually unique.", idx)
        else:
            logger.info("Sequence %d is not individually unique; preserved rotations = %s.", idx, sorted(preserved))

    joint_rotations = set.intersection(*per_sequence_rotations)
    if joint_rotations == {0}:
        logger.info("Only the identity rotation survives across all sequences.")
        return True

    logger.info("Non-identity rotations %s preserve all sequences.", sorted(joint_rotations - {0}))
    return False


def is_cyclically_unique(field, tol):
    """
    Check if a complete sequence is unique under cyclic rotations.
    """
    n = len(field)

    def is_same(a, b):
        for x, y in zip(a, b):
            if x is None and y is None:
                continue
            if x is None or y is None:
                return False
            if not is_close(x, y, tol=tol):
                return False
        return True

    for shift in range(1, n):
        rotated = field[shift:] + field[:shift]
        if is_same(field, rotated):
            return False

    return True


def get_unique_pattern_ref_index(current_knowledge: PolygonKnowledge,
                                 prior_knowledge: PolygonKnowledge,
                                 match_corner_coordinates: bool = False,
                                 find_match_in_individual_parameters: bool = False,
                                 tol: float = 0.01) -> Tuple[bool, Optional[int]]:
    """
    Find a unique rotational alignment from current knowledge to prior knowledge.
    """
    rck = current_knowledge
    rpk = prior_knowledge
    n_sides = rck.n_sides

    if find_match_in_individual_parameters:
        fields_to_check = [
            "slopes",
            "edge_unit_vectors",
            "lengths",
            "corners",
            "corner_angles",
            "dihedrals",
        ]

        def matches(curr_knw_rotated, prior_knw_field):
            for a, b in zip(curr_knw_rotated, prior_knw_field):
                if a is not None and b is None:
                    return False
                if a is not None and not is_close(a, b, tol=tol):
                    return False
            return True

        for field in fields_to_check:
            rpk_field = getattr(rpk, field)
            rck_field = getattr(rck, field)
            if not all(v is not None for v in rpk_field):
                continue
            if not is_cyclically_unique(rpk_field, tol):
                continue

            count = 0
            shift_idx_rck = None
            for shift in range(n_sides):
                rotated = rck_field[shift:] + rck_field[:shift]
                if matches(rotated, rpk_field):
                    count += 1
                    shift_idx_rck = shift
            if count == 1:
                return True, shift_idx_rck
        logger.info("No unique match found in individual parameters.")
        return False, None

    if match_corner_coordinates:
        for c in range(n_sides):
            for p in range(n_sides):
                rck_corner_c = rck.corners[c]
                rpk_corner_p = rpk.corners[p]
                if rck_corner_c is not None and rpk_corner_p is not None and is_close(rck_corner_c, rpk_corner_p, tol=tol):
                    print("Found matching corner coordinates at rck index", c, "and rpk index", p)
                    return True, (c - p) % n_sides
        return False, None

    rp_sequences = [rpk.dihedrals, rpk.corner_angles, rpk.slopes, rpk.lengths]
    rc_sequences = [rck.dihedrals, rck.corner_angles, rck.slopes, rck.lengths]
    n = len(rp_sequences[0])
    valid_rotations = []

    for k in range(n):
        pattern_found = True
        for rp, rc in zip(rp_sequences, rc_sequences):
            for i in range(n):
                a = rp[i]
                b = rc[(i + k) % n]
                if a is not None and b is not None and not is_close(a, b, tol=tol):
                    pattern_found = False
                    break
            if not pattern_found:
                break
        if pattern_found:
            valid_rotations.append(k)

    if len(valid_rotations) == 0:
        raise ValueError("No valid rotation: polygons do not match.")
    if len(valid_rotations) > 1:
        raise ValueError(f"Ambiguous mapping: multiple rotations possible {valid_rotations}")

    return True, valid_rotations[0]


def rearrange_rck_using_prior_knowledge(rck: PolygonKnowledge,
                                        rpk_first_idx_in_rck: int) -> None:
    """
    Rotate current knowledge arrays so the matched prior index becomes edge 0.
    """
    rck.slopes[:] = rck.slopes[rpk_first_idx_in_rck:] + rck.slopes[:rpk_first_idx_in_rck]
    rck.lengths[:] = rck.lengths[rpk_first_idx_in_rck:] + rck.lengths[:rpk_first_idx_in_rck]
    rck.edge_unit_vectors[:] = rck.edge_unit_vectors[rpk_first_idx_in_rck:] + rck.edge_unit_vectors[:rpk_first_idx_in_rck]
    rck.corners[:] = rck.corners[rpk_first_idx_in_rck:] + rck.corners[:rpk_first_idx_in_rck]
    rck.is_reflexive_angle[:] = rck.is_reflexive_angle[rpk_first_idx_in_rck:] + rck.is_reflexive_angle[:rpk_first_idx_in_rck]
    rck.corner_angles[:] = rck.corner_angles[rpk_first_idx_in_rck:] + rck.corner_angles[:rpk_first_idx_in_rck]
    rck.dihedrals[:] = rck.dihedrals[rpk_first_idx_in_rck:] + rck.dihedrals[:rpk_first_idx_in_rck]
    rck.internal_points_on_edge[:] = (
        rck.internal_points_on_edge[rpk_first_idx_in_rck:]
        + rck.internal_points_on_edge[:rpk_first_idx_in_rck]
    )
    print(f" => Rearranged rck using prior model where rck edge {rpk_first_idx_in_rck} is now edge 0.")


def _point_is_present(point: Tuple[float, float],
                      points: list[Tuple[float, float]]) -> bool:
    if not points:
        return False
    found, _ = point_in_list(point, points)
    return found


def fill_missing_parameters(rck: PolygonKnowledge,
                            rpk: PolygonKnowledge,
                            rpk_rck_matching_idx_found: bool) -> None:
    """
    Fill missing current-knowledge fields from prior knowledge where valid.
    """
    if rpk_rck_matching_idx_found:
        for i in range(rck.n_sides):
            if rck.edge_unit_vectors[i] is None and rpk.edge_unit_vectors[i] is not None:
                rck.edge_unit_vectors[i] = rpk.edge_unit_vectors[i]
                print(f" => Filled edge_unit_vectors of edge {i} as {rck.edge_unit_vectors[i]} using prior model")
            if rck.is_reflexive_angle[i] is None and rpk.is_reflexive_angle[i] is not None:
                rck.is_reflexive_angle[i] = rpk.is_reflexive_angle[i]
                print(f" => Filled is_reflexive_angle of edge {i} as {rck.is_reflexive_angle[i]} using prior model")
            if rck.dihedrals[i] is None and rpk.dihedrals[i] is not None:
                rck.dihedrals[i] = rpk.dihedrals[i]
                print(f" => Filled dihedral of edge {i} as {rck.dihedrals[i]} using prior model")
            if rck.corner_angles[i] is None and rpk.corner_angles[i] is not None:
                rck.corner_angles[i] = rpk.corner_angles[i]
                print(f" => Filled corner angle at {i} as {rck.corner_angles[i]} using prior model")
            if rck.lengths[i] is None and rpk.lengths[i] is not None:
                rck.lengths[i] = rpk.lengths[i]
                print(f" => Filled length of edge {i} as {rck.lengths[i]} using prior model")
            if rck.slopes[i] is None and rpk.slopes[i] is not None:
                rck.slopes[i] = rpk.slopes[i]
                print(f" => Filled slope of edge {i} as {rck.slopes[i]} using prior model")
            if rck.corners[i] is None and rpk.corners[i] is not None:
                rck.corners[i] = rpk.corners[i]
                print(f" => Filled corner {i} as {rck.corners[i]} using prior model")
            for point in rpk.internal_points_on_edge[i]:
                if not _point_is_present(point, rck.internal_points_on_edge[i]):
                    rck.internal_points_on_edge[i].append(point)
                    print(f" => Filled internal_points_on_edge {i} with {point} using prior model")
    else:
        if are_list_elements_uniform(rpk.is_reflexive_angle):
            for i in range(rck.n_sides):
                rck.is_reflexive_angle[i] = rpk.is_reflexive_angle[0]
            print(f" => Filled is_reflexive_angle of all edges as {rck.is_reflexive_angle[0]} using prior model")
        if are_list_elements_uniform(rpk.corner_angles):
            for i in range(rck.n_sides):
                rck.corner_angles[i] = rpk.corner_angles[0]
            print(f" => Filled corner_angles of all edges as {rck.corner_angles[0]} using prior model")
        if are_list_elements_uniform(rpk.lengths):
            for i in range(rck.n_sides):
                rck.lengths[i] = rpk.lengths[0]
            print(f" => Filled lengths of all edges as {rck.lengths[0]} using prior model")
        if are_list_elements_uniform(rpk.dihedrals):
            for i in range(rck.n_sides):
                rck.dihedrals[i] = rpk.dihedrals[0]
            print(f" => Filled dihedrals of all edges as {rck.dihedrals[0]} using prior model")
