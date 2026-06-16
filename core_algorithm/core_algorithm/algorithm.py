
from operator import gt
import numpy as np
import math
import copy
import logging
from typing import Tuple, Optional
from dataclasses import replace
import sympy as sp
from scipy.optimize import linprog
from .polygon_knowledge import PolygonKnowledge
from .helper import *
from .data_structures import *
import random

logger = logging.getLogger(__name__)


def feasible_bounded_lengths(edge_unit_vectors: list,
                             seed: int = 47,
                             lower_bound: float = 0.5,
                             upper_bound: float = 3.0) -> list:
    """
    Find feasible edge lengths within given bounds that satisfy closure condition

    :param edge_unit_vectors: List of edge unit vectors
    :param seed: Random seed for reproducibility
    :param lower_bound: Lower bound for edge lengths
    :param upper_bound: Upper bound for edge lengths
    :return: List of feasible edge lengths
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
    Apply basic inference rules until no change

    :param polygon_knowledge: Knowledge of the polygon
    :param min_points_to_remove_outlers: Minimum number of internal points to allow removing outliers
    :param inlier_distance_threshold:
    :return: None, Updates polygon_knowledge in place
    """

    changed = True
    know = polygon_knowledge
    num_sides = know.n_sides

    while changed:
        changed = False

        changed = bool(pre_process_polygon_knowledge(polygon_knowledge, min_points_to_remove_outlers, inlier_distance_threshold))

        # Rule: If atleast two points on an edge is known, and edge-unit-vector is not calculated, calculate it
        if not changed:
            # TODO: guard: if atleast 2 are unique and remove duplicates
            for i in range(num_sides):
                points_on_edge = know.get_all_points_on_edge(i)
                if len(points_on_edge) >= 2 and know.edge_unit_vectors[i] is None:
                    unit_vector = best_fit_unit_vector(points_on_edge)
                    know.edge_unit_vectors[i] = unit_vector
                    changed = True
                    print(f" => Computed edge_unit_vector of edge {i} as {know.edge_unit_vectors[i]} using two points on the edge")

        # Rule: If edge-unit-vector is known but slope is unknown, calculate slope
        if not changed:
            for i in range(num_sides):
                if know.slopes[i] is None and know.edge_unit_vectors[i] is not None:
                    if abs(know.edge_unit_vectors[i][0]) < 1e-5:
                        slope = np.inf
                    else:
                        slope = know.edge_unit_vectors[i][1]/know.edge_unit_vectors[i][0]
                    know.slopes[i] = slope
                    changed = True
                    print(f" => Computed slope of edge {i} as {know.slopes[i]}")

        # Rule: If previous edge-unit-vector and corner angle are known, find next edge-unit-vector
        if not changed:
            for i in range(num_sides):
                prev_i = (i - 1) % num_sides
                if (know.edge_unit_vectors[prev_i] is not None 
                    and know.corner_angles[i] is not None 
                    and know.edge_unit_vectors[i] is None):
                    # Compute the rotation angle: exterior angle = 180° - interior angle
                    angle_between_edges_rad = np.pi - np.deg2rad(know.corner_angles[i])
                    print(f" angle between edges -----> ", np.rad2deg(angle_between_edges_rad))
                    know.edge_unit_vectors[i] = rotate_vector_2d(know.edge_unit_vectors[prev_i], angle_between_edges_rad)
                    changed = True
                    print(f" => Computed edge_unit_vector of edge {i} as {know.edge_unit_vectors[i]} using previous edge_unit_vector and corner {know.corner_angles[i]} angle at {i}")

                elif (know.edge_unit_vectors[prev_i] is None 
                      and know.corner_angles[i] is not None 
                      and know.edge_unit_vectors[i] is not None):
                    # Compute the rotation angle: calculate in clockwise direction
                    angle_between_edges_rad = -(np.pi - np.deg2rad(know.corner_angles[i]))
                    print(f" angle between edges -----> ", np.rad2deg(angle_between_edges_rad))
                    know.edge_unit_vectors[prev_i] = rotate_vector_2d(know.edge_unit_vectors[i], angle_between_edges_rad)
                    changed = True
                    print(f" => Computed edge_unit_vector of edge {prev_i} as {know.edge_unit_vectors[prev_i]} using edge_unit_vector and corner angle {know.corner_angles[i]} at {i}")

        # Rule: If slopes of adjacent edges are known and a point each on them are known → compute shared corner
        if not changed:
            for i in range(num_sides):
                curr_edge_idx = i
                prev_edge_idx = (i - 1) % num_sides
                if know.slopes[prev_edge_idx] is not None and know.slopes[i] is not None and know.corners[i] is None:
                    p1, p2 = None, None
                    if len(know.internal_points_on_edge[prev_edge_idx]) > 0:
                        p1 = know.internal_points_on_edge[prev_edge_idx][0]
                    if len(know.internal_points_on_edge[curr_edge_idx]) > 0:
                        p2 = know.internal_points_on_edge[curr_edge_idx][0]
                    if p1 is not None and p2 is not None:
                        c = line_intersection(p1, know.slopes[prev_edge_idx], p2, know.slopes[curr_edge_idx])
                        if c is not None:
                            # Convert numpy array to tuple of floats to avoid boolean ambiguity errors later
                            c_tuple = tuple(float(x) for x in c)
                            know.corners[curr_edge_idx] = c_tuple
                            changed = True
                            print(f" => Computed corner {curr_edge_idx} as {c_tuple} using slopes and points of edges {prev_edge_idx} and {curr_edge_idx}")

        # Rule: If corner angle and one of adjacent slopes are known, compute next slope
        if not changed:
            for i in range(num_sides):
                prev_edge_idx = (i - 1) % num_sides
                corner_angle_deg = know.corner_angles[i]
                if corner_angle_deg is None:
                    continue
                else:
                    corner_angle_rad = np.deg2rad(corner_angle_deg)
                if (know.slopes[prev_edge_idx] is not None 
                    and know.slopes[i] is None):
                    angle_between_edges_rad = np.pi - corner_angle_rad
                    know.slopes[i] = get_rotated_edge_slope(know.slopes[prev_edge_idx], angle_between_edges_rad)
                    changed = True
                    print(f" => Computed slope of edge {i} as {know.slopes[i]} using corner angle at {i} and slope of previous edge {prev_edge_idx}")
                if (know.slopes[prev_edge_idx] is None 
                    and know.slopes[i] is not None):
                    angle_between_edges_rad = -(np.pi - corner_angle_rad)
                    know.slopes[prev_edge_idx] = get_rotated_edge_slope(know.slopes[i], angle_between_edges_rad)
                    changed = True
                    print(f" => Computed slope of edge {prev_edge_idx} as {know.slopes[prev_edge_idx]} using corner angle at {i} and slope of edge {i}")


        # Rule: If two adjacent corners are known but corresponding edge length is missing, compute it
        if not changed:
            for i in range(num_sides):
                a, b = i, (i + 1) % num_sides
                if know.corners[a] is not None and know.corners[b] is not None:
                    dx = know.corners[b][0] - know.corners[a][0]
                    dy = know.corners[b][1] - know.corners[a][1]
                    length = math.sqrt(dx**2 + dy**2)
                    if know.lengths[i] is None:
                        know.lengths[i] = length
                        changed = True
                        print(f" => Computed length of edge {i} as {know.lengths[i]} using corners {a} and {b}")

        # Rule: If only one corner angle is missing, compute it from other known angles
        if not changed:
            known_angles = [a for a in know.corner_angles if a is not None]
            if len(known_angles) == num_sides - 1:
                total_known = sum(known_angles)
                missing_angle = (num_sides - 2) * 180 - total_known
                for i in range(num_sides):
                    if know.corner_angles[i] is None:
                        know.corner_angles[i] = missing_angle
                        changed = True
                        print(f" => Computed missing corner angle at {i} as {missing_angle} using sum of known angles")

        # Rule: If two consecutive edge unit vectors are known, compute the corner angle between them
        if not changed:
            for i in range(num_sides):
                prev_edge_idx = (i - 1) % num_sides
                prev_edge_unit_vector = know.edge_unit_vectors[prev_edge_idx]
                edge_unit_vector = know.edge_unit_vectors[i]

                if (prev_edge_unit_vector is not None 
                    and edge_unit_vector is not None 
                    and know.corner_angles[i] is None):                            
                        theta = get_angle_between_vector(prev_edge_unit_vector, edge_unit_vector)
                        corner_angle = 180 - theta
                        know.corner_angles[i] = corner_angle
                        changed = True
                        print(f" => Computed corner angle at {i} as {corner_angle} using slopes of edges {prev_edge_idx} and {i}")

        # Rule: If corner angle is available and is_reflexive is not filled, complete it
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

        # Rule: If previous and current edge-unit-vector + current corner + current length are known, compute next corner
        if not changed:
            for i in range(num_sides):
                next_i = (i + 1) % num_sides
                prev_i = (i - 1) % num_sides

                edge_unit_vector = know.edge_unit_vectors[i]
                prev_edge_unit_vector = know.edge_unit_vectors[prev_i]

                edge_length = know.lengths[i]
                prev_edge_length = know.lengths[prev_i]

                corner = know.corners[i]
                prev_corner = know.corners[prev_i]
                next_corner = know.corners[next_i] # TODO: check if it will replace this in place. Checked, it replaces this in place but write proper comment

                # Case: Counterclockwise traversal
                if (edge_unit_vector is not None 
                    and edge_length is not None 
                    and next_corner is None 
                    and corner is not None):
                    dx = edge_length * edge_unit_vector[0]
                    dy = edge_length * edge_unit_vector[1]
                    know.corners[next_i] =  corner[0] + dx, corner[1] + dy
                    changed = True
                    print(f" => Computed corner {next_i} as {know.corners[next_i]} using corner {i}, and edge unit vector {edge_unit_vector} and length {edge_length}")

                # Case: Clockwise traversal
                elif (prev_edge_unit_vector is not None
                    and prev_edge_length is not None
                    and prev_corner is None
                    and corner is not None):
                    prev_edge_unit_vector = np.array(prev_edge_unit_vector)
                    dx = prev_edge_length * prev_edge_unit_vector[0]
                    dy = prev_edge_length * prev_edge_unit_vector[1]
                    know.corners[prev_i] =  corner[0] - dx, corner[1] - dy
                    changed = True
                    print(f" => Computed corner {prev_i} as {know.corners[prev_i]} using corner {i}, and previous edge unit vector {prev_edge_unit_vector} and length {prev_edge_length}")

        # Rule: If adjacent edge unit vectors of an edge and a point on each are known, only the edge length and unit vector are known for current edge, then solve for corners
        if not changed:
            for i in range(num_sides):
                next_i = (i + 1) % num_sides
                prev_i = (i - 1) % num_sides
                if (know.lengths[i] is not None and len(know.get_all_points_on_edge(i)) == 0 and know.edge_unit_vectors[i] is not None and
                    edge_unit_vector_and_atleast_one_point_known(know, prev_i) and edge_unit_vector_and_atleast_one_point_known(know, next_i)):
                    u_prev = sp.Matrix(know.edge_unit_vectors[prev_i])
                    P_prev = sp.Matrix(know.get_all_points_on_edge(prev_i)[0])

                    u_next = sp.Matrix(know.edge_unit_vectors[next_i])
                    P_next = sp.Matrix(know.get_all_points_on_edge(next_i)[0])

                    u_curr = sp.Matrix(know.edge_unit_vectors[i])
                    L_curr = know.lengths[i]

                    # Unknown parameters along the adjacent edges
                    s, t = sp.symbols('s t', real=True)

                    # Corners
                    C1 = P_prev + s * u_prev
                    C2 = P_next + t * u_next

                    # Constraint: C2 = C1 + L*u_curr
                    eq = C2 - (C1 + L_curr * u_curr)

                    # Solve
                    solution = sp.solve(eq, (s, t), dict=True)

                    C1_sol = C1.subs(solution[0])
                    C2_sol = C2.subs(solution[0])

                    C1 = tuple(float(v) for v in C1_sol)

                    corner_curr = C1_sol.evalf()
                    corner_next = C2_sol.evalf()
                    
                    corner_curr_tuple = tuple(float(v) for v in corner_curr)
                    corner_next_tuple = tuple(float(v) for v in corner_next)
                    
                    know.corners[i] = corner_curr_tuple
                    know.corners[next_i] = corner_next_tuple
                    changed = True
                    print(f" => Computed corners {corner_curr_tuple} and {corner_next_tuple} at index {i} using information of adjacent edge unit vectors, points on them, and current edge unit vector and its length")


def find_unique_pattern(polygon_knowledge: PolygonKnowledge,
                        tol: float = 1e-9) -> bool:
    """
    Check if multiple sequences are jointly unique under rotation, considering None as wildcard

    :param polygon_knowledge: Prior knowledge of the polygon
    :param tol: Tolerance for numerical comparison
    :param print_log: Whether to print detailed log
    :return is_unique: Whether the sequences are jointly unique
    """

    pk = polygon_knowledge
    sequences = [pk.dihedrals, pk.corner_angles, pk.slopes, pk.lengths] #TODO: add unit vector instead of slope
    n = len(sequences[0])

    def rotation_preserves(seq, k):
        for i in range(n):
            a = seq[i]
            b = seq[(i + k) % n]
            if a is not None and b is not None and not is_close(a, b, tol=tol):
                return False
        return True

    # find rotations that preserve each sequence
    per_sequence_rotations = []

    for idx, seq in enumerate(sequences):
        preserved = set()
        for k in range(n):
            if rotation_preserves(seq, k):
                preserved.add(k)
        per_sequence_rotations.append(preserved)

        if preserved == {0}:
            logger.info("Sequence %d is individually unique (only identity rotation preserves it).", idx)
        else:
            logger.info("Sequence %d is NOT individually unique; preserved rotations = %s.", idx, sorted(preserved))

    # find intersection of all preserved-rotation sets
    joint_rotations = set.intersection(*per_sequence_rotations)

    # explain result
    if joint_rotations == {0}:
        logger.info("Only the identity rotation survives across all sequences.")
        logger.info("Conclusion: the coupled sequences are JOINTLY UNIQUE under rotation-only symmetry.")
        return True

    logger.info("Non-identity rotations %s preserve all sequences simultaneously.", sorted(joint_rotations - {0}))
    logger.info("Conclusion: the coupled sequences are NOT jointly unique under rotation-only symmetry.")
    return False


def is_cyclically_unique(field, tol):
    """
    Check if a sequence is unique under cyclic rotations

    :param field: List of values (can contain None as wildcard)
    :param tol: Tolerance for numerical comparison
    """

    n = len(field)

    def is_same(a, b):
        return all(is_close(x, y, tol=tol) for x, y in zip(a, b))

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
    Pattern matching using combinations of features

    :param current_knowledge: Current knowledge of the polygon
    :param prior_knowledge: Prior model knowledge of the polygon
    :param match_corner_coordinates: Whether to match pattern first on the basis of corners before using rest of the knowledge
    :param find_match_in_individual_parameters: Whether to check if any of the parameters can individually give a unique match rather than checking for uniqueness across all parameters jointly
    :return: Whether a unique pattern is found and if so, the index of the first edge in current_knowledge
    """

    rck = current_knowledge
    rpk = prior_knowledge
    n_sides = rck.n_sides

    # Note: there are constraints such as no two adjacent edges have same slope or uniit-vecotr. 
    # This leads to adding few constraints to determinatoin of uniqueness (TODO?)
    if find_match_in_individual_parameters:
        # Motivation: for example, if initially it is known that only one edge dihedral angle is unique, then when it is perceived, 
        # it can lead to matching indices without 
        fields_to_check = ['slopes',  'edge_unit_vectors', 'lengths',
                           'corners', 'corner_angles',     'dihedrals']

        def matches(curr_knw_rotated, prior_knw_field):
            for a, b in zip(curr_knw_rotated, prior_knw_field):
                if a is not None and not is_close(a, b):
                    return False
            return True

        for field in fields_to_check:
            rpk_field = getattr(rpk, field)
            rck_field = getattr(rck, field)
            # one field of prior knowledge must be complete
            if not all(v is not None for v in rpk_field):
                continue
            # check cyclic uniqueness
            if not is_cyclically_unique(rpk_field, tol):
                continue

            count = 0
            shift_idx_rck = None
            for shift in range(n):
                rotated = rck_field[shift:] + rck_field[:shift]
                if matches(rotated, rpk_field):
                    count += 1
                    shift_idx_rck = shift
            if count == 1:
                return True, shift_idx_rck
        logger.info("No unique match found in individual parameters.")
        return False, None

    if match_corner_coordinates:
        for c in range(n_sides): # rck index
            for p in range(n_sides): # rpk index
                rck_corner_c = rck.corners[c]
                rpk_corner_p = rpk.corners[p]
                if rck_corner_c is not None and rpk_corner_p is not None and is_close(rck_corner_c, rpk_corner_p, tol=tol):
                    print("Found matching corner coordinates at rck index", c, "and rpk index", p)
                    return True, (c-p) % n_sides
        return False, None

    rp_sequences = [rpk.dihedrals, rpk.corner_angles, rpk.slopes, rpk.lengths]
    rc_sequences = [rck.dihedrals, rck.corner_angles, rck.slopes, rck.lengths]
    n = len(rp_sequences[0]) # number of edges
    m = len(rp_sequences)    # number of parameters

    valid_rotations = []

    for k in range(n):
        pattern_found = True
        for f in range(m):
            rp = rp_sequences[f]
            rc = rc_sequences[f]
            # check for consistency under k-rotation by shifting rc by k positions
            for i in range(n):
                a = rp[i]
                b = rc[(i + k) % n]
                if a is not None and b is not None and not is_close(a, b, tol=tol):
                    pattern_found = False
                    break
            # if current k position switch is invalid, break the loop
            if not pattern_found:
                break
        if pattern_found:
            valid_rotations.append(k)

    if len(valid_rotations) == 0:
        raise ValueError("No valid rotation: polygons do not match.")
    if len(valid_rotations) > 1:
        raise ValueError(f"Ambiguous mapping: multiple rotations possible {valid_rotations}")

    return True, valid_rotations[0]


def next_action(know: PolygonKnowledge,
                prev_action_instance: ActionInstance | None,
                rck_rearranged: bool,
                in_simulation: bool = False,
                gt: PolygonKnowledge = None) -> Tuple[Optional[ActionType], Optional[int]]:
    """
    Pick next edge feature to explore based on current knowledge. This is not linked to robot position, 
    but motion specification generator will take it into account to decide motion specification and setpoints

    :param know: Current knowledge of the polygon
    :param rck_rearranged: Whether rck has been rearranged using prior model
    :return: Next action and edge index. If no action possible, return (None, None)

    :param gt: Temporary know to fill out dof until constraint propagation symbolically is implemented
    """

    num_sides = know.n_sides
    edges_available_to_explore = []
    best_edge_idx  = None # edge whose exploration reduces dof the most among accessible edges to explore
    all_edge_unit_vectors_and_atleast_one_point_on_each_known = has_unit_vectors_and_points_for_all_edges(know)

    # if all edges are known, find any missing dihedrals
    if all_edge_unit_vectors_and_atleast_one_point_on_each_known:
        for i in range(num_sides):
            if know.dihedrals[i] is None:
                print("All edges are known, sliding to know dihedral angle")
                return (ActionType.SLIDE_OVER_SURFACE_PERPENDICULAR_TO_EDGE_GIVEN_ONE_POINT, i)

    if prev_action_instance.action_type:
        prev_action_spec = action_spec_from_action(prev_action_instance.action_type)
        prev_action_ref_edge_idx = prev_action_instance.edge_index # this is the reference edge index and not necessarily the edge to explore
    else:
        prev_action_spec = None
        prev_action_ref_edge_idx = None

    # perform default action
    if all(v is None for v in know.edge_unit_vectors):
        if (prev_action_instance.action_type and
            prev_action_spec.stop == Stop.UNTIL_EDGE_CONTACT):
            # after default first action, all edge unit vectors would still be unknown. Handled later in the code.
            pass
        else:
            print("Performing default action to explore first edge since no edge unit vector is known. ")
            print("prev_action_ref_edge_idx: ", prev_action_ref_edge_idx, "prev_action_spec: ", prev_action_spec)
            return (ActionType.SLIDE_OVER_SURFACE_UNTIL_EDGE, 0)

    # get best edge index to explore
    for i in range(num_sides):
        prev_i = (i - 1) % num_sides
        next_i = (i + 1) % num_sides

        edge_vector = know.edge_unit_vectors[i]
        prev_edge_vector = know.edge_unit_vectors[prev_i]
        next_edge_vector = know.edge_unit_vectors[next_i]

        points_on_prev_edge = know.get_all_points_on_edge(prev_i)
        points_on_next_edge = know.get_all_points_on_edge(next_i)
        points_on_edge = know.get_all_points_on_edge(i)

        should_explore_edge = edge_vector is None or len(points_on_edge) == 0
        if should_explore_edge:
            if not rck_rearranged:
                # when there is no matching indices found b/w rck and rpk, 
                # since prior knowledge cannot be used, the default action is 
                # to explore edges relative to previous explored edge
                # TODO: suppose the knowledge that all corner angles are 90 degrees is known, 
                # or all edge lengths are same are known, check propagation of knw and action selection
                if ((i == 0 and not edge_vector) or 
                    (prev_edge_vector is not None and len(points_on_prev_edge) > 0)):
                    best_edge_idx = i
                    print("Current knowledge is not matched with the prior knowledge. Best edge to explore is: ", best_edge_idx)
                    break
            else:
                # condition such as if edges are known to be parallel, then it is also explorable could go here
                if prev_edge_vector is not None and len(points_on_prev_edge) > 0:
                    edges_available_to_explore.append(i)
                elif next_edge_vector is not None and len(points_on_next_edge) > 0:
                    edges_available_to_explore.append(i)
                # this implicitly allows to explore the next edge from 
                # the previous action of 'edge detection' or 'move until corner'
                # if multiple edges can be explored with same resulting dof

    if not rck_rearranged:
        if best_edge_idx is None:
            print("No edges available to explore")
            return None, None
    elif len(edges_available_to_explore) == 0:
        print("No edges available to explore based on current knowledge.")
        return None, None
    else:
        # create a copy of know and see getting two points on which edge reduces dof the most
        best_edge_idx = edges_available_to_explore[0]
        best_dof = find_dof(know)
        
        if in_simulation:

            for edge_idx in edges_available_to_explore:
                temp_know = copy.deepcopy(know)
                # simulate measuring two points
                next_edge_idx = (edge_idx + 1) % num_sides
                internal_pts_on_edge = get_random_points_on_line(
                    gt.corners[next_edge_idx], gt.corners[edge_idx], num_points=2) # TODO: use symbolic constraint propagation here to get dof
                print(" Thinking... internal_pts_on_edge: ", internal_pts_on_edge)
                temp_know.internal_points_on_edge[edge_idx] = copy.deepcopy(internal_pts_on_edge)
                temp_know.is_reflexive_angle[edge_idx] = gt.is_reflexive_angle[edge_idx]
                propagate_parameters(temp_know)
                dof_after_two_points = find_dof(temp_know)
                if dof_after_two_points < best_dof:
                    best_dof = dof_after_two_points
                    best_edge_idx = edge_idx

    # find best action spec and the reference edge
    prev_edge_idx_of_best_edge_idx = (best_edge_idx - 1) % num_sides
    next_edge_idx_of_best_edge_idx = (best_edge_idx + 1) % num_sides

    dihedral_angle_of_best_edge_deg = know.dihedrals[best_edge_idx]
    dihedral_angle_of_prev_edge_of_best_edge_deg = know.dihedrals[prev_edge_idx_of_best_edge_idx]
    dihedral_angle_of_next_edge_of_best_edge_deg = know.dihedrals[next_edge_idx_of_best_edge_idx]
    
    reflexivity_of_best_edge_idx = know.is_reflexive_angle[best_edge_idx]
    reflexivity_of_next_edge_idx_of_best_edge_idx = know.is_reflexive_angle[next_edge_idx_of_best_edge_idx]
    
    corner_of_next_edge_idx_of_best_edge_idx = know.corners[next_edge_idx_of_best_edge_idx]
    corner_of_best_edge_idx = know.corners[best_edge_idx]

    next_to_best_edge_idx_edge_vector = know.edge_unit_vectors[next_edge_idx_of_best_edge_idx]
    previous_of_best_edge_idx_edge_vector = know.edge_unit_vectors[prev_edge_idx_of_best_edge_idx]
    
    pts_on_next_edge_idx_of_best_edge_idx = know.get_all_points_on_edge(next_edge_idx_of_best_edge_idx)
    pts_on_prev_edge_idx_of_best_edge_idx = know.get_all_points_on_edge(prev_edge_idx_of_best_edge_idx)
    pts_on_best_edge_idx = know.get_all_points_on_edge(best_edge_idx)

    is_adjacent_and_ordered = are_adjacent_and_action_in_order(
        prev_action_spec, best_edge_idx, prev_action_ref_edge_idx, num_sides
    )

    is_first_edge_case = (best_edge_idx == i == 0)

    is_parallel_from_outside = (
        prev_action_spec.mode == Mode.PARALLEL_IN_FREE_SPACE_FROM_OUTSIDE or
        prev_action_spec.mode == Mode.PARALLEL_OVER_SURFACE_FROM_OUTSIDE
    )

    if prev_action_instance.action_type:
        if ((is_adjacent_and_ordered and not is_parallel_from_outside)
            or is_first_edge_case
            or (not is_adjacent_and_ordered and is_parallel_from_outside)):
            
            # if already at the best edge
            if prev_action_spec.stop == Stop.UNTIL_EDGE_CONTACT_WITHIN_RANGE:
                if prev_action_spec.mode == Mode.PARALLEL_OVER_SURFACE:
                    reflexivity_of_interest = None
                    if prev_action_spec.direction == Direction.CCK:
                        reflexivity_of_interest = reflexivity_of_best_edge_idx
                    elif prev_action_spec.direction == Direction.CK:
                        reflexivity_of_interest = reflexivity_of_next_edge_idx_of_best_edge_idx
                    else:
                        raise ValueError("Direction of previous action is not found for within-range edge contact")
                    
                    # Case 1: found the target edge. Continue as if the previous action ended at edge contact on best_edge_idx.
                    if reflexivity_of_interest is False:
                        if dihedral_angle_of_best_edge_deg == 270:
                            if prev_action_spec.direction == Direction.CCK:
                                direction = Direction.CK
                                ref_edge = prev_edge_idx_of_best_edge_idx
                            elif prev_action_spec.direction == Direction.CK:
                                direction = Direction.CCK
                                ref_edge = next_edge_idx_of_best_edge_idx
                            mode = Mode.PARALLEL_IN_FREE_SPACE_FROM_OUTSIDE
                            stop = Stop.UNTIL_EDGE_CONTACT
                            best_action_spec = ActionSpec(direction, mode, stop)
                            best_action_type = SPEC_TO_ACTION[best_action_spec]
                            return (best_action_type, ref_edge) 
                        
                        elif reflexivity_of_next_edge_idx_of_best_edge_idx is None:
                            direction = Direction.CCK
                        elif prev_action_spec.direction == Direction.CCK:
                            direction = Direction.CCK
                        elif prev_action_spec.direction == Direction.CK:
                            direction = Direction.CK
                        else:
                            raise ValueError("Direction of traversal could not be decided after within-range edge contact")

                        if direction == Direction.CK:
                            reflexivity_of_adj_edge_idx_of_best_edge_idx = reflexivity_of_best_edge_idx
                            corner_of_adj_edge_idx_of_best_edge_idx = corner_of_best_edge_idx
                        elif direction == Direction.CCK:
                            reflexivity_of_adj_edge_idx_of_best_edge_idx = reflexivity_of_next_edge_idx_of_best_edge_idx
                            corner_of_adj_edge_idx_of_best_edge_idx = corner_of_next_edge_idx_of_best_edge_idx
                        else:
                            raise ValueError("Direction of traversal could not be decided after within-range edge contact")

                        if ((direction == Direction.CCK and next_to_best_edge_idx_edge_vector and len(pts_on_next_edge_idx_of_best_edge_idx) > 0) or
                            (direction == Direction.CK and previous_of_best_edge_idx_edge_vector and len(pts_on_prev_edge_idx_of_best_edge_idx) > 0)):
                            stop = Stop.VECTOR_ONLY
                        elif (reflexivity_of_adj_edge_idx_of_best_edge_idx is not None and
                              corner_of_adj_edge_idx_of_best_edge_idx is not None):
                            stop = Stop.VECTOR_ONLY
                        else:
                            stop = Stop.UNTIL_CORNER

                        if dihedral_angle_of_best_edge_deg == 90:
                            mode = Mode.AGAINST_VERTICAL
                        elif dihedral_angle_of_best_edge_deg == 270:
                            mode = Mode.AGAINST_EDGE
                        elif dihedral_angle_of_best_edge_deg is None:
                            raise ValueError("Unknown dihedral angle after within-range edge contact")
                        else:
                            raise NotImplementedError(f"Unhandled dihedral angle {dihedral_angle_of_best_edge_deg}")

                        best_action_spec = ActionSpec(direction, mode, stop)
                        best_action_type = SPEC_TO_ACTION[best_action_spec]
                        return (best_action_type, best_edge_idx)

                    # Case 2: did not find the target edge. The tested corner should now be reflexive, so approach from outside.
                    elif reflexivity_of_interest is True:
                        if prev_action_ref_edge_idx is None:
                            raise RuntimeError("Previous reference edge is required after within-range diagnostic slide")
                        if prev_action_spec.direction == Direction.CCK:
                            direction = Direction.CK
                        elif prev_action_spec.direction == Direction.CK:
                            direction = Direction.CCK
                        else:
                            raise ValueError("Previous action direction is required after within-range diagnostic slide")

                        ref_edge = prev_action_ref_edge_idx
                        mode = Mode.PARALLEL_OVER_SURFACE_FROM_OUTSIDE
                        stop = Stop.UNTIL_EDGE_CONTACT
                        best_action_spec = ActionSpec(direction, mode, stop)
                        best_action_type = SPEC_TO_ACTION[best_action_spec]
                        return (best_action_type, ref_edge)

                    else:
                        raise ValueError("Unexpected negative count of internal points on best edge.")
                else:
                    # log error
                    raise NotImplementedError("Unhandled case of being already at the best edge when sliding until edge contact within range for modes other than parallel over surface.")

            elif prev_action_spec.stop == Stop.UNTIL_EDGE_CONTACT:
                direction = mode = stop = None
                
                if (prev_action_spec.mode in [Mode.OVER_SURFACE, Mode.PARALLEL_OVER_SURFACE, Mode.PARALLEL_OVER_SURFACE_FROM_OUTSIDE] and
                    dihedral_angle_of_best_edge_deg == 270):
                    if prev_action_spec.direction == Direction.CCK:
                        direction = Direction.CK
                        ref_edge = prev_edge_idx_of_best_edge_idx
                    elif prev_action_spec.direction == Direction.CK:
                        direction = Direction.CCK
                        ref_edge = next_edge_idx_of_best_edge_idx
                    mode = Mode.PARALLEL_IN_FREE_SPACE_FROM_OUTSIDE
                    stop = Stop.UNTIL_EDGE_CONTACT
                    best_action_spec = ActionSpec(direction, mode, stop)
                    best_action_type = SPEC_TO_ACTION[best_action_spec]
                    return (best_action_type, ref_edge) # since the reference edge is the adjacent edge
                # decide direction of traversal. 
                # First based on reflexivity of adjacent edges, then based on default direction of traversal
                elif reflexivity_of_next_edge_idx_of_best_edge_idx is None:
                    direction = Direction.CCK
                elif reflexivity_of_best_edge_idx is None:
                    direction = Direction.CK
                elif prev_action_spec.mode == Mode.PARALLEL_IN_FREE_SPACE_FROM_OUTSIDE:
                    if prev_action_spec.direction == Direction.CCK:
                        direction = Direction.CK
                    if prev_action_spec.direction == Direction.CK:
                        direction = Direction.CCK
                else:
                    if prev_action_spec.direction == Direction.CCK:
                        direction = Direction.CCK
                    elif prev_action_spec.direction == Direction.CK:
                        direction = Direction.CK

                # identify towards which corner the action will be
                if direction == Direction.CK:
                    reflexivity_of_adj_edge_idx_of_best_edge_idx = reflexivity_of_best_edge_idx
                    corner_of_adj_edge_idx_of_best_edge_idx = corner_of_best_edge_idx
                elif direction == Direction.CCK:
                    reflexivity_of_adj_edge_idx_of_best_edge_idx = reflexivity_of_next_edge_idx_of_best_edge_idx
                    corner_of_adj_edge_idx_of_best_edge_idx = corner_of_next_edge_idx_of_best_edge_idx
                else:
                    raise ValueError("Direction of traversal could not be decided")

                # decide where to stop
                if ((direction == Direction.CCK and next_to_best_edge_idx_edge_vector and len(pts_on_next_edge_idx_of_best_edge_idx) > 0) or
                    (direction == Direction.CK and previous_of_best_edge_idx_edge_vector and len(pts_on_prev_edge_idx_of_best_edge_idx) > 0)):
                    stop = Stop.VECTOR_ONLY
                # if it is reflexive, to get a ref point to approach the adj edge irrepective of its dih.ang., it is req. to have corner info to limit to vector exploration
                elif (reflexivity_of_adj_edge_idx_of_best_edge_idx is not None and
                      corner_of_adj_edge_idx_of_best_edge_idx is not None):
                    stop = Stop.VECTOR_ONLY
                else:
                    stop = Stop.UNTIL_CORNER

                # decide how to slide
                if dihedral_angle_of_best_edge_deg == 90:
                    mode = Mode.AGAINST_VERTICAL
                elif dihedral_angle_of_best_edge_deg == 270:
                    mode = Mode.AGAINST_EDGE
                elif dihedral_angle_of_best_edge_deg is None:
                    raise ValueError("Unknown dihedral angle even after edge detection")
                else:
                    raise NotImplementedError(f"Unhandled dihedral angle {dihedral_angle_of_best_edge_deg}")

                best_action_spec = ActionSpec(direction, mode, stop)
                best_action_type = SPEC_TO_ACTION[best_action_spec]
                return (best_action_type, best_edge_idx)

            # when already at the corner of a best edge
            # Note: fact that if sliding over an edge has occured  until its corner then it has ref points which
            # allows to appraoch from outside the edge is coupled here
            elif prev_action_spec.stop == Stop.UNTIL_CORNER:
                ref_edge = None
                next_adj_edge_idx = None
                reflexivity_of_corner_of_interest_of_best_edge_idx = None
                direction = mode = stop = None

                # decide direction of traversal based on relative location of best edge index used for previous action
                if prev_action_ref_edge_idx == prev_edge_idx_of_best_edge_idx:
                    ref_edge = prev_edge_idx_of_best_edge_idx
                    reflexivity_of_corner_of_interest_of_best_edge_idx = reflexivity_of_best_edge_idx
                    direction = Direction.CCK
                elif prev_action_ref_edge_idx == next_edge_idx_of_best_edge_idx:
                    ref_edge = next_edge_idx_of_best_edge_idx
                    direction = Direction.CK
                    reflexivity_of_corner_of_interest_of_best_edge_idx = reflexivity_of_next_edge_idx_of_best_edge_idx
                else:
                    raise ValueError("Previous action on an edge until the corner was not adjacent to best edge")

                # decide adjacent edge index based on direction of traversal
                if direction == Direction.CCK:
                    next_adj_edge_idx = next_edge_idx_of_best_edge_idx
                    corner_of_adj_idx_of_best_idx_towards_motion = know.corners[next_edge_idx_of_best_edge_idx]
                    reflexivity_of_adj_idx_of_best_idx_towards_motion = know.is_reflexive_angle[next_edge_idx_of_best_edge_idx]
                    edge_vector_of_adj_edge_idx = know.edge_unit_vectors[next_adj_edge_idx]
                    pts_on_edge_vector_of_adj_edge_idx = know.get_all_points_on_edge(next_adj_edge_idx)
                elif direction == Direction.CK:
                    next_adj_edge_idx = prev_edge_idx_of_best_edge_idx
                    corner_of_adj_idx_of_best_idx_towards_motion = know.corners[best_edge_idx]
                    reflexivity_of_adj_idx_of_best_idx_towards_motion = know.is_reflexive_angle[best_edge_idx]
                    edge_vector_of_adj_edge_idx = know.edge_unit_vectors[prev_edge_idx_of_best_edge_idx]
                    pts_on_edge_vector_of_adj_edge_idx = know.get_all_points_on_edge(prev_edge_idx_of_best_edge_idx)


                # if reflexive, dih=270 => slide along edge
                # if relfexive, dih=90/unknown => slide from outside
                # if non-reflexive, dih=270 => move from outside parallel
                # if non-reflexive, dih=90/unknown => slide parallel to edge over surface
                if reflexivity_of_corner_of_interest_of_best_edge_idx is True:
                    if dihedral_angle_of_best_edge_deg == 270:
                        if prev_action_spec.mode == Mode.AGAINST_VERTICAL:
                            mode = Mode.PARALLEL_OVER_SURFACE_FROM_OUTSIDE
                            direction = flip_direction(direction)
                        elif prev_action_spec.mode == Mode.AGAINST_EDGE:
                            mode = Mode.AGAINST_EDGE
                            ref_edge = best_edge_idx # updating reference edge index from edge of reference
                    elif dihedral_angle_of_best_edge_deg == 90 or dihedral_angle_of_best_edge_deg is None:
                        mode = Mode.PARALLEL_OVER_SURFACE_FROM_OUTSIDE
                        direction = flip_direction(direction)
                elif reflexivity_of_corner_of_interest_of_best_edge_idx is False:
                    if dihedral_angle_of_best_edge_deg == 90:
                        if prev_action_spec.mode == Mode.AGAINST_VERTICAL:
                            mode = Mode.AGAINST_VERTICAL
                            ref_edge = best_edge_idx # updating reference edge index from edge of reference
                        elif prev_action_spec.mode == Mode.AGAINST_EDGE:
                            mode = Mode.PARALLEL_OVER_SURFACE
                    elif dihedral_angle_of_best_edge_deg is None:
                        mode = Mode.PARALLEL_OVER_SURFACE
                    elif dihedral_angle_of_best_edge_deg == 270:
                        mode = Mode.PARALLEL_IN_FREE_SPACE_FROM_OUTSIDE
                        direction = flip_direction(direction)
                elif reflexivity_of_corner_of_interest_of_best_edge_idx is None:
                    if prev_action_spec.mode == Mode.AGAINST_EDGE:
                        # Previous slide-against-edge-until-corner did not resolve reflexivity.
                        # Probe the candidate adjacent edge by sliding parallel to the edge
                        # that was just used as reference. This is done only if the reflexivity is
                        # not known.
                        mode = Mode.PARALLEL_OVER_SURFACE
                        stop = Stop.UNTIL_EDGE_CONTACT_WITHIN_RANGE
                    else:
                        raise ValueError(
                            "Corner reflexivity is unknown after UNTIL_CORNER action, "
                            f"but no diagnostic parallel-over-surface action is defined for mode {prev_action_spec.mode}"
                        )
                
                # next_adj_edge_idx has corner, reflexivity, no need to slide until corner
                if stop is None:
                    if (mode == Mode.PARALLEL_IN_FREE_SPACE_FROM_OUTSIDE or
                        mode == Mode.PARALLEL_OVER_SURFACE or
                        mode == Mode.PERPENDICULAR_TO_EDGE_OVER_SURFACE or
                        mode == Mode.PARALLEL_OVER_SURFACE_FROM_OUTSIDE):
                        stop = Stop.UNTIL_EDGE_CONTACT
                    elif (mode == Mode.AGAINST_EDGE or
                        mode == Mode.AGAINST_VERTICAL):
                        # if next edge-vector is (known and atleast one point on it is known) or (unknown and reflex+corner known), then no need to slide until corner
                        if (edge_vector_of_adj_edge_idx is not None and len(pts_on_edge_vector_of_adj_edge_idx) > 0):
                            stop = Stop.VECTOR_ONLY
                        # if it is reflexive, to get a ref point to approach the adj edge irrepective of its dih.ang., it is req. to have corner info to limit to vector exploration
                        elif (reflexivity_of_adj_idx_of_best_idx_towards_motion is not None and
                            corner_of_adj_idx_of_best_idx_towards_motion is not None):
                            stop = Stop.VECTOR_ONLY
                        else:
                            stop = Stop.UNTIL_CORNER
                    else:
                        raise ValueError("Unhandled Mode, or mode not set")

                best_action_spec = ActionSpec(direction, mode, stop)
                best_action_type = SPEC_TO_ACTION[best_action_spec]
                return (best_action_type, ref_edge)
            
            else:
                print("Previous action was to just get edge-vector and it is not ending at a corner or an edge. Selecting action independent of robot end-effector location")
                
        elif (are_adjacent(best_edge_idx, prev_action_ref_edge_idx, num_sides) and not
            is_adjacent_and_ordered):
            print("Debug: best_edge_idx: ", best_edge_idx, " prev_action_ref_edge_idx (ref): ", prev_action_ref_edge_idx)
            print("Though action took adjacent edge for reference, it did not traverse in the desired direction to take it ahead. Selecting action independent of robot end-effector location")

        # edge vector is unknown or no point on it is known; previous edge is known and atleast one point on it is known
        # so slide to find its internal point
        ref_edge = None
        dihedral_angle_of_reference_edge = None
        reflexivity_of_corner_of_interest_of_best_edge_idx = None
        corner_of_interest = None
        direction = mode = stop = None

        prev_edge_eligible_as_reference = edge_unit_vector_and_atleast_one_point_known(know, prev_edge_idx_of_best_edge_idx)
        next_edge_eligible_as_reference = edge_unit_vector_and_atleast_one_point_known(know, next_edge_idx_of_best_edge_idx)

        if not next_edge_eligible_as_reference and prev_edge_eligible_as_reference:
            ref_edge = prev_edge_idx_of_best_edge_idx
        elif not prev_edge_eligible_as_reference and next_edge_eligible_as_reference:
            ref_edge = next_edge_idx_of_best_edge_idx
        elif next_edge_eligible_as_reference and prev_edge_eligible_as_reference:
            if reflexivity_of_best_edge_idx is not None and reflexivity_of_next_edge_idx_of_best_edge_idx is None:
                ref_edge = prev_edge_idx_of_best_edge_idx
            elif reflexivity_of_best_edge_idx is None and reflexivity_of_next_edge_idx_of_best_edge_idx is not None:
                ref_edge = next_edge_idx_of_best_edge_idx
            elif reflexivity_of_best_edge_idx is None and reflexivity_of_next_edge_idx_of_best_edge_idx is None:
                if corner_of_best_edge_idx is not None:
                    ref_edge = prev_edge_idx_of_best_edge_idx
                elif corner_of_next_edge_idx_of_best_edge_idx is not None:
                    ref_edge = next_edge_idx_of_best_edge_idx
                elif dihedral_angle_of_prev_edge_of_best_edge_deg is not None:
                    ref_edge = prev_edge_idx_of_best_edge_idx
                elif dihedral_angle_of_next_edge_of_best_edge_deg is not None:
                    ref_edge = next_edge_idx_of_best_edge_idx
                else:
                    ref_edge = prev_edge_idx_of_best_edge_idx
            else:
                ref_edge = prev_edge_idx_of_best_edge_idx
        if ref_edge is None:
            raise RuntimeError("Best edge doesn't have any adjacent edge with known edge vector and atleast a point")

        if ref_edge == prev_edge_idx_of_best_edge_idx:
            corner_of_interest = corner_of_best_edge_idx
            reflexivity_of_corner_of_interest_of_best_edge_idx = reflexivity_of_best_edge_idx
            dihedral_angle_of_reference_edge = dihedral_angle_of_prev_edge_of_best_edge_deg
            direction = Direction.CCK
        elif ref_edge == next_edge_idx_of_best_edge_idx:
            corner_of_interest = corner_of_next_edge_idx_of_best_edge_idx
            reflexivity_of_corner_of_interest_of_best_edge_idx = reflexivity_of_next_edge_idx_of_best_edge_idx
            dihedral_angle_of_reference_edge = dihedral_angle_of_next_edge_of_best_edge_deg
            direction = Direction.CK
        if reflexivity_of_corner_of_interest_of_best_edge_idx is None:
            if dihedral_angle_of_reference_edge is None:
                direction = Direction.CCK
                mode = Mode.PERPENDICULAR_TO_EDGE_OVER_SURFACE
                stop = Stop.UNTIL_EDGE_CONTACT
            else:
                if dihedral_angle_of_reference_edge == 270.0:
                    mode = Mode.AGAINST_EDGE
                    stop = Stop.UNTIL_CORNER
                elif dihedral_angle_of_reference_edge == 90.0:
                    mode = Mode.AGAINST_VERTICAL
                    stop = Stop.UNTIL_CORNER
                else:
                    raise ValueError("Dihedral angle is other than 90 or 270 or None")
        else:
            if reflexivity_of_corner_of_interest_of_best_edge_idx is False:
                mode = Mode.PARALLEL_OVER_SURFACE
                stop = Stop.UNTIL_EDGE_CONTACT
            elif reflexivity_of_corner_of_interest_of_best_edge_idx is True:
                # TODO: save what actions were performed over each edge in that edge. So if an edge was slided until corner/corner is known, then only appraoching from outside works.
                # convert action type to action spec
                prev_spec = ACTION_TO_SPEC[prev_action_instance.action_type]
                if corner_of_interest is not None: # or prev_spec.stop == Stop.UNTIL_CORNER: (with assumption of prev. motion along adj edge). Ideally, the history of action over the edge includes sliding until corner
                    mode = Mode.PARALLEL_OVER_SURFACE_FROM_OUTSIDE
                    stop = Stop.UNTIL_EDGE_CONTACT
                    direction = flip_direction(direction)
                else:
                    if dihedral_angle_of_reference_edge == 270.0:
                        mode = Mode.AGAINST_EDGE
                        stop = Stop.UNTIL_CORNER
                    elif dihedral_angle_of_reference_edge == 90.0:
                        mode = Mode.AGAINST_VERTICAL
                        stop = Stop.UNTIL_CORNER
                    else:
                        raise ValueError("Dihedral angle is other than 90 or 270 or None")

        best_action_spec = ActionSpec(direction, mode, stop)
        best_action_type = SPEC_TO_ACTION[best_action_spec]
        return (best_action_type, ref_edge)
    return None, None


def rearrange_rck_using_prior_knowledge(rck: PolygonKnowledge,
                                        rpk_first_idx_in_rck: int) -> None:
    """
    Rearrange rck parameters based on prior model using unique pattern match

    :param rck: Current knowledge of the polygon
    :param rpk: Prior knowledge of the polygon
    :param rpk_first_idx_in_rck: First index in rpk mapped to rck
    :return: None, updates rck in place
    """
    rck.slopes[:] = rck.slopes[rpk_first_idx_in_rck:] + rck.slopes[:rpk_first_idx_in_rck]
    rck.lengths[:] = rck.lengths[rpk_first_idx_in_rck:] + rck.lengths[:rpk_first_idx_in_rck]
    rck.edge_unit_vectors[:] = rck.edge_unit_vectors[rpk_first_idx_in_rck:] + rck.edge_unit_vectors[:rpk_first_idx_in_rck]
    rck.corners[:] = rck.corners[rpk_first_idx_in_rck:] + rck.corners[:rpk_first_idx_in_rck]
    rck.is_reflexive_angle[:] = rck.is_reflexive_angle[rpk_first_idx_in_rck:] + rck.is_reflexive_angle[:rpk_first_idx_in_rck]
    rck.corner_angles[:] = rck.corner_angles[rpk_first_idx_in_rck:] + rck.corner_angles[:rpk_first_idx_in_rck]
    rck.dihedrals[:] = rck.dihedrals[rpk_first_idx_in_rck:] + rck.dihedrals[:rpk_first_idx_in_rck]
    rck.internal_points_on_edge[:] = rck.internal_points_on_edge[rpk_first_idx_in_rck:] + rck.internal_points_on_edge[:rpk_first_idx_in_rck]
    print(f" => Rearranged rck using prior model where rck edge {rpk_first_idx_in_rck} is now edge 0.")


def fill_missing_parameters(rck: PolygonKnowledge,
                            rpk: PolygonKnowledge,
                            rpk_rck_matching_idx_found: bool) -> None:
    """
    Fill missing parameters in rck using prior model

    :param rck: Current knowledge of the polygon
    :param rpk: Prior knowledge of the polygon
    :return: None, updates rck in place
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
            if len(rpk.internal_points_on_edge[i]) > 0:
                for point in rpk.internal_points_on_edge[i]:
                    if not point_in_list(point, rck.internal_points_on_edge[i]):
                        rck.internal_points_on_edge[i].append(point)
                        print(f" => Filled internal_points_on_edge {i} with {point} using prior model")
    else:
        # check if same value for all indices of features found, if so, update it to ck
        if are_list_elements_uniform(rpk.is_reflexive_angle):
            for i in range(rck.n_sides):
                rck.is_reflexive_angle[i] = rpk.is_reflexive_angle[0]
            print(f" => Filled is_reflexive_angle of all edges as {rck.is_reflexive_angle[0]} using prior model")
        if are_list_elements_uniform(rpk.corner_angles):
            for i in range(rck.n_sides):
                rck.corner_angles[i] = rpk.corner_angles[0]
            print(f" => Filled is_reflexive_angle of all edges as {rck.corner_angles[0]} using prior model")
        if are_list_elements_uniform(rpk.lengths):
            for i in range(rck.n_sides):
                rck.lengths[i] = rpk.lengths[0]
            print(f" => Filled is_reflexive_angle of all edges as {rck.lengths[0]} using prior model")
        if are_list_elements_uniform(rpk.dihedrals):
            for i in range(rck.n_sides):
                rck.dihedrals[i] = rpk.dihedrals[0]
            print(f" => Filled is_reflexive_angle of all edges as {rck.dihedrals[0]} using prior model")
