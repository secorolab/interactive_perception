
from random import random
import numpy as np
import math
from typing import Union, Tuple, Optional, List, Callable, TypeVar, Sequence, Dict
from .polygon_knowledge import PolygonKnowledge
from .data_structures import *

def is_close(p1: Union[float, Tuple[float, float]], 
             p2: Union[float, Tuple[float, float]], 
             tol: float = 0.02) -> bool:
    """
    Check if two points or two angles are within a certain distance tolerance.

    :param p1: Point (x1, y1), angle or distance
    :param p2: Point (x2, y2), angle or distance
    :param tol: Distance tolerance
    :return: Whether distance between p1 and p2 is within tolerance
    """
    # check if p1 and p2 are not None or doesn't contain None
    if p1 is None or p2 is None:
        raise ValueError("None value provided for point or angle comparison")
    # check if p1 and p2 are angles (float or int)
    if isinstance(p1, (int, float)) and isinstance(p2, (int, float)):
        return abs(p1 - p2) <= tol
    # else, assume p1 and p2 are points (x, y)
    if p1[0] is None or p1[1] is None or p2[0] is None or p2[1] is None:
        raise ValueError("Point coordinates cannot be None")
    return np.linalg.norm(np.asarray(p1) - np.asarray(p2)) <= tol


def point_in_list(point: Tuple[float, float],
                  point_list: List[Tuple[float, float]],
                  tol: float = 0.01) -> Tuple[bool, Optional[int]]:
    """
    Check whether a point exists in a list of points within a tolerance.

    :param point: (x, y) tuple representing the query point
    :param point_list: list of (x, y) tuples
    :param tol: distance tolerance
    :return: (found, index) where index is None if not found
    """
    if point is None:
        raise ValueError("Point to search for cannot be None")
    
    if len(point_list) == 0:
        raise ValueError("List of points is empty")

    for idx, p in enumerate(point_list):
        if is_close(point, p, tol):
            return True, idx

    return False, None


def line_intersection(p1: Tuple[float, float], 
                      m1: float, 
                      p2: Tuple[float, float], 
                      m2: float) -> Optional[Tuple[float, float]]:
    """
    Intersect two lines: y = m1*(x-x1) + y1, y = m2*(x-x2) + y2

    :param p1: Point on line 1
    :param m1: Slope of line 1
    :param p2: Point on line 2
    :param m2: Slope of line 2
    :return: Intersection point or None if lines are parallel
    """
    if m1 == m2:
        return None
    x1, y1 = p1
    x2, y2 = p2
    xi = ((m1*x1 - y1) - (m2*x2 - y2)) / (m1 - m2)
    yi = m1*(xi - x1) + y1
    return (xi, yi)


def find_dof(know: PolygonKnowledge) -> int:
    """
    DOF estimate

    :param know: Knowledge of the polygon
    :return: Estimated degrees of freedom remaining
    """
    # Assumption: the surface slope is already known, and it is considered as 2D

    num_sides = know.n_sides
    dof = (2*(num_sides-1)-1) + 3 + num_sides # total params: (2*(n-1) -1) intrinsic dof + 3 rigid body dof + num_sides of dihedral angles = absolute dof
    # Note: intrinsic is reduced by one as the edge_unit_vector information counted for rigid body also accounts for one corner angle
    # subtract 1 rigid-body dof (extrinsic) if atleast one edge_unit_vector is known
    if any(u is not None for u in know.edge_unit_vectors):
        dof -= 1
    # subtract 2 rigid-body dof (extrinsic) if atleast one corner is known
    if any(c is not None for c in know.corners):
        dof -= 2

    # subtract one intrinsic dof for each known corner angle and length
    # -1 for last edge, and additional -1 for known edge_unit_vector (as it constrains one corner angle)
    dof -= min(num_sides-2, sum(1 for a in know.corner_angles if a is not None))
    dof -= min(num_sides-1, sum(1 for l in know.lengths if l is not None))
    dof -= sum(1 for d in know.dihedrals if d is not None)

    return dof


T = TypeVar("T")
def is_sorted_by_key(seq: Sequence[T],
                    key: Callable[[T], float],
                    reverse: bool = False,
                    tol: float = 5e-3,) -> bool:
    """
    Check if the values in the sequence are sorted based on the distance to the corner.

    :param seq: List of elements to check for sorting.
    :param key: Function mapping each element to a sortable scalar key.
    :param reverse: A flag, if set to True, checks if the sequence is sorted in descending order. Default is False
    :return: Whether the sequence is sorted based on the distance to the corner in the specified order
    """
    
    max_violations = 3 if len(seq) > 10 else 1 # Allow more violations for longer sequences
    
    if len(seq) < 2:
        return True

    distances: list[float] = [float(key(x)) for x in seq]

    if any(math.isnan(d) for d in distances):
        raise ValueError("Key function returned NaN")

    # if reverse: # descending order
    #     return bool(all(distances[i] - distances[i + 1] >= -tol for i in range(len(distances) - 1)))
    # else: # ascending order
    #     return bool(all(distances[i] - distances[i + 1] <= tol for i in range(len(distances) - 1)))
    
    violations = 0
    for i in range(len(distances) - 1):

        if reverse:  # descending
            ok = distances[i] - distances[i + 1] >= -tol
        else:        # ascending
            ok = distances[i] - distances[i + 1] <= tol

        if not ok:
            violations += 1

            if violations > max_violations:
                print(f"Sequence is not sorted by key with tolerance {tol}. Violations: {violations} out of {len(seq)-1} comparisons.")
                return False
    
    return True

def get_random_points_on_line(p1: Tuple[float, float], 
                              p2: Tuple[float, float], 
                              num_points: int = 2) -> List[Tuple[float, float]]:
    """
    Generate random points on the line segment between p1 and p2 in the same order

    :param p1: Start point
    :param p2: End point
    :param num_points: Number of random points to generate
    :return: List random points on the line segment
    """

    eps = 0.1
    t_values = np.sort(np.random.uniform(eps, 1 - eps, num_points))
    
    # debug
    print(f"Generating {num_points} random points on line segment between {p1} and {p2} with t values: {t_values}")
    print(f"sampled points before sorting: {[ (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])) for t in t_values]}")

    return [ (p1[0] + t * (p2[0] - p1[0]),
              p1[1] + t * (p2[1] - p1[1]))
              for t in t_values ]

def get_unit_vector(head_point: Tuple[float, float],
                    tail_point: Tuple[float, float]) -> np.ndarray:
    """
    Compute the unit vector pointing from tail_point to head_point.

    :param head_point: Coordinates of the head
    :param tail_point: Coordinates of the tail
    :return: Direction cosines
    """
    v = np.array(head_point, dtype=float) - np.array(tail_point, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError("Cannot compute unit vector from coincident points")
    unit_vector = v/norm

    return (float(unit_vector[0]), float(unit_vector[1]))


def get_rotated_edge_slope(m: float,
                           theta_rad: float) -> float:
    """
    Rotate a line (given by slope m) by a signed angle

    :param m: Slope of the original line
    :param theta_rad: Signed rotation angle in radians
    :return: Slope of the rotated line
    """
    phi = np.arctan(m)
    phi_rot = phi + theta_rad

    cos_theta = np.cos(phi_rot)
    if np.isclose(cos_theta, 0.0):
        return np.inf

    return np.tan(phi_rot)


def get_angle_between_vector(vector_1: np.array,
                             vector_2: np.array) -> float:
    """
    Get angle between two vectors
    
    :param vector_1: First vector
    :param vector_2: Decond vector
    :return: Angle between two vectors in degrees
    """
    # Calculate the dot product
    dot_product = np.dot(vector_1, vector_2)
    
    # Calculate the magnitudes (norms) of the vectors
    norm_1 = np.linalg.norm(vector_1)
    norm_2 = np.linalg.norm(vector_2)
    
    # Calculate the cosine of the angle using the dot product and norms
    cos_angle = dot_product / (norm_1 * norm_2)
    
    # To handle any floating point inaccuracies that might result in values slightly greater than 1 or less than -1
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Calculate the angle in radians
    angle_radians = np.arccos(cos_angle)
    
    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)
    
    vector_1_3d = np.append(vector_1, 0)
    vector_2_3d = np.append(vector_2, 0)
    # Calculate the cross product to check the direction
    cross_product = np.cross(vector_1_3d, vector_2_3d)
    
    # If the z-component of the cross product is negative, the angle is clockwise (negate the angle)
    if cross_product[2] < 0:
        angle_degrees = -angle_degrees

    return float(angle_degrees)


def best_fit_unit_vector(points: List[Tuple[float, float]]):
    """
    Compute the unit direction vector of the best-fit line
    through a set of 2D points using PCA

    :param points: List of ordered points
    :return: Unit vector along the ordered points
    """
    if len(points) < 2:
        raise ValueError("At least two points are required")

    pts = np.asarray(points, dtype=float)

    # Compute centroid
    centroid = pts.mean(axis=0)

    # Center the points
    centered = pts - centroid

    # Covariance matrix
    cov = np.cov(centered, rowvar=False)

    # Principal eigenvector
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    direction = eigenvectors[:, np.argmax(eigenvalues)]

    # Normalize
    unit_vector = direction / np.linalg.norm(direction)

    ref_direction = pts[-1] - pts[0]
    if np.dot(unit_vector, ref_direction) < 0:
        unit_vector = -unit_vector

    return (float(unit_vector[0]), float(unit_vector[1]))


def rotate_vector_2d(v: Tuple[float, float],
                    angle_rad: float,
                    normalize: bool = True) -> Tuple[float, float]:
    """
    Rotate a 2D vector by a given angle. Positive angle corresponds to an anticlockwise rotation.

    :param v: 2D vector (x, y).
    :param angle_rad: Rotation angle in radians.
    :param normalize: If True, normalize the result to unit length.
    :return: Rotated 2D vector.
    """
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    x, y = v
    xr = x * cos_theta - y * sin_theta
    yr = x * sin_theta + y * cos_theta

    if normalize:
        norm = math.sqrt(xr**2 + yr**2)
        xr /= norm
        yr /= norm

    return (float(xr), float(yr))


def are_list_elements_uniform(parameter_list: List,
                              tolerance : float = 1e-3) -> bool:
    """
    Check if all elements in the list are the same and not None
    
    :param parameter_list: List of parameters of a feature
    :param tolerance: Tolerance for equality
    :return: whetrher all elements are the same and not None
    """
    if len(parameter_list) > 0 and None not in parameter_list:
        if isinstance(parameter_list[0], bool):
            return all(x is parameter_list[0] for x in parameter_list)

        average_value = sum(parameter_list) / len(parameter_list)
        if all(is_close(x, average_value, tol=tolerance) for x in parameter_list):
            return True
    return False


def edge_unit_vector_and_atleast_one_point_known(know: PolygonKnowledge,
                                                 edge_idx: int) -> bool:
    """
    Check if all edge unit vectors are known

    :param know: knowledge of the polygon
    :param edge_idx: index of edge
    :return: whether corresponding edge unit vector is known with atleast one point
    """
    if know.edge_unit_vectors[edge_idx] is None or len(know.get_all_points_on_edge(edge_idx))==0:
        return False
    return True


def action_spec_from_action(action: ActionType) -> Optional[ActionSpec]:
    """
    Retrieve the ActionSpec corresponding to a given action.

    :param action: Action identifier of type ActionType.
    :return: Corresponding ActionSpec if the action exists in the mapping,
             otherwise None.
    """
    return ACTION_TO_SPEC.get(action)


def are_adjacent(index1: int, index2: int, num_sides: int):
    """
    Check whether two indices are adjacent in a circular structure.

    :param index1: First index.
    :param index2: Second index.
    :param num_sides: Total number of sides (length of the circular structure).
    :return: True if the indices are adjacent, False otherwise.
    """
    if (index1 + 1) % num_sides == index2 or (index2 + 1) % num_sides == index1:
        return True
    return False


def are_adjacent_and_action_in_order(prev_action_spec: ActionSpec, 
                                     target_edge, 
                                     prev_action_edge_idx, 
                                     num_sides):
    """
    Check whether two indices are adjacent in a circular structure. CCK or CK direction of previous action determines the expected order of adjacency.

    :param prev_action_spec: previous action specification
    :param target_edge: target edge index
    :param prev_action_edge_idx: edge index used as reference in previous action
    :param num_sides: Total number of sides (length of the circular structure).
    :return: True if the indices are adjacent and the previous action was directed towards the target_edge, False otherwise.
    """
    prev_edge_idx_of_best_edge_idx = (target_edge - 1) % num_sides
    next_edge_idx_of_best_edge_idx = (target_edge + 1) % num_sides

    edges_are_adjacent = are_adjacent(target_edge, prev_action_edge_idx, num_sides)
    prev_action_towards_target_edge = True

    if prev_action_edge_idx == prev_edge_idx_of_best_edge_idx:
        if ((prev_action_spec.mode == Mode.PARALLEL_OVER_SURFACE_FROM_OUTSIDE or
             prev_action_spec.mode == Mode.PARALLEL_IN_FREE_SPACE_FROM_OUTSIDE) and 
             prev_action_spec.direction == Direction.CCK):
             prev_action_towards_target_edge = False
        elif prev_action_spec.direction == Direction.CK:
            prev_action_towards_target_edge = False
    if prev_action_edge_idx == next_edge_idx_of_best_edge_idx:
        if ((prev_action_spec.mode == Mode.PARALLEL_OVER_SURFACE_FROM_OUTSIDE or
             prev_action_spec.mode == Mode.PARALLEL_IN_FREE_SPACE_FROM_OUTSIDE) and 
             prev_action_spec.direction == Direction.CK):
             prev_action_towards_target_edge = False
        elif prev_action_spec.direction == Direction.CCK:
            prev_action_towards_target_edge = False

    if prev_action_towards_target_edge and edges_are_adjacent:
        return True

    return False


def flip_direction(direction: Direction):
    """
    Flip a rotational direction between clockwise and counter-clockwise.

    :param direction: Direction to flip (Direction.CK or Direction.CCK).
    :return: Flipped direction, or None if the input direction is invalid.
    """
    if direction == Direction.CK:
        return Direction.CCK
    elif direction == Direction.CCK:
        return Direction.CK
    else:
        raise ValueError("Direction to flip is None")


def has_unit_vectors_and_points_for_all_edges(know: PolygonKnowledge) -> bool:
    """
    Check if all edge unit vectors are known with minimum one point on each edge

    :param know: Knowledge of the polygon
    :return: whether all edge unit vectors are known with atleast one point on each
    """
    num_sides = know.n_sides
    for i in range(num_sides):
        if edge_unit_vector_and_atleast_one_point_known(know, i) is False:
            return False
    return True


def pre_process_polygon_knowledge(polygon_knowledge: PolygonKnowledge,
                         min_points_to_remove_outlers: int = 4,
                         inlier_distance_threshold: int = 0.02,
                         distance_threshold_between_points: float = 0.005) -> bool:
    graph_updated = False
    know = polygon_knowledge
    num_sides = know.n_sides
    changed = True
    
    while changed:
        changed = False

        # If corner coordinates are in internal_points_on_edge, remove them
        if not changed:
            for i in range(num_sides):
                next_idx = (i+1)%num_sides
                tail_corner = know.corners[i]
                head_corner = know.corners[next_idx]
                # Collect points to remove to avoid modifying list while iterating
                points_to_remove = []
                for point in know.internal_points_on_edge[i]:
                    if tail_corner is not None and is_close(tail_corner, point):
                        points_to_remove.append(point)
                        print(f" => Removed tail corner {tail_corner} of edge {i} from internal_points_on_edge[{i}]")
                    elif head_corner is not None and is_close(head_corner, point):
                        points_to_remove.append(point)
                        print(f" => Removed head corner {head_corner} of edge {i} from internal_points_on_edge[{i}]")
                
                # Remove collected points - use index-based removal to avoid numpy array issues
                for point_to_remove in points_to_remove:
                    # Find and remove the matching point using is_close for safe comparison
                    for j, p in enumerate(know.internal_points_on_edge[i]):
                        try:
                            if is_close(point_to_remove, p):
                                know.internal_points_on_edge[i].pop(j)
                                changed = True
                                break
                        except (ValueError, TypeError):
                            # is_close might fail if point format is unexpected, skip
                            pass

        # If internal points on an edge are not ordered in counterclockwise direction, rearrange them
        # This is currently on the basis of knowledge of a corner. When accumulating the points by sliding, 
        # it is assumed to be ordered in CCW by default
        if not changed:
            for i in range(num_sides):
                prev_idx = (i - 1) % num_sides
                current_corner = know.corners[i]
                if current_corner:
                    current_corner_arr = np.asarray(current_corner)
                    pts_on_curr_edge = know.internal_points_on_edge[i]

                    # Edge i: ascending order
                    if len(pts_on_curr_edge) > 1:
                        key_fn = lambda p: np.linalg.norm(np.asarray(p) - current_corner_arr)
                        if not is_sorted_by_key(pts_on_curr_edge, key_fn):
                            print(f" => The distances of points on edge {i} to corner {i} are not in ascending order: {[key_fn(p) for p in pts_on_curr_edge]}")
                            pts_on_curr_edge.sort(key=key_fn)
                            know.internal_points_on_edge[i] = pts_on_curr_edge
                            changed = True
                            print(f" => Rearranged points on edge {i} in ascending order to {pts_on_curr_edge}")

                    # Edge i-1: descending order
                    pts_on_prev_edge = know.internal_points_on_edge[prev_idx]
                    if len(pts_on_prev_edge) > 1:
                        key_fn = lambda p: np.linalg.norm(np.asarray(p) - current_corner_arr)
                        if not is_sorted_by_key(pts_on_prev_edge, key_fn, reverse=True):
                            print(f" => The distances of points on edge {prev_idx} to corner {i} are not in descending order: {[key_fn(p) for p in pts_on_prev_edge]}")
                            pts_on_prev_edge.sort(key=key_fn, reverse=True)
                            know.internal_points_on_edge[prev_idx] = pts_on_prev_edge
                            changed = True
                            print(f" => Rearranged points on edge {prev_idx} in descending order to  {pts_on_prev_edge}")
                            
        # If atleast 'min_points_to_remove_outliers' number of internal points exist, then remove outliers 
        # outside a threshold 'inlier_distance_threshold'
        if not changed:
            for i in range(num_sides):
                internal_points = know.internal_points_on_edge[i]
                if len(internal_points) >= min_points_to_remove_outlers:
                    all_points = []
                    all_points.extend(internal_points)
                    centroid = np.mean(all_points, axis=0)
                    unit_vector = best_fit_unit_vector(all_points)
                    normal = np.array([-unit_vector[1], unit_vector[0]])

                    to_remove = [
                        p for p in internal_points
                        if abs(np.dot(np.array(p) - centroid, normal)) > inlier_distance_threshold
                    ]

                    # Remove outliers
                    for point in to_remove:
                        if len(internal_points) >= min_points_to_remove_outlers:
                            for j, p in enumerate(internal_points):
                                try:
                                    if is_close(point, p):
                                        internal_points.pop(j)
                                        changed = True
                                        print(f" => Removed outlier point {point} from edge {i}")
                                        break
                                except (ValueError, TypeError):
                                    pass

        # If any points on edge is outside the edge bounded by corners, remove them. This could happen when slide until corner overshoots the corner in edge case
        if not changed:
            for i in range(num_sides):
                corner_start = know.corners[i]
                corner_end = know.corners[(i+1)%num_sides]
                if corner_start is None or corner_end is None:
                    continue
                
                corner_start_arr = np.asarray(corner_start)
                corner_end_arr = np.asarray(corner_end)
                edge_vector = corner_end_arr - corner_start_arr
                edge_length = np.linalg.norm(edge_vector)
                if edge_length == 0:
                    continue
                edge_unit_vector = edge_vector / edge_length

                to_remove = []
                for point in know.internal_points_on_edge[i]:
                    point_arr = np.asarray(point)
                    vec_from_start = point_arr - corner_start_arr
                    projection_length = np.dot(vec_from_start, edge_unit_vector)

                    if projection_length < 0 or projection_length > edge_length:
                        to_remove.append(point)

                for point in to_remove:
                    # Use index-based removal to avoid numpy array issues
                    for j, p in enumerate(know.internal_points_on_edge[i]):
                        try:
                            if is_close(point, p):
                                know.internal_points_on_edge[i].pop(j)
                                changed = True
                                print(f" => Removed point {point} from edge {i} as it lies outside the edge bounds")
                                break
                        except (ValueError, TypeError):
                            pass
        
        # Rule: if any of internal points is at a distance less than a threshold from other points, 
        # then replace them with their average, such that number of points doesn't go below 2
        if not changed:
            for i in range(num_sides):
                internal_points = know.internal_points_on_edge[i]
                if len(internal_points) < 2:
                    continue
                
                points_array = np.array(internal_points)
                to_merge = []
                merged_indices = set()

                for j in range(len(points_array)):
                    if j in merged_indices:
                        continue
                    close_points = [points_array[j]]
                    for k in range(j + 1, len(points_array)):
                        if k in merged_indices:
                            continue
                        if np.linalg.norm(points_array[j] - points_array[k]) < distance_threshold_between_points:
                            close_points.append(points_array[k])
                            merged_indices.add(k)
                    if len(close_points) > 1:
                        to_merge.append((j, close_points))

                for idx, group in to_merge:
                    if len(internal_points) - (len(group) - 1) < min_points_to_remove_outlers:
                        continue
                    average_point = tuple(np.mean(group, axis=0))
                    # Remove points by index to avoid numpy array issues
                    for point in group:
                        for j, p in enumerate(internal_points):
                            try:
                                if is_close(point, p):
                                    internal_points.pop(j)
                                    break
                            except (ValueError, TypeError):
                                pass
                    internal_points.append(average_point)
                    changed = True
                    print(f" => Merged points {group} into {average_point} on edge {i}")

        if changed:
            graph_updated = True

    return graph_updated


def generate_internal_angles(n_sides: int, 
                             equal_angles: bool = False,
                             min_angle: float = 40.0,
                             max_angle: float = 280.0,
                             random_seed: float = 47):
    """
    Generate n internal angles for a polygon summing to (n-2)*180 degrees
    
    :param n_sides: Number of angles
    :param equal_angles: If True, all angles are equal
    :param min_angle: Minimum angle value when generating random angles
    :param max_angle: Maximum angle value when generating random angles
    :param random_seed: Random seed for reproducibility
    :return: List of n internal angles in degrees
    """

    total_sum = (n_sides - 2) * 180.0

    if equal_angles:
        return [total_sum / n_sides] * n_sides

    rng = np.random.default_rng(random_seed)

    # generate random angles
    angles = rng.uniform(min_angle, max_angle, size=n_sides)

    # normalize so they sum to (n-2)*180
    scale = total_sum / angles.sum()
    angles = angles * scale

    return angles.tolist()