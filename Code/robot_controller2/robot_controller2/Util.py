import os
import subprocess
import math
from typing import List, Tuple
from robot_controller2 import Templates
from enum import Enum
import numpy as np
import copy
from scipy.spatial.transform import Rotation as Rscipy
from scipy.spatial.transform import Rotation as R

gt = "GREATER_THAN"
eq = "EQUAL"
lt = "LESS_THAN"
pre = "PRE_CONDITION"
per = "PER_CONDITION"
prev = "PREVAIL_CONDITION"
post = "POST_CONDITION"

class StateOfExecution(Enum):
    IDLE = 0
    WAITING_FOR_SERVER = 1
    EXECUTING = 2
    COMPLETED = 3
    FAILED = 4

class OrientationInput(Enum):
    """Enum for specifying desired orientation relative to direction of motion."""
    IN_DIR_MOTION = 0
    AGAINST_DIR_MOTION = 1
    RIGHT_OF_DIR_MOTION = 2
    LEFT_OF_DIR_MOTION = 3

## helper functions

def resample_line_points(points, spacing=0.01):
    """
    Given noisy 2D points roughly along a line, return evenly spaced points.

    points: (N,2) array-like
    spacing: desired spacing (same units as input, e.g. 0.01 = 1 cm)

    returns: (M,2) numpy array of resampled points
    """
    
    pts = np.asarray(points, dtype=float)

    if len(pts) < 2:
        raise ValueError("Need at least 2 points")

    mean = pts.mean(axis=0)
    centered = pts - mean

    cov = centered.T @ centered / (len(pts) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    if eigvals.max() < 1e-6:
        raise RuntimeError("Points have near-zero spread")

    direction = eigvecs[:, np.argmax(eigvals)]
    direction /= np.linalg.norm(direction)

    t = centered @ direction
    t_sorted = np.sort(t)

    t_min, t_max = t_sorted[0], t_sorted[-1]

    if t_max - t_min < spacing:
        raise RuntimeError("Point extent smaller than spacing")

    n = int(np.floor((t_max - t_min) / spacing)) + 1
    t_resampled = t_min + spacing * np.arange(n)

    return mean + np.outer(t_resampled, direction)

def unit_vector_from_points_2d(points):
    """
    Estimate a unit direction vector from noisy 2D points using PCA.

    :param points: array-like of shape (N, 2)
    :return: unit vector (vx, vy)
    """
    pts = np.asarray(points, dtype=float)

    if pts.shape[0] < 2:
        raise ValueError("Need at least 2 points")

    # center the data (remove translation)
    mean = pts.mean(axis=0)
    centered = pts - mean

    # covariance matrix
    cov = np.dot(centered.T, centered) / (len(pts) - 1)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # take eigenvector with largest eigenvalue
    direction = eigvecs[:, np.argmax(eigvals)]

    # normalize
    direction /= np.linalg.norm(direction)
    return direction[0:2]
    
def pose_from_points(points, use_ransac=False, max_iterations=100, distance_threshold=0.005, min_inliers_ratio=0.7):
    """
    Given >=3 points in 3D, compute:
      - centroid (middle point)
      - pose: position + quaternion with Z axis as normal to the points plane
    
    If use_ransac=True and len(points) > 3, RANSAC is applied to remove noise
    and fit a plane to inlier points. For 3 points, RANSAC is skipped.
    """
    pts = np.array(points)

    # RANSAC plane fitting to remove noise (only if > 3 points)
    if use_ransac and len(pts) > 3:
        min_inliers = max(3, int(min_inliers_ratio * len(pts)))  # At least min_inliers_ratio of points as inliers
        
        best_inliers = None
        best_inlier_count = 0
        
        for _ in range(max_iterations):
            # Randomly select 3 points to define a plane
            sample_indices = np.random.choice(len(pts), 3, replace=False)
            sample_points = pts[sample_indices]
            
            # Calculate plane normal from 3 points
            v1 = sample_points[1] - sample_points[0]
            v2 = sample_points[2] - sample_points[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            
            if norm < 1e-6:  # Points are collinear, skip this sample
                continue
            
            normal = normal / norm
            
            # Calculate signed distance of all points to this plane
            # Using point-to-plane distance formula: |dot(p - p0, n)|
            d = sample_points[0]  # Point on plane
            distances = np.abs(np.dot(pts - d, normal))
            
            # Find inliers (points close to the plane)
            inliers = distances < distance_threshold
            inlier_count = np.sum(inliers)
            
            # Keep best model
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
        
        # Use inliers if enough are found, otherwise use all points
        if best_inliers is not None and best_inlier_count >= min_inliers:
            pts = pts[best_inliers]
        else:
            print("Not enough inliers found, using all points for pose estimation.")
        # If not enough inliers found, keep all points
    
    # Centroid from (filtered or original) points
    centroid = np.mean(pts, axis=0)

    # Estimate normal via PCA
    cov = np.cov((pts - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, np.argmin(eigvals)]
    normal /= np.linalg.norm(normal)

    # Create local axes
    z_axis = normal

    # Pick x_axis as projection of global x
    ref = np.array([1.0, 0.0, 0.0])

    x_axis = ref - np.dot(ref, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)

    # Convert to quaternion
    quaternion = Rscipy.from_matrix(R).as_quat()

    return centroid, quaternion

def get_ccw_angle(a, b):
    """
    Returns the counterclockwise angle from vector a to b in radians,
    in the range [0, 2*pi).
    
    a, b: iterables like [x, y]
    """
    x1, y1 = a
    x2, y2 = b

    # dot product
    dot = x1 * x2 + y1 * y2

    # 2D cross product (z-component)
    cross = x1 * y2 - y1 * x2

    # signed angle (-pi, pi]
    theta = math.atan2(cross, dot)

    # normalize to [0, 2*pi)
    if theta < 0:
        theta += 2 * math.pi

    return theta

def edit_condition(input, arguments):
    if "action_name" in arguments:
        input["KINOVA_GEN3_2_RIGHT"]['action_name'] = arguments["action_name"]
    if "frame_name" in arguments:
        input["KINOVA_GEN3_2_RIGHT"]['frame_name'] = arguments["frame_name"]
    if "constraint_count" in arguments:
        input["KINOVA_GEN3_2_RIGHT"][arguments["condition_type"]]['constraint_count'] = arguments["constraint_count"]
    if "number_of_disjunctions" in arguments:
        input["KINOVA_GEN3_2_RIGHT"][arguments["condition_type"]]['number_of_disjunctions'] = arguments["number_of_disjunctions"]
    if "position" in arguments:
        # 'position' is the index of constraint in the motion specification
        cnt = 0
        for x in input["KINOVA_GEN3_2_RIGHT"][arguments["condition_type"]]['constraints']:
            cnt = cnt + 1
            if arguments['position'] == cnt:
                if "disjunction_id" in arguments:
                    input["KINOVA_GEN3_2_RIGHT"][arguments["condition_type"]]["constraints"][x] = {
                        'disjunction_id': arguments["disjunction_id"],
                        'type': arguments["type"],
                        'value': arguments["value"],
                        'operator': arguments["operator"]}
                else:
                    input["KINOVA_GEN3_2_RIGHT"][arguments["condition_type"]]["constraints"][x] = {
                        'type': arguments["type"],
                        'value': arguments["value"],
                        'operator': arguments["operator"]}
                break
    return input

def offset_points(points, edge_uv, distance, side='left'):
    """
    Offset points by a certain distance in the direction perpendicular to edge_uv.
    
    :param points: list of (x, y) tuples representing points to offset
    :param edge_uv: tuple (ux, uy) representing unit vector along the edge direction
    :param distance: how much to offset the points (same units as points)
    :param side: 'left' or 'right' to determine which perpendicular direction to use
    """
    
    t = np.asarray(edge_uv, dtype=float)
    t = t / np.linalg.norm(t)

    if side == 'left':
        n = np.array([-t[1], t[0]])
    elif side == 'right':
        n = np.array([t[1], -t[0]])
    else:
        raise ValueError("side must be 'left' or 'right'")

    return np.asarray(points) + distance * n

def resolve_orientation(dir_of_motion, orientation_input: OrientationInput) -> Tuple[List[float], float]:
    """
    Resolve desired orientation based on direction of motion and user input.
    
    :param dir_of_motion: Tuple (dx, dy) representing direction of motion in 2D plane
    :param orientation_input: User input for desired orientation relative to direction of motion
    
    Returns:
        Tuple of (quaternion [x, y, z, w], yaw angle in radians) where:
        - quaternion: [x, y, z, w] representing desired orientation for action execution
        - yaw: yaw angle extracted from the quaternion in radians
    """
    
    dx, dy = dir_of_motion
    norm = float(np.hypot(dx, dy))
    if norm == 0:
        raise ValueError("Zero direction vector")

    dx, dy = dx / norm, dy / norm

    if orientation_input == OrientationInput.IN_DIR_MOTION:
        x = np.array([dx, dy, 0])
    elif orientation_input == OrientationInput.AGAINST_DIR_MOTION:
        x = np.array([-dx, -dy, 0])
    elif orientation_input == OrientationInput.RIGHT_OF_DIR_MOTION:
        x = np.array([dy, -dx, 0])
    elif orientation_input == OrientationInput.LEFT_OF_DIR_MOTION:
        x = np.array([-dy, dx, 0])
    else:
        x = np.array([dx, dy, 0])

    z = np.array([0, 0, -1])
    y = np.cross(z, x)

    # Normalize y (important!)
    y = y / np.linalg.norm(y)

    # Recompute x to ensure orthogonality
    x = np.cross(y, z)

    rot_matrix = np.column_stack((x, y, z))
    
    # rotate rotation matrix by -90 degrees along z-axis, since the x-axis of the frame at the end-effector is at 90 degrees 
    # to base_link when camera is facing along x-axis of end-effector
    rot_z_neg_90 = R.from_euler('z', -90, degrees=True).as_matrix()
    rot_matrix = rot_matrix @ rot_z_neg_90
    
    quat = R.from_matrix(rot_matrix).as_quat()
    
    # extract yaw angle from quaternion (rotation around z-axis)
    yaw = float(R.from_quat(quat).as_euler('zyx', degrees=True)[0])
    
    # since ee is in upside down configuration, yaw angle is flipped, so we take negative of the yaw angle
    yaw = -yaw

    return quat.tolist(), yaw


def _prune_unused_constraints(template, condition_type):
    """
    Remove unused constraint placeholders from a motion specification template.
    Keeps only constraints up to the constraint_count value.
    
    :param template: Motion specification template dictionary
    :param condition_type: Type of condition to prune ('POST_CONDITION', 'PRE_CONDITION', etc.)
    """
    conditions = template["KINOVA_GEN3_2_RIGHT"].get(condition_type, {})
    if not conditions or 'constraints' not in conditions:
        return template
    
    constraints = conditions['constraints']
    constraint_count = conditions.get('constraint_count', 0)
    
    # Remove all constraints beyond constraint_count
    # Handle both int and str keys (dict keys can be either depending on how they were created)
    keys_to_remove = []
    for k in constraints.keys():
        try:
            key_value = int(k) if isinstance(k, str) else k
            if key_value > constraint_count:
                keys_to_remove.append(k)
        except (ValueError, TypeError):
            # Skip keys that can't be converted to int
            pass
    
    for key in keys_to_remove:
        del constraints[key]
    
    return template


## functions to create motion specifications for different action types

def make_action_goal_move(position, orientation, frame_name="marker_frame_0"):
    current_template = copy.deepcopy(Templates.move)
    tolerance = 0.01

    px, py, pz = position
    # Convert numpy types to Python native types
    px = float(px) if px is not None else None
    py = float(py) if py is not None else None
    pz = float(pz) if pz is not None else None
    
    xl, yl, zl = px - tolerance, py - tolerance, pz - tolerance
    xh, yh, zh = px + tolerance, py + tolerance, pz + tolerance
    
    qx, qy, qz, qw = orientation
    qx = float(qx) if qx is not None else None
    qy = float(qy) if qy is not None else None
    qz = float(qz) if qz is not None else None
    qw = float(qw) if qw is not None else None

    current_template = edit_condition(current_template, {
                              "frame_name": frame_name
                              })

    current_template = edit_condition(current_template, {
                              "condition_type": "PER_CONDITION",
                              "position": 1,
                              "type": "POSITION_XYZ",
                              "value": [px, py, pz],
                              "operator": [eq, eq, eq]
                              })

    current_template = edit_condition(current_template, {
                                "condition_type": "PER_CONDITION",
                                "position": 2,
                                "type": "ORIENTATION_QUATERNION",
                                "value": [qx, qy, qz, qw],
                                "operator": [eq, eq, eq, eq]
                              })

    current_template = edit_condition(current_template, {
                                "condition_type": "POST_CONDITION",
                                "disjunction_id": 1,
                                "position": 1,
                                "type": "POSITION_XYZ",
                                "value": [xl, yl, zl],
                                "operator": [gt, gt, gt]
                              })

    current_template = edit_condition(current_template, {
                                "condition_type": "POST_CONDITION",
                                "disjunction_id": 1,
                                "position": 2,
                                "type": "POSITION_XYZ",
                                "value": [xh, yh, zh],
                                "operator": [lt, lt, lt]
                              })

    return str(current_template)

def make_action_goal_touch(velocity, orientation, action_name="touch_neutral", frame_name="eddie_base_link", min_contact_time=1.0):
    current_template = copy.deepcopy(Templates.touch)
    velocity_x, velocity_y, velocity_z = velocity
    # Convert numpy types to Python native types
    velocity_x = float(velocity_x) if velocity_x is not None else None
    velocity_y = float(velocity_y) if velocity_y is not None else None
    velocity_z = float(velocity_z) if velocity_z is not None else None
    
    qx, qy, qz, qw = orientation
    qx = float(qx) if qx is not None else None
    qy = float(qy) if qy is not None else None
    qz = float(qz) if qz is not None else None
    qw = float(qw) if qw is not None else None

    if action_name:
        current_template = edit_condition(current_template, {
            "action_name": action_name
        })
    else:
        current_template = edit_condition(current_template, {
            "action_name": "touch_neutral"
        })

    current_template = edit_condition(current_template, {
        "frame_name": frame_name
    })

    current_template = edit_condition(current_template, {
        "condition_type": "PER_CONDITION",
        "position": 1,
        "type": "VELOCITY_XYZ",
        "value": [velocity_x, velocity_y, velocity_z],
        "operator": [eq, eq, eq]
    })
    current_template = edit_condition(current_template, {
        "condition_type": "PER_CONDITION",
        "position": 2,
        "type": "ORIENTATION_QUATERNION",
        "value": [qx, qy, qz, qw],
        "operator": [eq, eq, eq, eq]
    })
    
    vel_threshold = 0.005

    if action_name == "touch_neutral":
        current_template = edit_condition(current_template, {
            "condition_type": "POST_CONDITION",
            "number_of_disjunctions": 1,
            "disjunction_id": 1,
            "constraint_count": 2,
            "position": 1,
            "type": "VELOCITY_XYZ",
            "value": [None, None, -vel_threshold],
            "operator": [None, None, gt]
        })
        current_template = edit_condition(current_template, {
            "condition_type": "POST_CONDITION",
            "number_of_disjunctions": 1,
            "disjunction_id": 1,
            "position": 2,
            "type": "TIME_LIMIT",
            "value": min_contact_time,
            "operator": gt
        })
        
    elif action_name == "touch_edge":
        zero_vel_threshold = 0.005
        velocity_ul = zero_vel_threshold
        velocity_ll = -zero_vel_threshold
        
        current_template = edit_condition(current_template, {
            "condition_type": "POST_CONDITION",
            "number_of_disjunctions": 1,
            "disjunction_id": 1,
            "constraint_count": 3,
            "position": 1,
            "type": "VELOCITY_XYZ",
            "value": [velocity_ul, velocity_ul, velocity_ul],
            "operator": [lt, lt, lt]
        })
        current_template = edit_condition(current_template, {
            "condition_type": "POST_CONDITION",
            "number_of_disjunctions": 1,
            "disjunction_id": 1,
            "constraint_count": 3,
            "position": 2,
            "type": "VELOCITY_XYZ",
            "value": [velocity_ll, velocity_ll, velocity_ll],
            "operator": [gt, gt, gt]
        })
        current_template = edit_condition(current_template, {
            "condition_type": "POST_CONDITION",
            "disjunction_id": 1,
            "position": 3,
            "type": "TIME_LIMIT",
            "value": min_contact_time,
            "operator": gt
        })

    return str(current_template)

def make_action_goal_slide(position=[None, None, None], velocity=[None, None, None], force=[None, None, None], orientation=None, action_name="slide_to_explore_plane", frame_name="eddie_base_link", time=1.5):
    """
    position is the id of constraint to be updated
    """
    current_template = copy.deepcopy(Templates.slide)

    # Convert numpy types to Python native types
    velocity = [float(v) if v is not None else None for v in velocity]
    vel_opr_list = [eq if v is not None else None for v in velocity]
    
    position = [float(p) if p is not None else None for p in position]
    pos_opr_list = [eq if p is not None else None for p in position]
    
    force = [float(f) if f is not None else None for f in force]
    force_opr_list = [eq if f is not None else None for f in force]

    force_magnitude = np.linalg.norm([f for f in force if f is not None])
    unit_vector_of_force = [(f / force_magnitude) if (f is not None and force_magnitude > 0) else None
                            for f in force]
    velocity_spike_threshold = 0.06
    velocity_spike_threshold_vector = [float(velocity_spike_threshold * uv) if uv is not None else None for uv in unit_vector_of_force]
    velocity_spike_opr_list = [
        gt if uv is not None and uv > 0
        else lt if uv is not None and uv < 0
        else None
        for uv in unit_vector_of_force
    ]
    
    if orientation:
        qx, qy, qz, qw = orientation
        qx = float(qx) if qx is not None else None
        qy = float(qy) if qy is not None else None
        qz = float(qz) if qz is not None else None
        qw = float(qw) if qw is not None else None

    if action_name:
        current_template = edit_condition(current_template, {
            "action_name": action_name
        })

    current_template = edit_condition(current_template, {
        "frame_name": frame_name
    })

    current_template = edit_condition(current_template, {
        "condition_type": "PER_CONDITION",
        "position": 1,
        "type": "VELOCITY_XYZ",
        "value": velocity,
        "operator": vel_opr_list
    })
    
    current_template = edit_condition(current_template, {
        "condition_type": "PER_CONDITION",
        "position": 2,
        "type": "FORCE_XYZ",
        "value": force,
        "operator": force_opr_list
    })
    
    current_template = edit_condition(current_template, {
        "condition_type": "PER_CONDITION",
        "position": 3,
        "type": "POSITION_XYZ",
        "value": position,
        "operator": pos_opr_list
    })

    if orientation:
        current_template = edit_condition(current_template, {
            "condition_type": "PER_CONDITION",
            "position": 4,
            "type": "ORIENTATION_QUATERNION",
            "value": [qx, qy, qz, qw],
            "operator": [eq, eq, eq, eq]
        })
    else:
        current_template = edit_condition(current_template, {
            "condition_type": "PER_CONDITION",
            "position": 4,
            "type": "ORIENTATION_QUATERNION",
            "value": [None, None, None, None],
            "operator": [None, None, None, None]
        })
    
    if (action_name == "slide_to_explore_plane" or 
        action_name == "find_edge_by_sliding" or 
        action_name == "slide_against_surface_vector_only"):
        """
        disjunction 1: slide over surface until time limit (to explore plane)
        """
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "number_of_disjunctions": 1,
                "constraint_count": 1,
                "disjunction_id": 1,
                "position": 1,
                "type": "TIME_LIMIT",
                "value": time,
                "operator": gt
            })

    elif action_name == "slide_until_edge":
        """
        disjunction 1: slide until contact with a surface where dihedral angle=90
        disjunction 2: slide until contact with a surface where dihedral angle=270
        Note: this action is not used when reflexivity is unknown;
        """
        
        zero_vel_threshold = 0.001
        zero_vel_ul = zero_vel_threshold
        zero_vel_ll = -zero_vel_threshold
        
        # disjunction 1
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "number_of_disjunctions": 2,
                "constraint_count": 6,
                "disjunction_id": 1,
                "position": 1,
                "type": "VELOCITY_XYZ",
                "value": [zero_vel_ul, zero_vel_ul, zero_vel_ul],
                "operator": [lt, lt, lt]
            })
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 1,
                "position": 2,
                "type": "VELOCITY_XYZ",
                "value": [zero_vel_ll, zero_vel_ll, zero_vel_ll],
                "operator": [gt, gt, gt]
            })
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 1,
                "position": 3,
                "type": "POSITION_XYZ",
                "value": [None, None, -0.01],
                "operator": [None, None, gt]
            })
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 1,
                "position": 4,
                "type": "TIME_LIMIT",
                "value": 3.0,
                "operator": gt
            })

        # disjunction 2
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 2,
                "position": 5,
                "type": "VELOCITY_XYZ",
                "value": [None, None, -0.015],
                "operator": [None, None, lt]
            })
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 2,
                "position": 6,
                "type": "POSITION_XYZ",
                "value": [None, None, -0.015],
                "operator": [None, None, lt]
            })

    elif action_name == "slide_against_edge_until_corner":
        """
        disjunction 1 (against edge)    : reflexive corner
        disjunction 2 (against edge)    : non-reflexive corner
        """
        
        zero_vel_threshold = 0.001
        zero_vel_ul = zero_vel_threshold
        zero_vel_ll = -zero_vel_threshold
        
        # disjunction 1
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "number_of_disjunctions": 2,
                "constraint_count": 7,
                "disjunction_id": 1,
                "position": 1,
                "type": "VELOCITY_XYZ",
                "value": [zero_vel_ul, zero_vel_ul, zero_vel_ul],
                "operator": [lt, lt, lt]
            })
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 1,
                "position": 2,
                "type": "VELOCITY_XYZ",
                "value": [zero_vel_ll, zero_vel_ll, zero_vel_ll],
                "operator": [gt, gt, gt]
            })
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 1,
                "position": 3,
                "type": "POSITION_XYZ",
                "value": [None, None, -0.01],
                "operator": [None, None, lt]
            })
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 1,
                "position": 4,
                "type": "TIME_LIMIT",
                "value": 2.0,
                "operator": gt
            })
        
        # disjunction 2
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 2,
                "position": 5,
                "type": "VELOCITY_XYZ",
                "value": [velocity_spike_threshold_vector[0], velocity_spike_threshold_vector[1], None],
                "operator": velocity_spike_opr_list
            })
        # current_template = edit_condition(current_template, {
        #         "condition_type": "POST_CONDITION",
        #         "disjunction_id": 2,
        #         "position": 6,
        #         "type": "POSITION_XYZ",
        #         "value": [None, None, -0.01],
        #         "operator": [None, None, lt]
        #     })
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 2,
                "position": 6,
                "type": "TIME_LIMIT",
                "value": 2.0,
                "operator": gt
            })

    elif action_name == "slide_against_vertical_surface_until_corner":
        """
        disjunction 1 (against vertical surface)  : slide until edge where dihedral angle=90; non-reflexive corner;
        disjunction 2 (against vertical surface)  : slide until edge where dihedral angle=270; non-reflexive corner;
        disjunction 3 (against vertical surface)  : slide until edge where reflexive corner;
        """
        
        zero_vel_threshold = 0.001
        zero_vel_ul = zero_vel_threshold
        zero_vel_ll = -zero_vel_threshold
        
        # disjunction 1
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "number_of_disjunctions": 3,
                "constraint_count": 8,
                "disjunction_id": 1,
                "position": 1,
                "type": "VELOCITY_XYZ",
                "value": [zero_vel_ul, zero_vel_ul, zero_vel_ul],
                "operator": [lt, lt, lt]
            })
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 1,
                "position": 2,
                "type": "VELOCITY_XYZ",
                "value": [zero_vel_ll, zero_vel_ll, zero_vel_ll],
                "operator": [gt, gt, gt]
            })
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 1,
                "position": 3,
                "type": "POSITION_XYZ",
                "value": [None, None, -0.01],
                "operator": [None, None, gt]
            })
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 1,
                "position": 4,
                "type": "TIME_LIMIT",
                "value": 2.0,
                "operator": gt
            })

        # disjunction 2
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 2,
                "position": 5,
                "type": "VELOCITY_XYZ",
                "value": [None, None, -0.015],
                "operator": [None, None, lt]
            })
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 2,
                "position": 6,
                "type": "POSITION_XYZ",
                "value": [None, None, -0.01],
                "operator": [None, None, lt]
            })
        
        # disjunction 3
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 3,
                "position": 7,
                "type": "VELOCITY_XYZ",
                "value": [velocity_spike_threshold_vector[0], velocity_spike_threshold_vector[1], None],
                "operator": [gt, gt, None]
            })
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 3,
                "position": 8,
                "type": "POSITION_XYZ",
                "value": [None, None, -0.01],
                "operator": [None, None, gt]
            })
    
    # Prune unused constraint placeholders before returning
    current_template = _prune_unused_constraints(current_template, "POST_CONDITION")
    return str(current_template)


def make_action_goal_yaw(position, yaw=0.0, yaw_threshold=5.0, frame_name="eddie_base_link", time_limit=15.0):
    """
    Create a motion specification to attain a specific yaw orientation at a given position.
    Note: yaw is in degrees
    """
    
    
    current_template = copy.deepcopy(Templates.yaw)

    pos_x, pos_y, pos_z = position
    # Convert numpy types to Python native types
    pos_x = float(pos_x) if pos_x is not None else None
    pos_y = float(pos_y) if pos_y is not None else None
    pos_z = float(pos_z) if pos_z is not None else None
    yaw = float(yaw) if yaw is not None else None
    yaw_threshold = float(yaw_threshold) if yaw_threshold is not None else None
    time_limit = float(time_limit) if time_limit is not None else None

    current_template = edit_condition(current_template, {
        "frame_name": frame_name
    })
    current_template = edit_condition(current_template, {
        "condition_type": "PER_CONDITION",
        "position": 1,
        "type": "POSITION_XYZ",
        "value": [pos_x, pos_y, pos_z],
        "operator": [eq, eq, eq]
    })
    current_template = edit_condition(current_template, {
        "condition_type": "PER_CONDITION",
        "position": 2,
        "type": "ORIENTATION_YAW",
        "value": yaw,
        "operator": eq
    })

    current_template = edit_condition(current_template, {
        "condition_type": "POST_CONDITION",
        "disjunction_id": 1,
        "position": 1,
        "type": "TIME_LIMIT",
        "value": time_limit,
        "operator": gt
    })

    current_template = edit_condition(current_template, {
        "condition_type": "POST_CONDITION",
        "disjunction_id": 2,
        "position": 2,
        "type": "ORIENTATION_YAW",
        "value": yaw + yaw_threshold,
        "operator": lt
    })

    current_template = edit_condition(current_template, {
        "condition_type": "POST_CONDITION",
        "disjunction_id": 2,
        "position": 3,
        "type": "ORIENTATION_YAW",
        "value": yaw - yaw_threshold,
        "operator": gt
    })

    print(str(current_template))
    #print(json.dumps(current_template, indent=2))
    return str(current_template)