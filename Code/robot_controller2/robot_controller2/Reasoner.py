#!/home/linux/Master-thesis/Project-Code/ros2_ws/.venv/bin/python3

from rclpy.node import Node as ROSNode
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from rclpy.time import Time
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import tf2_geometry_msgs
import math
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from motion_specification_interfaces.action import MotionSpecification
from robot_controller2 import Util
from robot_controller2 import Templates
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from tf2_geometry_msgs import tf2_geometry_msgs
from geometry_msgs.msg import Vector3
import rclpy
import tf2_ros
from geometry_msgs.msg import TwistStamped
from tf2_geometry_msgs import tf2_geometry_msgs
from geometry_msgs.msg import Vector3
from tf2_ros import TransformException
from rclpy.duration import Duration
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Quaternion
from geometry_msgs.msg import Vector3
from robot_controller2 import Object_knowledge
import numpy as np
from scipy.spatial.transform import Rotation as R
import networkx as nx
import matplotlib.pyplot as plt
import time
import uuid
import re
import random
import itertools
import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
import tf_transformations
from robot_controller2 import Graph
from action_msgs.msg import GoalStatus
from rclpy.task import Future
from scipy.spatial.transform import Rotation as Rscipy


# Updated or generated variables

global current_position

global current_orientation

global current_frame

global robot_knowledge

global object_knowledge

global previous_segment

previous_segment = None

global current_segment

graph_complete = False

collected_points = []

marker_positions = []

start = True

def calculate_parallel_velocity(p1, p2, speed=0.04):
    x1, y1, z1 = p1
    x2, y2, z2 = p2


    # Vector along p1 and p2
    dx = x2 - x1
    dy = y2 - y1
    magnitude = math.sqrt(dx ** 2 + dy ** 2)
    if magnitude < 1e-9:
        return (0.0, 0.0, 0.0)

    ux = dx / magnitude
    uy = dy / magnitude

    FLATTEN_FACTOR = 1.0
    vx = ux * speed
    vy = uy * speed * FLATTEN_FACTOR


    return (vx, vy, -0.03)



def cartesian_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def resolve_point_coords(graph, point_id):
    """Extracts [x, y] coordinates from a point node if available."""
    node = graph.nodes.get(point_id, {})
    if node.get('type') not in {'corner', 'contact_point'}:
        return None
    coords = node.get('data', [])
    if isinstance(coords, list) and len(coords) >= 2:
        return coords[:2]
    return None

def resolve_segment_info(graph, segment_id):
    """Returns corner_ids, length of a segment node."""
    node = graph.nodes.get(segment_id, {})
    corners = node.get('corners', [])
    length = node.get('length')
    return corners, length

def match_nodes(tolerance=1e-3):
    global object_knowledge, robot_knowledge

    segment_matches = {}
    deferred_segments = []

    # Try matching robot_knowledge segments to OKG segments
    for b_seg_id, b_seg_data in robot_knowledge.nodes(data=True):

        if b_seg_data.get("type") != "segment":
            continue

        b_corners, b_length = resolve_segment_info(robot_knowledge, b_seg_id)
        if b_length is None:
            continue

        # Gather contact points lying on this segment
        contact_coords = []
        for neighbor in robot_knowledge.neighbors(b_seg_id):
            if robot_knowledge.nodes[neighbor].get("type") == "contact_point":
                coords = resolve_point_coords(robot_knowledge, neighbor)
                if coords:
                    contact_coords.append(coords)

        max_contact_distance = 0.0
        if len(contact_coords) >= 2:
            max_contact_distance = max(
                cartesian_distance(a, b)
                for i, a in enumerate(contact_coords)
                for j, b in enumerate(contact_coords)
                if i < j
            )

        candidates = []
        for o_seg_id, o_seg_data in object_knowledge.nodes(data=True):
            if o_seg_data.get("type") != "segment":
                #print("Node not segment")
                continue

            o_corners, o_length = resolve_segment_info(object_knowledge, o_seg_id)
            if o_length is None:
                #print("Node contains no length information")
                continue

            # Length check
            if abs(o_length - b_length) > tolerance:
                #print("Matching length")
                continue

            # Contact point constraint
            if max_contact_distance and max_contact_distance > o_length + tolerance:
                continue

            # 2D edge angle check
            b_angle = None
            for neighbor in robot_knowledge.neighbors(b_seg_id):
                n_data = robot_knowledge.nodes[neighbor]
                if n_data.get("type") == "2d_edge_angle":
                    b_angle = n_data.get("angle")
                    break

            if b_angle is not None:
                o_angle = None
                for neighbor in object_knowledge.neighbors(o_seg_id):
                    n_data = object_knowledge.nodes[neighbor]
                    if n_data.get("type") == "2d_edge_angle":
                        o_angle = n_data.get("angle")
                        break

                if o_angle is None or abs(o_angle - b_angle) > tolerance:
                    continue

            candidates.append((o_seg_id, o_corners))

        if len(candidates) == 1:
            matched_id = candidates[0][0]
            robot_knowledge.nodes[b_seg_id]["matched_id"] = matched_id
            object_knowledge.nodes[matched_id]["matched_id"] = b_seg_id
            segment_matches[b_seg_id] = matched_id

        else:
            deferred_segments.append(b_seg_id)

def move_towards_origin(point, distance=0.1):
    """
    Move a point towards the origin in its own frame by a fixed distance.
    """
    x, y, z = point

    # Direction vector toward origin
    dx = -x
    dy = -y

    length = math.hypot(dx, dy)

    # Normalize direction and scale by desired movement
    move_x = dx / length * distance
    move_y = dy / length * distance

    return [x + move_x, y + move_y, z]

def check_point_integrity(new_point):
    """
    Check if the given point lies on the same edge
    as any segment in the global robot_knowledge graph, but
    only for segments with at least two existing contact points.
    """
    global robot_knowledge

    px, py = new_point[0], new_point[1]

    for seg_id, seg_data in robot_knowledge.nodes(data=True):
        if seg_data.get("type") != "segment":
            continue

        contact_points = []
        for neighbor in robot_knowledge.neighbors(seg_id):
            if robot_knowledge.nodes[neighbor].get("type") == "contact_point":
                coords = resolve_point_coords(robot_knowledge, neighbor)
                if coords:
                    contact_points.append((coords[0], coords[1]))

        # Only check if we have at least two contact points
        if len(contact_points) < 2:
            continue

        # Get the two reference points
        (x1, y1), (x2, y2) = contact_points[:2]

        # Check collinearity using cross product
        area = abs((px - x1) * (y2 - y1) - (py - y1) * (x2 - x1))
        if area > 1e-6:
            continue

        # Check if point lies within segment bounds (not just on infinite line)
        min_x, max_x = sorted([x1, x2])
        min_y, max_y = sorted([y1, y2])
        if (min_x - 1e-6 <= px <= max_x + 1e-6) and (min_y - 1e-6 <= py <= max_y + 1e-6):
            return seg_id

    return None



def ee_velocity_with_angle(ee_pos, angle=0.0, speed=0.02):
    """
    Compute a 2D XY velocity for the end-effector, pointing toward the tf origin but rotated by a given angle.
    """
    ee_xy = np.array(ee_pos[:2])
    direction = -ee_xy  # vector from end-effector to origin
    norm = np.linalg.norm(direction) + 1e-9
    if norm == 0:
        return [0.0, 0.0, 0.0]

    # Normalize
    direction = direction / norm

    # Rotation matrix
    rot = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

    # Apply rotation
    rotated_dir = rot @ direction
    vx, vy = rotated_dir * speed

    return [float(vx), float(vy), -0.006]

def flip_dominant(v):
    vx, vy, vz = v

    if abs(vx) > abs(vy):
        # x is dominant
        vx = -vx
    else:
        # y is dominant
        vy = -vy

    return [vx, vy, vz]


def line_intersection(p1, p2, q1, q2):
    """
    Case 1: Intersection of two lines defined by two points each.
    p1, p2: points on line 1
    q1, q2: points on line 2
    Returns intersection point (x,y) or None if parallel.
    """
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    x3, y3 = q1[:2]
    x4, y4 = q2[:2]

    # Solve determinant
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-10:
        return None  # parallel lines

    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    return [px, py, 0.0]


def line_intersection_with_angle(p1, p2, q1, angle_deg, clockwise=False):
    """
    Case 2: Intersection of line defined by (p1,p2) with another line
    passing through q1 at a given angle relative to (p1,p2).
    """
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    dx, dy = x2-x1, y2-y1

    # Normalize direction of first line
    d = np.array([dx, dy], dtype=float)
    d /= np.linalg.norm(d)

    # Build rotation matrix
    theta = np.deg2rad(angle_deg)
    if clockwise:
        theta = -theta

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    d2 = R @ d  # rotated direction for edge 2

    # Intersection of line p1 p2 with line q1 + t*d2
    return line_intersection(p1, p2, q1, (q1[0] + d2[0], q1[1] + d2[1]))


def pose_from_points(points):
    """
    Given >=3 points in 3D, compute:
      - centroid (middle point)
      - pose: position + quaternion with Z axis as normal to the points plane
    """
    pts = np.array(points)

    # Centroid
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


def pose_from_points_end(points, idx_x=(1,5)):
    global robot_knowledge
    """
    Compute pose from points:
      - x-axis points exactly from points[idx_x[0]] to points[idx_x[1]]
      - z-axis is plane normal 
      - y-axis = z x x to make right-handed frame
    """

    points = [robot_knowledge.nodes["pt_0"]["data"], robot_knowledge.nodes["pt_1"]["data"], robot_knowledge.nodes["pt_2"]["data"], robot_knowledge.nodes["pt_3"]["data"]]
    pts = np.array(points)
    if pts.shape[0] < 3:
        raise ValueError("Need at least 3 points")

    # Centroid
    centroid = np.mean(pts, axis=0)

    # Plane normal via PCA
    cov = np.cov((pts - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    z_axis = eigvecs[:, np.argmin(eigvals)]
    z_axis /= np.linalg.norm(z_axis)

    # X axis exactly between the two points
    x_axis = pts[0] - pts[1]
    x_axis /= np.linalg.norm(x_axis)

    # Make z perpendicular to x (keep plane normal component)
    z_axis = z_axis - np.dot(z_axis, x_axis) * x_axis
    z_axis /= np.linalg.norm(z_axis)

    # Y axis for right-handed frame
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Recompute z to ensure exact orthonormality
    z_axis = np.cross(x_axis, y_axis)

    # Rotation matrix: columns = [x, y, z]
    R_mat = np.column_stack((x_axis, y_axis, z_axis))
    quaternion = R.from_matrix(R_mat).as_quat()

    return centroid, quaternion

def offset_edge(p1, p2, d=0.03, outward=False):
    """
    Compute a parallel offset of an edge by distance d in XY plane.
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    v = p2 - p1
    v[2] = 0.0  # ensure in XY
    norm = np.linalg.norm(v[:2])

    # unit normal (rotate v by +90)
    n = np.array([-v[1], v[0], 0.0]) / norm
    if outward:
        n = -n

    p1_off = p1 + d * n
    p2_off = p2 + d * n

    p1_off = [float(x) for x in p1_off]
    p2_off = [float(x) for x in p2_off]
    return p1_off, p2_off

def move_from_edge(corner, slope_point, corner_angle, length):
    x0, y0 = corner[0], corner[1]
    x1, y1 = slope_point[0], slope_point[1]

    # Edge vector
    vx = x1 - x0
    vy = y1 - y0

    # Normalize
    mag = math.sqrt(vx**2 + vy**2)
    vx /= mag
    vy /= mag

    # Convert angle to radians
    # negative for clockwise rotation
    theta = math.radians(-corner_angle)

    # Rotate vector
    rx = vx * math.cos(theta) - vy * math.sin(theta)
    ry = vx * math.sin(theta) + vy * math.cos(theta)

    # Scale and translate
    new_x = x0 + rx * length
    new_y = y0 + ry * length
    new_z = corner[2]

    return [new_x, new_y, new_z]



class ReasonerNode(ROSNode):

    def __init__(self):

        super().__init__('reasoner_node')

        self.client = ActionClient(self, MotionSpecification, 'motion_specification')

        self.publisher_marker = self.create_publisher(PoseStamped, '/marker_position', 10)

        self.publisher_corner = self.create_publisher(PoseStamped, '/corner_position', 10)

        self.publisher_plane = self.create_publisher(PoseStamped, '/plane_position', 10)

        self.tf_ready = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.subscription = self.create_subscription(
            PoseStamped,
            '/ee_pose',
            self.ee_callback,
            10
        )

        self.main_loop()


    def main_loop(self):

        global graph_complete, robot_knowledge, object_knowledge, previous_segment, current_segment, current_position, current_orientation

        object_knowledge = Graph.create_graph_from_json(Object_knowledge.rectangle_polygon)

        robot_knowledge = Graph.create_simplified_graph_from_json(Object_knowledge.rectangle_polygon)

        Graph.update_belief_state_visualization(robot_knowledge)

        self.send_goal(Templates.neutral)

        # Main loop: Perform actions until the graph has been reconstructed
        while not graph_complete:

            # Match the current elements in the BSG
            match_nodes()

            # Calculate corners and fill in gaps
            self.update_robot_knowledge()

            #Generate new action list to execute
            action_list = self.generate_new_action_list()

            # Update the Visualization of the BSG
            Graph.update_belief_state_visualization(robot_knowledge)

            for action in action_list:
                time.sleep(1.0)

                result = self.send_goal(action)

                if result:
                    self.get_logger().info("Action succeeded.")
                    self.action_handler(result)
                else:
                    self.get_logger().info("Action not successfull, retrying.")
                    self.send_goal(action)

        self.get_logger().inf("Plan successfully executed!")

        return


    def register_contact(self):
        global robot_knowledge, object_knowledge, current_segment, previous_segment

        existing_segment = check_point_integrity(current_position)
        if existing_segment:
            # Dont add a new node
            previous_segment = current_segment
            current_segment = existing_segment
            return

        else:
            num_seg_okg = len(Graph.get_all_corners(object_knowledge))
            num_seg_belief = len(Graph.get_all_corners(robot_knowledge))
            if not num_seg_okg == num_seg_belief or num_seg_belief is None:
                # No matching segment, create a new one
                seg_id = Graph.add_new_line_segment(robot_knowledge)
                if num_seg_belief is 0:
                    previous_segment = seg_id
                else:
                    previous_segment = current_segment
                current_segment = seg_id

                num_okg = len(Graph.get_all_corners(object_knowledge))
                num_belief = len(Graph.get_all_corners(robot_knowledge))

                if num_okg == num_belief + 1:
                    # Add the last corner
                    corner_id = Graph.add_new_corner(robot_knowledge, object_knowledge, seg_id)

                    robot_knowledge.add_edge(corner_id, current_segment)
                    robot_knowledge.add_edge(corner_id, "line_segment_0")
                    robot_knowledge.add_edge(corner_id, "polygon_0")
                elif num_okg <= num_belief + 1:
                    print("Already has maximum amount of nodes !")



        Graph.update_belief_state_visualization(robot_knowledge)



    def action_handler(self, result):
        global robot_knowledge, object_knowledge, current_segment, previous_segment

        result_indx= result.disjunction_indices[0]

        if result.ms_action_name == "slide_plane_single":

            self.register_contact()

            if result_indx == 1 or result_indx == 3:
                print("OFF THE EDGE")

                # Automatically add a 2d_corner_angle as all corners should have one
                angle_node_id = Graph.get_next_2d_edge_angle_id(robot_knowledge)

                robot_knowledge.add_node(angle_node_id, type="2d_edge_angle")
                robot_knowledge.add_edge(current_segment, angle_node_id)

                robot_knowledge.nodes[angle_node_id]["angle"] = 90.0

                edge_velocity = ee_velocity_with_angle(current_position, 0.0)

                Graph.update_belief_state_visualization(robot_knowledge)

                if self.send_goal(Util.make_action_goal_touch(edge_velocity, current_orientation, current_position,
                                                              frame_name="marker_frame_0")) is not None:

                    id = Graph.get_next_contact_point_id(robot_knowledge)
                    robot_knowledge.add_node(id, type="contact_point", data=current_position)
                    robot_knowledge.add_edge(current_segment, id)
                    previous_segment = current_segment
                    Graph.update_belief_state_visualization(robot_knowledge)

            if result_indx == 2:
                print("HIT A WALL")
                # Automatically add a 2d_corner_angle as all corners should have one
                angle_node_id = Graph.get_next_2d_edge_angle_id(robot_knowledge)

                robot_knowledge.add_node(angle_node_id, type="2d_edge_angle")
                robot_knowledge.add_edge(current_segment, angle_node_id)

                robot_knowledge.nodes[angle_node_id]["angle"] = 270.0

                edge_velocity = ee_velocity_with_angle(current_position, 0.0)

                touch_velocity = edge_velocity

                touch_velocity[:1] = [-v for v in edge_velocity[:1]]

                Graph.update_belief_state_visualization(robot_knowledge)

                if self.send_goal(Util.make_action_goal_touch(touch_velocity, current_orientation, current_position,
                                                              frame_name="marker_frame_0")) is not None:
                    id = Graph.get_next_contact_point_id(robot_knowledge)
                    robot_knowledge.add_node(id, type="contact_point", data=current_position)
                    robot_knowledge.nodes[id]["geometry"] = "wall"
                    robot_knowledge.add_edge(current_segment, id)
                    previous_segment = current_segment
                    Graph.update_belief_state_visualization(robot_knowledge)

            return

        elif result.ms_action_name == "slide_blind":
            global marker_positions, collected_points

            marker_positions.append(current_position)

            if len(marker_positions) < 4:
                return
            else:
                position, orientation = pose_from_points(collected_points)
                self.send_marker(position=position, orientation=orientation, marker_type="marker", frame="eddie_base_link")

            return

        elif result.ms_action_name == "touch_neutral":
            return

        elif result.ms_action_name == "touch_contact":

            id = Graph.get_next_contact_point_id(robot_knowledge)
            robot_knowledge.add_node(id, type="contact_point", data=current_position)
            robot_knowledge.add_edge(current_segment, id)

            self.send_marker(position=current_position, frame="marker_frame_0")

            return

        elif result.ms_action_name == "slide_plane_multiple":

            self.register_contact()

            result = result.disjunction_indices[0]

            if result == 1 or result == 3:

                print("Off the edge!")
                # Automatically add a 2d_corner_angle as all corners should have one
                angle_node_id = Graph.get_next_2d_edge_angle_id(robot_knowledge)

                robot_knowledge.add_node(angle_node_id, type="2d_edge_angle")
                robot_knowledge.add_edge(current_segment, angle_node_id)

                robot_knowledge.nodes[angle_node_id]["angle"] = 90.0

                Graph.update_belief_state_visualization(robot_knowledge)

                edge_velocity = ee_velocity_with_angle(current_position, 0.0)

                # IF there are already two contact points on the current segment then do not slide
                old_position = current_position

                if self.send_goal(Util.make_action_goal_touch(edge_velocity, current_orientation, current_position,
                                                              frame_name="marker_frame_0")) is not None:

                    while Graph.count_contact_points(robot_knowledge, current_segment) < 2:

                        new_position = current_position
                        angle = -0.45
                        while cartesian_distance(new_position, old_position) < 0.035:
                            print(cartesian_distance(new_position, old_position))
                            collected_points = []
                            edge_velocity = ee_velocity_with_angle(current_position, angle)
                            self.send_goal(Util.make_action_goal_slide(edge_velocity, current_orientation, mode="slide_edge",
                                                                       frame_name="marker_frame_0"))
                            angle = angle - 0.3
                            new_position = current_position


                        id = Graph.get_next_contact_point_id(robot_knowledge)
                        robot_knowledge.add_node(id, type="contact_point", data=current_position)
                        robot_knowledge.add_edge(current_segment, id)
                        old_position = current_position
                        previous_segment = current_segment
                        Graph.update_belief_state_visualization(robot_knowledge)

            if result == 2:
                print("Hit a wall")

                # Automatically add a 2d_corner_angle as all corners should have one
                angle_node_id = Graph.get_next_2d_edge_angle_id(robot_knowledge)

                robot_knowledge.add_node(angle_node_id, type="2d_edge_angle")
                robot_knowledge.add_edge(current_segment, angle_node_id)

                robot_knowledge.nodes[angle_node_id]["angle"] = 270.0

                Graph.update_belief_state_visualization(robot_knowledge)

                edge_velocity = ee_velocity_with_angle(current_position, 0.0)

                touch_velocity = flip_dominant(edge_velocity)[:2]

                touch_velocity.append(0.0)

                old_position = current_position

                if self.send_goal(
                        Util.make_action_goal_yaw([current_position[0], current_position[1], current_position[2]],
                                                  yaw=-90, yaw_threshold=0.1,
                                                 frame_name="marker_frame_0", time_limit=10.0)) and self.send_goal(Util.make_action_goal_touch(touch_velocity, current_orientation, current_position,
                                                frame_name="marker_frame_0")) is not None:

                    while Graph.count_contact_points(robot_knowledge, current_segment) < 2:

                        new_position = current_position
                        angle = 0.45
                        while cartesian_distance(new_position, old_position) < 0.035:
                            collected_points = []
                            edge_velocity = ee_velocity_with_angle(current_position, angle)

                            # Inverse the velocity
                            edge_velocity = flip_dominant(edge_velocity)[:2]

                            edge_velocity.append(0.0)

                            self.send_goal(Util.make_action_goal_slide(edge_velocity, current_orientation, mode="slide_edge",
                                                                       frame_name="marker_frame_0"))
                            angle = angle + 0.3
                            new_position = current_position

                        id = Graph.get_next_contact_point_id(robot_knowledge)
                        robot_knowledge.add_node(id, type="contact_point", data=current_position)
                        robot_knowledge.nodes[id]["geometry"] = "wall"
                        robot_knowledge.add_edge(current_segment, id)
                        old_position = current_position
                        previous_segment = current_segment
                        Graph.update_belief_state_visualization(robot_knowledge)

            return

        return


    def generate_new_action_list(self):
        global robot_knowledge, object_knowledge, current_segment, previous_segment, start

        def no_knowledge():
            """
            Returns True if the nx.Graph contains no relevant information,
            False otherwise.
            """
            for node, data in object_knowledge.nodes(data=True):
                node_type = data.get("type")

                # Check segment lengths
                if node_type == "segment":
                    length_value = object_knowledge.nodes[node].get("length")
                    if length_value is not None:
                        return False

                # Check 2D corner angles
                if node_type == "2d_corner_angle":
                    angle_value = object_knowledge.nodes[node].get("angle")
                    if angle_value is not None:
                        return False

                # Check 2D edge angles
                if node_type == "2d_edge_angle":
                    angle_value = object_knowledge.nodes[node].get("value")
                    if angle_value is not None:
                        return False

            return True


        if start:
            action_list = [
            Util.make_action_goal_slide([0.03, 0.0, -0.02], current_orientation, mode="slide_blind", frame_name="eddie_base_link", time=5.0),
            Util.make_action_goal_slide([0.0, 0.03, -0.02], current_orientation, mode="slide_blind", frame_name="eddie_base_link", time=5.0),
            Util.make_action_goal_slide([-0.03, 0.0, -0.02], current_orientation, mode="slide_blind", frame_name="eddie_base_link", time=5.0),
            Util.make_action_goal_slide([0.0, -0.03, -0.02], current_orientation, mode="slide_blind", frame_name="eddie_base_link", time=5.0),
            ]
            start = False
            return action_list


        # Hardcoded start list. This will always be done no matter what.
        velocity = [0.06, 0.0, -0.04]
        rpy = Util.quat_vel_to_rpy(velocity)
        orientation = Util.calculate_quaternion(rpy[0], rpy[1], rpy[2])

        if list(robot_knowledge.nodes) == ["polygon_0"]:
            action_list = [
            Util.make_action_goal_yaw([0.0, 0.0, 0.0], yaw=+90.0, yaw_threshold=0.1,
                                          frame_name="marker_frame_0", time_limit=15.0),
            Util.make_action_goal_slide([0.06, 0.0, -0.04], orientation, mode="slide_plane_multiple", frame_name="marker_frame_0")
            ]
            return action_list

        
        # No Information in the OKG, explore in order. Always gets two contact point on every edge.
        if no_knowledge():
            print("No Knowledge")
            if Graph.count_contact_points(robot_knowledge, current_segment) >= 2:
                p1, p2 = Graph.get_two_contact_points_data(robot_knowledge, current_segment)
                move_position = move_towards_origin(Graph.get_one_contact_point_data(robot_knowledge, current_segment))
                slide_velocity = calculate_parallel_velocity(p1, p2)
                slide_orientation = Util.quat_vel_to_rpy(slide_velocity)
                slide_yaw = Util.quat_vel_to_rpy_yaw(slide_velocity)
                slide_orientation = Util.calculate_quaternion(slide_orientation[0], slide_orientation[1],
                                                              slide_orientation[2])
                move_position_x, move_position_y, move_position_z = move_position
                action_list = [
                    Util.make_action_goal_move([None, None, 0.15], current_orientation, current_position,
                                               "marker_frame_0"),
                    # str(Templates.reset),
                    Util.make_action_goal_move([move_position_x, move_position_y, 0.15],
                                               current_orientation, current_position,
                                               "marker_frame_0"),
                    Util.make_action_goal_yaw([move_position_x, move_position_y, 0.1], yaw=slide_yaw, yaw_threshold=0.1,
                                              frame_name="marker_frame_0", time_limit=15.0),

                    Util.make_action_goal_touch([0.0, 0.0, -0.02], orientation=slide_orientation, current_position=current_position,
                                                frame_name="marker_frame_0"),
                    Util.make_action_goal_slide(slide_velocity, slide_orientation, mode="slide_plane_multiple",
                                                frame_name="marker_frame_0")]
                return action_list
        else:
            # There is information and the next best action has to be chosen. Still explores in one direction.
            # Get all line segments
            line_segments = Graph.get_all_segments(robot_knowledge)

            # Pick highest indexed segment
            highest_segment = line_segments[-1]
            seg_data = robot_knowledge.nodes[highest_segment]

            # Find corner neighbors of this line segment
            corner_neighbors = [
                nbr for nbr in robot_knowledge.neighbors(highest_segment)
                if robot_knowledge.nodes[nbr].get("type") == "corner"
            ]

            # Pick highest indexed corner neighbor
            highest_corner = max(corner_neighbors, key=lambda n: int(str(n).split("_")[-1]))

            # Check if this corner has a neighbor of type 2d_corner_angle
            angle_neighbors = [
                nbr for nbr in robot_knowledge.neighbors(highest_corner)
                if robot_knowledge.nodes[nbr].get("type") == "2d_corner_angle"
            ]

            has_angle_value = False
            if angle_neighbors:
                # Take the first one (or check all if multiple possible)
                angle_node = angle_neighbors[0]
                print(robot_knowledge.nodes[angle_node].get("angle"))
                has_angle_value = (
                        robot_knowledge.nodes[angle_node].get("angle") is not None
                )

            if has_angle_value: #has_segment_value and has_angle_value:
                # Generate all variables here
                p1, p2 = Graph.get_two_contact_points_data(robot_knowledge, current_segment) #Not current segment but highest indexed segment fulfilling the requirements
                move_position = move_towards_origin(Graph.get_one_contact_point_data(robot_knowledge, current_segment))
                slide_velocity = calculate_parallel_velocity(p1, p2)
                slide_velocity = [float(item) for item in slide_velocity]
                slide_orientation = Util.quat_vel_to_rpy(slide_velocity)
                slide_yaw = Util.quat_vel_to_rpy_yaw(slide_velocity)
                slide_orientation = Util.calculate_quaternion(slide_orientation[0], slide_orientation[1],
                                                              slide_orientation[2])
                move_position_x, move_position_y, move_position_z = move_position
                action_list = [
                    Util.make_action_goal_move([None, None, 0.15], current_orientation, current_position,
                                               "marker_frame_0"),
                    Util.make_action_goal_move([move_position_x, move_position_y, 0.15],
                                               current_orientation, current_position,
                                               "marker_frame_0"),
                    Util.make_action_goal_yaw([move_position_x, move_position_y, 0.1], yaw=slide_yaw, yaw_threshold=0.1,
                                              frame_name="marker_frame_0", time_limit=15.0),

                    Util.make_action_goal_touch([0.0, 0.0, -0.04], slide_orientation, current_position,
                                                frame_name="marker_frame_0"),
                    Util.make_action_goal_slide(slide_velocity, slide_orientation, mode="slide_plane_single",
                                                frame_name="marker_frame_0")]

            else:
                print("Generating multiple points")
                # Generate all variables here
                p1, p2 = Graph.get_two_contact_points_data(robot_knowledge, current_segment)
                move_position = move_towards_origin(Graph.get_one_contact_point_data(robot_knowledge, current_segment))
                slide_velocity = calculate_parallel_velocity(p1, p2)
                slide_orientation = Util.quat_vel_to_rpy(slide_velocity)
                slide_yaw = Util.quat_vel_to_rpy_yaw(slide_velocity)
                slide_orientation = Util.calculate_quaternion(slide_orientation[0], slide_orientation[1],
                                                              slide_orientation[2])
                move_position_x, move_position_y, move_position_z = move_position
                action_list = [
                    Util.make_action_goal_move([None, None, 0.15], current_orientation, current_position,
                                               "marker_frame_0"),
                    Util.make_action_goal_move([move_position_x, move_position_y, 0.15],
                                               current_orientation, current_position,
                                               "marker_frame_0"),
                    Util.make_action_goal_yaw([move_position_x, move_position_y, 0.1], yaw=slide_yaw, yaw_threshold=0.1,
                                              frame_name="marker_frame_0", time_limit=15.0),

                    Util.make_action_goal_touch([0.0, 0.0, -0.04], slide_orientation, current_position,
                                                "marker_frame_0"),
                    Util.make_action_goal_slide(slide_velocity, slide_orientation, mode="slide_plane_multiple",
                                                frame_name="marker_frame_0")]


        return action_list



    def update_robot_knowledge(self):
        global robot_knowledge, object_knowledge, graph_complete

        def compensate_contact_points(d=0.04, outward=True):
            """For each segment in robot_knowledge, if it has exactly two contact_point
            neighbors, offset them using offset_edge and mark them compensated."""
            global robot_knowledge

            for node_id, data in robot_knowledge.nodes(data=True):
                if data.get("type") != "segment":
                    continue

                neighbors = list(robot_knowledge.neighbors(node_id))
                contact_points = [
                    (n_id, robot_knowledge.nodes[n_id])
                    for n_id in neighbors
                    if robot_knowledge.nodes[n_id].get("type") == "contact_point"
                ]
                print(contact_points)

                if len(contact_points) < 2:
                    print("NOT ENOUGH CONTACT POINTS")
                    continue

                (cp1_id, cp1_data), (cp2_id, cp2_data) = contact_points

                if cp1_data.get("compensated") and cp2_data.get("compensated"):
                    continue

                try:
                    p1 = cp1_data["data"]
                    p2 = cp2_data["data"]
                except Exception as e:
                    print(f"Could not parse contact point data for {cp1_id}, {cp2_id}: {e}")
                    continue

                if cp1_data.get("geometry") == "wall" and cp2_data.get("geometry") == "wall":

                    try:
                        p1_off, p2_off = offset_edge(p1, p2, outward=True)
                    except ValueError as e:
                        continue

                else:
                    try:
                        p1_off, p2_off = offset_edge(p1, p2, outward=False)
                    except ValueError as e:
                        continue

                robot_knowledge.nodes[cp1_id]["data"] = list(p1_off)
                robot_knowledge.nodes[cp2_id]["data"] = list(p2_off)
                robot_knowledge.nodes[cp1_id]["compensated"] = True
                robot_knowledge.nodes[cp2_id]["compensated"] = True

                self.send_marker(position=p1_off, marker_type="marker", frame="marker_frame_0")
                self.send_marker(position=p2_off, marker_type="marker", frame="marker_frame_0")


        def assign_segment_lengths():
            """Assign missing lengths from object_knowledge (if all equal) or
            compute from two corner coordinates when available."""
            segment_lengths = [
                node_data["length"]
                for _, node_data in object_knowledge.nodes.items()
                if node_data.get("type") == "segment" and "length" in node_data
            ]
            all_same_length = len(segment_lengths) > 0 and all(l == segment_lengths[0] for l in segment_lengths)

            for seg_id, seg_data in robot_knowledge.nodes(data=True):
                if seg_data.get("type") != "segment":
                    continue

                corners_with_data = [
                    n for n in robot_knowledge.neighbors(seg_id)
                    if robot_knowledge.nodes[n].get("type") == "corner"
                       and isinstance(robot_knowledge.nodes[n].get("data"), (list, tuple))
                       and len(robot_knowledge.nodes[n]["data"]) == 3
                ]
                if len(corners_with_data) == 2:
                    p1 = robot_knowledge.nodes[corners_with_data[0]]["data"]
                    p2 = robot_knowledge.nodes[corners_with_data[1]]["data"]
                    robot_knowledge.nodes[seg_id]["length"] = cartesian_distance(p1, p2)

                if all_same_length:
                    if segment_lengths and segment_lengths[0] is not None:
                        robot_knowledge.nodes[seg_id]["length"] = segment_lengths[0].nodes[seg_id]["length"] = \
                        segment_lengths[0]

        def publish_corner(corner_id, pos):
            robot_knowledge.nodes[corner_id].update(data=pos)
            self.send_marker(position=pos, marker_type="corner", frame="marker_frame_0")


        def resolve_corners():
            segments = Graph.get_all_segments(robot_knowledge)

            # Resolve all corners where there are enough contact points and the corner does not contain data.
            for seg_a, seg_b in itertools.combinations(segments, 2):
                corners = [
                    n for n in robot_knowledge.neighbors(seg_a)
                    if robot_knowledge.nodes[n].get("type") == "corner" and n in robot_knowledge.neighbors(seg_b)
                ]

                for corner in corners:
                    # Skip only if this corner already has coordinates
                    if robot_knowledge.nodes[corner].get("data"):
                        continue

                    points_a = Graph.get_two_contact_points_data(robot_knowledge, seg_a)
                    points_b = Graph.get_two_contact_points_data(robot_knowledge, seg_b)

                    print("points a and b", points_a, points_b)
                    if not points_a or not points_b:
                        continue

                    corner_pos = None
                    points_a = [p for p in points_a if p]
                    points_b = [p for p in points_b if p]

                    if len(points_a) == 2 and len(points_b) == 2:
                        corner_pos = line_intersection(*points_a, *points_b)

                    elif (len(points_a) == 2 and len(points_b) == 1) or (len(points_a) == 1 and len(points_b) == 2):
                        if len(points_a) == 2:
                            p1, p2 = points_a
                            q1 = points_b[0]
                            segment = seg_b
                        else:
                            p1, p2 = points_b
                            q1 = points_a[0]
                            segment = seg_a

                        angle = None
                        for neighbor in robot_knowledge.neighbors(corner):
                            node = robot_knowledge.nodes.get(neighbor, {})
                            if node.get("type") == "2d_corner_angle" and "angle" in node:
                                angle = node["angle"]
                                break
                        if angle:

                            corner_pos = line_intersection_with_angle(p1, p2, q1, angle)

                            contact_point = [
                                n for n in robot_knowledge.neighbors(segment)
                                if robot_knowledge.nodes[n].get("type") == "contact_point"
                            ]

                            if robot_knowledge.nodes[contact_point[0]].get("geometry") == "wall":

                                try:
                                    corner_off, p_off = offset_edge(corner_pos, q1, outward=True)
                                except ValueError as e:
                                    continue

                            else:
                                try:
                                    corner_off, p_off = offset_edge(corner_pos, q1, outward=False)
                                except ValueError as e:
                                    continue

                            robot_knowledge.nodes[contact_point[0]]["data"] = p_off

                            self.send_marker(position=p_off, frame="marker_frame_0")

                            corner_pos = corner_off

                    if corner_pos:
                        publish_corner(corner, corner_pos)


        def forward_resolve_nodes():
            global robot_knowledge, object_knowledge

            if len(Graph.get_all_segments(robot_knowledge)) == 0:
                return

            robot_current = "line_segment_0"
            # use matched_id here
            object_current = robot_knowledge.nodes[robot_current].get("matched_id", robot_current)

            prev_robot_line = None
            prev_object_line = None
            prev_pt = None

            collected_contact_points = []
            collected_pts = []
            collected_lengths = []

            while True:

                robot_node = robot_knowledge.nodes[robot_current]
                object_node = object_knowledge.nodes[object_current]

                # Copy missing angle neighbors
                for neighbor in object_knowledge.neighbors(object_current):
                    # copy corner angles for pt nodes
                    if robot_current.startswith("pt_"):
                        idx = int(robot_current.split("_")[-1])
                        corner = f"corner_angle_{idx}"

                        if corner not in robot_knowledge:
                            # Copy from object_knowledge
                            robot_knowledge.add_node(corner, **object_knowledge.nodes[corner])
                            robot_knowledge.add_edge(robot_current, corner)
                        else:
                            # Ensure edge exists only once
                            if not robot_knowledge.has_edge(robot_current, corner):
                                robot_knowledge.add_edge(robot_current, corner)

                    # copy edge angles for line segments
                    if robot_current.startswith("line_segment_") and neighbor.startswith("edge_angle_"):
                        if neighbor not in robot_knowledge:
                            robot_knowledge.add_node(neighbor, **object_knowledge.nodes[neighbor])
                            robot_knowledge.add_edge(robot_current, neighbor)

                # Stop when we hit pt with no data
                if robot_current.startswith("pt_") and robot_node.get("data") is None:

                    if prev_robot_line is not None:
                        print("previous segemnt", prev_robot_line)

                        pnt = prev_robot_line

                        if prev_robot_line is not "line_segment_0":

                            idx = int(prev_robot_line.split("_")[-1])
                            pnt = f"line_segment_{(idx - 1) % 4}"

                        print("pnt", pnt)

                        # contact points of previous robot line
                        cps = [
                            robot_knowledge.nodes[n].get("data")
                            for n in robot_knowledge.neighbors(pnt)
                            if n.startswith("contact_point_")
                        ]


                        collected_contact_points.extend(cps)

                        # previous pt data
                        if prev_pt:
                            collected_pts.append(robot_knowledge.nodes[prev_pt].get("data"))

                        # copy length from object_knowledge using prev_object_line
                        length = object_knowledge.nodes[prev_object_line].get("length")
                        robot_knowledge.nodes[prev_robot_line]["length"] = length
                        collected_lengths.append(length)

                    break

                # Update previous nodes
                if robot_current.startswith("line_segment_"):
                    prev_robot_line = robot_current
                    prev_object_line = object_current

                if robot_current.startswith("pt_"):
                    prev_pt = robot_current

                # Move to next robot node
                if robot_current.startswith("line_segment_"):
                    idx = int(robot_current.split("_")[-1])
                    robot_current = f"pt_{idx}"
                else:
                    idx = int(robot_current.split("_")[-1])
                    robot_current = f"line_segment_{(idx + 1) % 4}"

                # Move to next object node based on index
                if object_current.startswith("line_segment_"):
                    idx = int(object_current.split("_")[-1])
                    object_current = f"pt_{idx}"
                else:
                    idx = int(object_current.split("_")[-1])
                    object_current = f"line_segment_{(idx + 1) % 4}"

            if collected_contact_points and collected_pts:
                robot_knowledge.nodes[Graph.get_all_corners(robot_knowledge)[-1]][
                    "data"] = move_from_edge(
                    collected_pts[0], collected_contact_points[0], 90.0, collected_lengths[0])

                self.send_marker(position=robot_knowledge.nodes[Graph.get_all_corners(robot_knowledge)[-1]][
                    "data"], marker_type="corner", frame="marker_frame_0")

                Graph.add_new_line_segment(robot_knowledge)

                new_line = Graph.get_all_segments(robot_knowledge)[-1]

                # The new lines matched_id must follow the same progression as before:
                # Traversing in positive direction
                prev_object_idx = int(prev_object_line.split("_")[-1])
                new_object_idx = (prev_object_idx + 1) % 4
                new_object_line = f"line_segment_{new_object_idx}"
                # Assign the matched_id so later steps know the correspondence
                robot_knowledge.nodes[new_line]["matched_id"] = new_object_line
                robot_knowledge.nodes[new_line]["length"] = object_knowledge.nodes[new_object_line]["length"]

                # COLLECT PARAMETERS FOR NEXT STEP

                # The new pts data
                new_pt_data = robot_knowledge.nodes[Graph.get_all_corners(robot_knowledge)[-1]]["data"]

                # One contact point from the previous line segment
                previous_line = prev_robot_line
                previous_contact_points = [
                    robot_knowledge.nodes[n].get("data")
                    for n in robot_knowledge.neighbors(previous_line)
                    if n.startswith("contact_point_")
                ]
                previous_contact_point = previous_contact_points[0] if previous_contact_points else None

                # The newly assigned length
                new_length = robot_knowledge.nodes[new_line]["length"]

                last_pt = Graph.get_all_corners(robot_knowledge)[-1]

                # Corner angle attached to this PT
                corner_angle_neighbors = [
                    n for n in robot_knowledge.neighbors(last_pt)
                    if n.startswith("corner_angle_")
                ]

                # There should be exactly one
                corner_angle_value = None
                if corner_angle_neighbors:
                    corner_angle_node = corner_angle_neighbors[0]
                    corner_angle_value = robot_knowledge.nodes[corner_angle_node].get("angle")


                Graph.add_new_corner(robot_knowledge, object_knowledge, Graph.get_all_segments(robot_knowledge)[-1])


                robot_knowledge.nodes[Graph.get_all_corners(robot_knowledge)[-1]][
                    "data"] = move_from_edge(new_pt_data, previous_contact_point, corner_angle_value, new_length)

                self.send_marker(position=robot_knowledge.nodes[Graph.get_all_corners(robot_knowledge)[-1]][
                    "data"], marker_type="corner", frame="marker_frame_0")


                Graph.add_new_line_segment(robot_knowledge)

                corners = Graph.get_all_corners(robot_knowledge)

                # Last corner
                last_corner = corners[-1]
                last_corner_data = robot_knowledge.nodes[last_corner]["data"]

                # Identify the last line segment added
                new_line = Graph.get_all_segments(robot_knowledge)[-1]

                # Determine its matched_id in object_knowledge
                prev_object_idx = int(prev_object_line.split("_")[-1])

                new_object_idx = (prev_object_idx + 2) % 4
                new_object_line = f"line_segment_{new_object_idx}"

                # Assign the matched_id
                robot_knowledge.nodes[new_line]["matched_id"] = new_object_line

                # Copy length from object_knowledge
                if new_object_line in object_knowledge.nodes:
                    new_length = object_knowledge.nodes[new_object_line]["length"]
                    robot_knowledge.nodes[new_line]["length"] = new_length
                else:
                    print(f"[WARN] Object knowledge missing {new_object_line}, skipping length copy")
                    new_length = None

                # Previous corner
                prev_corner = corners[-2]
                prev_corner_data = robot_knowledge.nodes[prev_corner]["data"]

                corner_angle_neighbors = [
                    n for n in robot_knowledge.neighbors(last_corner)
                    if n.startswith("corner_angle_")
                ]
                corner_angle_value = None
                if corner_angle_neighbors:
                    corner_angle_node = corner_angle_neighbors[0]
                    corner_angle_value = robot_knowledge.nodes[corner_angle_node].get("angle")

                Graph.add_new_corner(robot_knowledge, object_knowledge, Graph.get_all_segments(robot_knowledge)[-1])

                robot_knowledge.nodes[Graph.get_all_corners(robot_knowledge)[-1]][
                    "data"] = move_from_edge(last_corner_data, prev_corner_data, corner_angle_value, new_length)

                self.send_marker(position=robot_knowledge.nodes[Graph.get_all_corners(robot_knowledge)[-1]][
                    "data"], marker_type="corner", frame="marker_frame_0")

                robot_knowledge.add_edge("line_segment_0", Graph.get_all_corners(robot_knowledge)[-1])

                print("ADDED LAST CORNER:")
                return

        # MAIN EXECUTION
        compensate_contact_points()
        resolve_corners()
        compensate_contact_points()
        forward_resolve_nodes()
        assign_segment_lengths()


        num_okg = len(Graph.get_all_corners(object_knowledge))
        num_belief = len(Graph.get_all_corners(robot_knowledge))

        if num_okg == num_belief:
            plane_position, plane_orientation = pose_from_points_end(Graph.get_all_corner_data(robot_knowledge))
            self.send_marker(position=plane_position, orientation=plane_orientation, frame="marker_frame_0", marker_type="plane")
            assign_segment_lengths()
            Graph.update_belief_state_visualization(robot_knowledge)
            graph_complete = True
            while True:
                time.sleep(1.0)


        return


    # Send all marker information over to the visualizer
    def send_marker(self, marker_type="marker", frame="eddie_base_link", position=None, orientation=None):

        pose_stamped = PoseStamped()

        if position is not None:
            x, y, z = position
        else:
            x, y, z = current_position


        if orientation is None:
            qx, qy, qz, qw = current_orientation
        else:
            qx, qy, qz, qw = orientation

        # Set header
        pose_stamped.header.stamp = self.get_clock().now().to_msg()

        if frame:
            pose_stamped.header.frame_id = frame
        else:
            pose_stamped.header.frame_id = current_frame

        # Set position
        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        #pose_stamped.pose.position.z = z

        if current_frame == "marker_frame_0":
            pose_stamped.pose.position.z = 0.0
        else:
            pose_stamped.pose.position.z = z

        # Set orientation
        pose_stamped.pose.orientation.x = qx
        pose_stamped.pose.orientation.y = qy
        pose_stamped.pose.orientation.z = qz
        pose_stamped.pose.orientation.w = qw


        if marker_type == "marker":
            self.publisher_marker.publish(pose_stamped)
            print("SENDING MARKER")
        elif marker_type == "corner":
            self.publisher_corner.publish(pose_stamped)
            print("SENDING CORNER")
        elif marker_type == "plane":
            self.publisher_plane.publish(pose_stamped)
            print("SENDING PLANE")


        return

    def ee_callback(self, msg):
        global current_position, current_orientation, current_frame

        frame_id = msg.header.frame_id

        current_frame = frame_id

        #Take ee position here and unpack it into current_position
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        current_position = [x, y, z]

        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w

        current_orientation = [qx, qy, qz, qw]

    def send_goal(self, inp):

        if not self.client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("No Server found!")
            return None

        goal_msg = MotionSpecification.Goal()
        goal_msg.motion_specification = str(inp) #Add the actual message here

        #print(str(inp))

        self.get_logger().info("Sending Goal!")

        # Send a goal to the server
        send_goal_future = self.client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)

        #Wait until the action returns
        rclpy.spin_until_future_complete(self, send_goal_future)

        # Get the goal handle
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Goal was rejected!")
            return None

        self.get_logger().info("Goal was accepted, waiting for results.")

        goal_results = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, goal_results)

        result = goal_results.result()

        if result.status == 4:
            self.get_logger().info("Goal succeeded.")
            return result.result
        else:
            self.get_logger().warn("Goal failed.")
            return None

    def feedback_callback(self, feedback_msg):
        pose = feedback_msg.feedback.tcp_position  # adjust to match your action definition
        position = list(pose)
        collected_points.append(position)
        return position


def main(args=None):

    rclpy.init(args=args)

    # Create the marker publisher node
    reasoner_node = ReasonerNode()

    # Clean up on shutdown
    reasoner_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

