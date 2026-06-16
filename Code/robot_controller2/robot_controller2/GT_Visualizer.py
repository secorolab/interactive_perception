import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
import tf2_ros
from builtin_interfaces.msg import Duration as BuiltinDuration
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R 
from enum import Enum

try:
    from aruco_interfaces.msg import ArucoMarkers
except ImportError:
    ArucoMarkers = None

@dataclass
class MarkerObservation:
    position: Point
    orientation: Quaternion
    frame_id: str

class ModeOfComputation(Enum):
    MARKERS_FOR_POSE_GIVEN_DIMENSIONS = 1
    MARKERS_FOR_POSE_AND_DIMENSIONS = 2
    # MARKERS_FOR_ORIENTATION_ONLY = 3


class MarkerLayoutMode(Enum):
    CORNER_MARKERS = "corner_markers"
    MID_MARKER_IDS = "mid_marker_ids"
    
class EXPERIMENT_CONFIG(Enum):
    SINGLE_RECT_SURFACE_FLAT = 1
    SINGLE_RECT_SURFACE_WITH_SLOPE = 2
    

def _normalize(vector: np.ndarray, name: str) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 1e-9:
        raise ValueError(f"Cannot normalize zero-length {name}")
    return vector / norm


def plane_normal_from_points(point_array: np.ndarray) -> np.ndarray:
    centroid = np.mean(point_array, axis=0)
    covariance = np.cov((point_array - centroid).T)
    _, eigenvectors = np.linalg.eigh(covariance)
    z_axis = _normalize(eigenvectors[:, 0], "plane normal")
    if z_axis[2] < 0.0:
        z_axis = -z_axis
    return z_axis


def right_handed_rotation_from_x_axis(point_array: np.ndarray, x_axis: np.ndarray) -> np.ndarray:
    z_axis = plane_normal_from_points(point_array)
    x_axis = _normalize(x_axis, "x axis")
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis

    y_axis = _normalize(np.cross(z_axis, x_axis), "y axis")
    z_axis = _normalize(np.cross(x_axis, y_axis), "z axis")
    return np.column_stack((x_axis, y_axis, z_axis))


def x_axis_from_corner_markers(point_array: np.ndarray) -> np.ndarray:
    if point_array.shape[0] != 4:
        raise ValueError("corner_markers mode requires exactly 4 marker points")

    edge_candidates = []
    for i in range(4):
        for j in range(i + 1, 4):
            vector = point_array[j] - point_array[i]
            edge_candidates.append((np.linalg.norm(vector), vector))

    # The two shortest pairwise distances are the two opposite width edges.
    shortest_edges = sorted(edge_candidates, key=lambda item: item[0])[:2]
    reference = shortest_edges[0][1]
    aligned_edges = []
    for _, edge in shortest_edges:
        if np.dot(edge, reference) < 0.0:
            edge = -edge
        aligned_edges.append(edge)

    x_axis = _normalize(np.sum(aligned_edges, axis=0), "corner marker x axis")
    if abs(x_axis[0]) >= abs(x_axis[1]):
        return x_axis if x_axis[0] >= 0.0 else -x_axis
    return x_axis if x_axis[1] >= 0.0 else -x_axis


def pose_from_points(
    points: Sequence[Point],
    n_markers_per_surface: int,
    marker_layout_mode: MarkerLayoutMode,
    marker_points_by_id: Optional[Dict[int, Point]] = None,
    bottom_mid_marker_id: int = -1,
    top_mid_marker_id: int = -1,
) -> Tuple[Point, Quaternion]:
    """Compute a plane pose from marker positions in a common frame."""
    point_array = np.array([[point.x, point.y, point.z] for point in points], dtype=np.float64)
    if point_array.shape[0] < n_markers_per_surface:
        raise ValueError(f"Need {n_markers_per_surface} marker points to compute ground truth pose")

    centroid = np.mean(point_array, axis=0)

    if marker_layout_mode == MarkerLayoutMode.CORNER_MARKERS:
        x_axis = x_axis_from_corner_markers(point_array)
    elif marker_layout_mode == MarkerLayoutMode.MID_MARKER_IDS:
        if marker_points_by_id is None:
            raise ValueError("mid_marker_ids mode requires marker_points_by_id")
        if bottom_mid_marker_id not in marker_points_by_id or top_mid_marker_id not in marker_points_by_id:
            raise ValueError(
                "mid_marker_ids mode requires valid bottom_mid_marker_id and top_mid_marker_id"
            )
        bottom = marker_points_by_id[bottom_mid_marker_id]
        top = marker_points_by_id[top_mid_marker_id]
        x_axis = np.array(
            [top.x - bottom.x, top.y - bottom.y, top.z - bottom.z],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unsupported marker layout mode {marker_layout_mode}")

    rotation_matrix = right_handed_rotation_from_x_axis(point_array, x_axis)
    quaternion_from_rot_matrix = R.from_matrix(rotation_matrix).as_quat()
    
    return (
        Point(x=float(centroid[0]), y=float(centroid[1]), z=float(centroid[2])),
        Quaternion(x=float(quaternion_from_rot_matrix[0]), y=float(quaternion_from_rot_matrix[1]), z=float(quaternion_from_rot_matrix[2]), w=float(quaternion_from_rot_matrix[3]))
    )


class GT_VisualizerNode(Node):
    """Publishes ArUco-derived ground truth geometry and supporting TF frames."""

    def __init__(self):
        super().__init__("gt_visualizer_node")

        self.declare_parameter("aruco_topic", "/aruco/markers")
        self.declare_parameter("ee_pose_topic", "/ee_pose")
        self.declare_parameter("plane_topic", "/plane_position")
        self.declare_parameter("target_frame", "eddie_base_link")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("camera_frame", "camera_color_frame")
        self.declare_parameter("end_effector_frame", "end_effector")
        self.declare_parameter("ground_truth_frame", "ground_truth_object")
        self.declare_parameter("map_to_base_xyz", [2.0, 2.28, 0.8])
        self.declare_parameter("ee_to_camera_xyz", [0.0, 0.05639, -0.00305])
        self.declare_parameter("ee_pose_timeout_sec", 0.5)
        self.declare_parameter("marker_layout_mode", MarkerLayoutMode.CORNER_MARKERS.value)
        self.declare_parameter("n_markers_per_surface", 0)
        self.declare_parameter("bottom_mid_marker_id", -1)
        self.declare_parameter("top_mid_marker_id", -1)
        self.declare_parameter("capture_markers_in_single_frame", False)
        self.declare_parameter("single_frame_marker_ids", "")

        self.aruco_topic = self.get_parameter("aruco_topic").value
        self.ee_pose_topic = self.get_parameter("ee_pose_topic").value
        self.plane_topic = self.get_parameter("plane_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.map_frame = self.get_parameter("map_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.end_effector_frame = self.get_parameter("end_effector_frame").value
        self.ground_truth_frame = self.get_parameter("ground_truth_frame").value
        self.map_to_base_xyz = self._parameter_vector("map_to_base_xyz", 3)
        self.ee_to_camera_xyz = self._parameter_vector("ee_to_camera_xyz", 3)
        self.ee_pose_timeout_sec = float(self.get_parameter("ee_pose_timeout_sec").value)
        self.marker_layout_mode = self._marker_layout_mode_from_parameter()
        configured_marker_count = int(self.get_parameter("n_markers_per_surface").value)
        self.n_markers_per_surface = (
            configured_marker_count
            if configured_marker_count > 0
            else self.default_marker_count_for_layout(self.marker_layout_mode)
        )
        self.bottom_mid_marker_id = int(self.get_parameter("bottom_mid_marker_id").value)
        self.top_mid_marker_id = int(self.get_parameter("top_mid_marker_id").value)
        self.capture_markers_in_single_frame = bool(
            self.get_parameter("capture_markers_in_single_frame").value
        )
        self.single_frame_marker_ids = self._marker_id_list_from_parameter("single_frame_marker_ids")

        self.marker_pub = self.create_publisher(Marker, "/visualization_marker", 10)
        self.corner_pub = self.create_publisher(Point, "/ground_truth_corners", 10)
        self.centroid_pub = self.create_publisher(PoseStamped, "/ground_truth_centroid", 10)
        self.centroid_msg = PoseStamped()
        
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.ground_truth: Dict[int, MarkerObservation] = {}
        self.plane_observations: List[Pose] = []
        self.ground_truth_computed = False
        self.manual_marker_capture_complete = False
        self.fixed_ground_truth_tf: Optional[TransformStamped] = None
        self.last_ee_pose_time: Optional[Time] = None
        self.warned_stale_ee_pose = False

        self.create_timer(0.5, self.publish_ground_truth)
        self.create_timer(1.0, self.publish_support_transforms)
        self.slope_angle_degrees = 0.0

        if ArucoMarkers is None:
            self.get_logger().warn("aruco_interfaces is unavailable; ArUco ground truth is disabled")
        else:
            self.create_subscription(ArucoMarkers, self.aruco_topic, self.ground_truth_callback, 10)

        self.create_subscription(PoseStamped, self.ee_pose_topic, self.ee_callback, 2)
        self.create_subscription(PoseStamped, self.plane_topic, self.plane_callback, 10)

        self.mode_of_computation = ModeOfComputation.MARKERS_FOR_POSE_GIVEN_DIMENSIONS
        self.experiment_config = EXPERIMENT_CONFIG.SINGLE_RECT_SURFACE_FLAT
        
        if self.mode_of_computation == ModeOfComputation.MARKERS_FOR_POSE_GIVEN_DIMENSIONS:
            self.get_logger().info("Mode: MARKERS_FOR_POSE_GIVEN_DIMENSIONS - using marker positions to compute pose given dimensions of the object.")
            if self.experiment_config == EXPERIMENT_CONFIG.SINGLE_RECT_SURFACE_FLAT:
                self.get_logger().info("Experiment Config: SINGLE_RECT_SURFACE_FLAT - expecting a single rectangular surface lying flat on the table.")
                self.n_surfaces = 1
                for i in range(self.n_surfaces):
                    self.n_sides = 4
                    self.gt_length = 0.4940
                    self.gt_width = 0.25
                    self.slope_angle_degrees = 0.0
            elif self.experiment_config == EXPERIMENT_CONFIG.SINGLE_RECT_SURFACE_WITH_SLOPE:
                self.get_logger().info("Experiment Config: SINGLE_RECT_SURFACE_WITH_SLOPE - expecting a single rectangular surface with a slope.")
                self.n_surfaces = 1
                for i in range(self.n_surfaces):
                    self.n_sides = 4
                    self.gt_length = 0.4940
                    self.gt_width = 0.25
                    self.slope_angle_degrees = 30.0

        self.get_logger().info("Created GT_Visualizer node")

    def minimum_marker_count_for_layout(self) -> int:
        if self.marker_layout_mode == MarkerLayoutMode.CORNER_MARKERS:
            return 4
        if self.marker_layout_mode == MarkerLayoutMode.MID_MARKER_IDS:
            return 3
        raise ValueError(f"Unsupported marker layout mode {self.marker_layout_mode}")

    def _parameter_vector(self, name: str, expected_length: int) -> List[float]:
        value = list(self.get_parameter(name).value)
        if len(value) != expected_length:
            raise ValueError(f"Parameter '{name}' must contain {expected_length} values")
        return [float(item) for item in value]

    def _marker_layout_mode_from_parameter(self) -> MarkerLayoutMode:
        value = str(self.get_parameter("marker_layout_mode").value)
        try:
            return MarkerLayoutMode(value)
        except ValueError as exc:
            valid_modes = ", ".join(mode.value for mode in MarkerLayoutMode)
            raise ValueError(f"marker_layout_mode must be one of: {valid_modes}") from exc

    def _marker_id_list_from_parameter(self, name: str) -> List[int]:
        value = str(self.get_parameter(name).value).strip()
        if not value:
            return []
        return [int(item.strip()) for item in value.split(",") if item.strip()]

    def default_marker_count_for_layout(self, marker_layout_mode: MarkerLayoutMode) -> int:
        if marker_layout_mode == MarkerLayoutMode.CORNER_MARKERS:
            return 4
        if marker_layout_mode == MarkerLayoutMode.MID_MARKER_IDS:
            return 9
        raise ValueError(f"Unsupported marker layout mode {marker_layout_mode}")

    def ee_pose_is_fresh(self) -> bool:
        if self.ee_pose_timeout_sec <= 0.0:
            return True
        if self.last_ee_pose_time is None:
            if not self.warned_stale_ee_pose:
                self.get_logger().warn(
                    f"No {self.ee_pose_topic} message has arrived; skipping ArUco transform."
                )
                self.warned_stale_ee_pose = True
            return False

        age = self.get_clock().now() - self.last_ee_pose_time
        if age > Duration(seconds=self.ee_pose_timeout_sec):
            if not self.warned_stale_ee_pose:
                self.get_logger().warn(
                    f"Latest {self.ee_pose_topic} is older than "
                    f"{self.ee_pose_timeout_sec:.2f}s; skipping ArUco transform."
                )
                self.warned_stale_ee_pose = True
            return False

        return True

    def capture_single_frame_ground_truth(self, msg, source_frame: str) -> None:
        if len(self.ground_truth) >= self.n_markers_per_surface:
            return

        detections = {}
        for marker_id, pose in zip(msg.marker_ids, msg.poses):
            marker_id = int(marker_id)
            if marker_id not in detections:
                detections[marker_id] = pose

        if self.single_frame_marker_ids:
            selected_ids = self.single_frame_marker_ids
            missing_ids = [marker_id for marker_id in selected_ids if marker_id not in detections]
            if missing_ids:
                self.get_logger().debug(
                    f"Waiting for marker IDs in one frame; missing {missing_ids}"
                )
                return
        else:
            if len(detections) < self.n_markers_per_surface:
                self.get_logger().debug(
                    f"Waiting for {self.n_markers_per_surface} markers in one frame; "
                    f"saw {len(detections)}"
                )
                return
            selected_ids = sorted(detections)[:self.n_markers_per_surface]

        transformed_observations = {}
        for marker_id in selected_ids:
            pose_in = PoseStamped()
            pose_in.header.stamp = Time(seconds=0).to_msg()
            pose_in.header.frame_id = source_frame
            pose_in.pose = detections[marker_id]

            try:
                pose_out = self.tf_buffer.transform(
                    pose_in,
                    self.target_frame,
                    timeout=Duration(seconds=0.5),
                )
            except TransformException as exc:
                self.get_logger().warn(
                    f"Could not transform same-frame ArUco marker {marker_id} "
                    f"from {source_frame} to {self.target_frame}: {exc}"
                )
                return

            transformed_observations[marker_id] = MarkerObservation(
                position=pose_out.pose.position,
                orientation=pose_out.pose.orientation,
                frame_id=pose_out.header.frame_id,
            )

        prompt = (
            f"Add {len(selected_ids)} ArUco markers from the same frame as ground truth "
            f"(ids={selected_ids})? [y/N]: "
        )
        try:
            consent = input(prompt).strip().lower()
        except EOFError:
            consent = ""

        if consent not in ("y", "yes"):
            self.get_logger().info("Skipped same-frame ground truth capture; user did not approve.")
            return

        self.ground_truth = transformed_observations
        self.ground_truth_computed = False
        self.fixed_ground_truth_tf = None
        self.get_logger().info(
            f"Stored {len(self.ground_truth)} ground truth markers from one frame: {selected_ids}"
        )

    def ground_truth_callback(self, msg) -> None:
        """Store the first transformed pose for each ArUco marker id."""
        if self.manual_marker_capture_complete:
            return

        source_frame = msg.header.frame_id or self.camera_frame

        if not self.ee_pose_is_fresh():
            return

        if self.capture_markers_in_single_frame:
            self.capture_single_frame_ground_truth(msg, source_frame)
            return

        for marker_id, pose in zip(msg.marker_ids, msg.poses):
            marker_id = int(marker_id)
            if marker_id in self.ground_truth:
                continue

            pose_in = PoseStamped()
            pose_in.header.stamp = Time(seconds=0).to_msg()
            pose_in.header.frame_id = source_frame
            pose_in.pose = pose

            try:
                pose_out = self.tf_buffer.transform(
                    pose_in,
                    self.target_frame,
                    timeout=Duration(seconds=0.5),
                )
            except TransformException as exc:
                self.get_logger().warn(
                    f"Could not transform ArUco marker {marker_id} "
                    f"from {source_frame} to {self.target_frame}: {exc}"
                )
                continue

            # get user consent if the marker_id value pose in camera_color_frame 
            # is acceptable to be added to the ground truth. 
            # This is to prevent adding wrong markers due to noise or detection errors.
            quaternion_to_euler = R.from_quat([
                pose_in.pose.orientation.x,
                pose_in.pose.orientation.y,
                pose_in.pose.orientation.z,
                pose_in.pose.orientation.w
            ]).as_euler('zyx', degrees=True)
            
            prompt = (
                f"Add marker {marker_id} pose from camera_color_frame to ground truth? "
                f"position=({pose_in.pose.position.x:.4f}, "
                f"{pose_in.pose.position.y:.4f}, "
                f"{pose_in.pose.position.z:.4f}), "
                f"orientation=({quaternion_to_euler[0]:.1f}°, "
                f"{quaternion_to_euler[1]:.1f}°, "
                f"{quaternion_to_euler[2]:.1f}°), [y/N/done]: "
            )
            try:
                consent = input(prompt).strip().lower()
            except EOFError:
                consent = ""

            if consent in ("d", "done"):
                self.manual_marker_capture_complete = True
                self.ground_truth_computed = False
                self.fixed_ground_truth_tf = None
                self.get_logger().info(
                    f"Finished manual marker capture with {len(self.ground_truth)} markers: "
                    f"{sorted(self.ground_truth)}"
                )
                return

            if consent not in ("y", "yes"):
                self.get_logger().info(
                    f"Skipped marker {marker_id}; user did not approve the camera_color_frame pose."
                )
                continue

            self.ground_truth[marker_id] = MarkerObservation(
                position=pose_out.pose.position,
                orientation=pose_out.pose.orientation,
                frame_id=pose_out.header.frame_id,
            )
            self.get_logger().info(
                f"Stored ground truth marker {marker_id} in {pose_out.header.frame_id}: "
                f"({pose_out.pose.position.x:.5f}, "
                f"{pose_out.pose.position.y:.5f}, "
                f"{pose_out.pose.position.z:.5f}). "
                f"Stored markers: {len(self.ground_truth)} "
                f"(target {self.n_markers_per_surface}; type 'done' at the next prompt to finish early)"
            )
            try:
                next_action = input("Continue marker capture? [Y/done]: ").strip().lower()
            except EOFError:
                next_action = ""
            if next_action in ("d", "done"):
                self.manual_marker_capture_complete = True
                self.ground_truth_computed = False
                self.fixed_ground_truth_tf = None
                self.get_logger().info(
                    f"Finished manual marker capture with {len(self.ground_truth)} markers: "
                    f"{sorted(self.ground_truth)}"
                )
                return

    def plane_callback(self, msg: PoseStamped) -> None:
        self.plane_observations.append(msg.pose)

    def ee_callback(self, msg: PoseStamped) -> None:
        self.last_ee_pose_time = self.get_clock().now()
        self.warned_stale_ee_pose = False

        transform = TransformStamped()
        transform.header.stamp = self.last_ee_pose_time.to_msg()
        transform.header.frame_id = msg.header.frame_id
        transform.child_frame_id = self.end_effector_frame
        transform.transform.translation.x = msg.pose.position.x
        transform.transform.translation.y = msg.pose.position.y
        transform.transform.translation.z = msg.pose.position.z
        transform.transform.rotation = msg.pose.orientation
        self.tf_broadcaster.sendTransform(transform)

    def publish_ground_truth(self) -> None:
        if self.manual_marker_capture_complete:
            minimum_count = self.minimum_marker_count_for_layout()
            if len(self.ground_truth) < minimum_count:
                self.get_logger().warn(
                    f"Need at least {minimum_count} markers for {self.marker_layout_mode.value}; "
                    f"only {len(self.ground_truth)} were captured."
                )
                return
        elif len(self.ground_truth) < self.n_markers_per_surface:
            return

        if not self.ground_truth_computed:
            try:
                self.compute_ground_truth_pose()
            except ValueError as exc:
                self.get_logger().warn(str(exc))
                return

        if self.fixed_ground_truth_tf is not None:
            self.fixed_ground_truth_tf.header.stamp = self.get_clock().now().to_msg()
            self.tf_broadcaster.sendTransform(self.fixed_ground_truth_tf)

    def compute_ground_truth_pose(self) -> None:
        sorted_observations = sorted(self.ground_truth.items(), key=lambda item: item[0])
        ordered_points = [observation.position for _, observation in sorted_observations]
        marker_points_by_id = {
            marker_id: observation.position
            for marker_id, observation in sorted_observations
        }
        centroid, orientation = pose_from_points(
            ordered_points,
            len(ordered_points) if self.manual_marker_capture_complete else self.n_markers_per_surface,
            self.marker_layout_mode,
            marker_points_by_id=marker_points_by_id,
            bottom_mid_marker_id=self.bottom_mid_marker_id,
            top_mid_marker_id=self.top_mid_marker_id,
        )
        
        r = R.from_quat([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ])

        # get yaw_degrees from the original computed orientation, and use it to construct a new orientation with known pitch and roll
        yaw_deg, _, _ = r.as_euler('zyx', degrees=True)
        known_pitch_deg = self.slope_angle_degrees

        # forcing roll to 0
        upd_quat = R.from_euler(
            'zyx',
            [yaw_deg, known_pitch_deg, 0.0],
            degrees=True
        ).as_quat()

        orientation = Quaternion()
        orientation.x = upd_quat[0]
        orientation.y = upd_quat[1]
        orientation.z = upd_quat[2]
        orientation.w = upd_quat[3]

        self.centroid_msg.header.stamp = self.get_clock().now().to_msg()
        self.centroid_msg.header.frame_id = self.target_frame
        self.centroid_msg.pose.position = centroid
        self.centroid_msg.pose.orientation = orientation
        self.centroid_pub.publish(self.centroid_msg)

        self.fixed_ground_truth_tf = TransformStamped()
        self.fixed_ground_truth_tf.header.frame_id = self.target_frame
        self.fixed_ground_truth_tf.child_frame_id = self.ground_truth_frame
        self.fixed_ground_truth_tf.transform.translation.x = centroid.x
        self.fixed_ground_truth_tf.transform.translation.y = centroid.y
        self.fixed_ground_truth_tf.transform.translation.z = centroid.z
        self.fixed_ground_truth_tf.transform.rotation = orientation

        euler_ang_deg = R.from_quat([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ]).as_euler('zyx', degrees=True)
        euler_rpy_rad = R.from_quat([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ]).as_euler('xyz', degrees=False)
        ground_truth_json = {
            "pose": {
                "position": [
                    float(centroid.x),
                    float(centroid.y),
                    float(centroid.z),
                ],
                "quaternion": [
                    float(orientation.x),
                    float(orientation.y),
                    float(orientation.z),
                    float(orientation.w),
                ],
                "euler": [float(value) for value in euler_rpy_rad],
            },
            "corners": {
                "points": [
                    [
                        float(point.x),
                        float(point.y),
                        float(point.z),
                    ]
                    for point in self.ordered_ground_truth_points()
                ]
            },
        }
        
        self.ground_truth_computed = True
        self.get_logger().info(
            f"Computed {self.ground_truth_frame} from {len(ordered_points)} markers in {self.target_frame}. \
            Centroid: ({centroid.x:.5f}, {centroid.y:.5f}, {centroid.z:.5f}). \
            Orientation (euler): ({euler_ang_deg[0]:.5f}, {euler_ang_deg[1]:.5f}, {euler_ang_deg[2]:.5f})"
        )
        self.get_logger().info(
            "Ground truth JSON snippet:\n"
            f"{json.dumps(ground_truth_json, indent=2)}"
        )

    def publish_support_transforms(self) -> None:
        now = self.get_clock().now().to_msg()
        
        zero_quat_array = R.from_euler('XYZ', 
                                       [0.0, 0.0, 0.0], 
                                       degrees=True).as_quat()
        zero_quat = Quaternion(x=float(zero_quat_array[0]), y=float(zero_quat_array[1]), 
                              z=float(zero_quat_array[2]), w=float(zero_quat_array[3]))

        self.tf_broadcaster.sendTransform(
            self.make_transform(
                self.map_frame,
                self.target_frame,
                self.map_to_base_xyz,
                zero_quat,
                now,
            )
        )
        
        quat_array = R.from_euler('XYZ', 
                                  [180.0, 180.0, 0.0], 
                                  degrees=True).as_quat()
        quat = Quaternion(x=float(quat_array[0]), y=float(quat_array[1]), 
                         z=float(quat_array[2]), w=float(quat_array[3]))
        self.tf_broadcaster.sendTransform(
            self.make_transform(
                self.end_effector_frame,
                self.camera_frame,
                self.ee_to_camera_xyz,
                quat,
                now,
            )
        )

        if self.fixed_ground_truth_tf is not None:
            self.publish_ground_truth_corners()
            self.publish_ground_truth_surface()
            self.publish_centroid()

    def make_transform(
        self,
        parent_frame: str,
        child_frame: str,
        translation: Sequence[float],
        rotation: Quaternion,
        stamp,
    ) -> TransformStamped:
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        transform.transform.translation.x = float(translation[0])
        transform.transform.translation.y = float(translation[1])
        transform.transform.translation.z = float(translation[2])
        transform.transform.rotation = rotation
        return transform

    def publish_ground_truth_corners(self) -> None:
        for point in self.ordered_ground_truth_points():
            self.corner_pub.publish(point)

    def publish_centroid(self) -> None:
        if self.fixed_ground_truth_tf is None:
            return
        self.centroid_pub.publish(self.centroid_msg)

    def publish_ground_truth_surface(self) -> None:
        points = self.ordered_ground_truth_points()
        if len(points) < 4:
            return

        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "ground_truth_surface"
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = BuiltinDuration(sec=0)

        marker.points.extend([points[0], points[1], points[2]])
        marker.points.extend([points[0], points[2], points[3]])

        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.5
        marker.color.a = 0.4

        self.marker_pub.publish(marker)

    def ordered_ground_truth_points(self) -> List[Point]:
        
        rotation_matrix = R.from_quat([
            self.fixed_ground_truth_tf.transform.rotation.x,
            self.fixed_ground_truth_tf.transform.rotation.y,
            self.fixed_ground_truth_tf.transform.rotation.z,
            self.fixed_ground_truth_tf.transform.rotation.w
        ]).as_matrix()
        x_axis = rotation_matrix[:, 0]
        y_axis = rotation_matrix[:, 1]
        
        if self.mode_of_computation == ModeOfComputation.MARKERS_FOR_POSE_GIVEN_DIMENSIONS:
            for i in range(self.n_surfaces):
                if self.experiment_config in (EXPERIMENT_CONFIG.SINGLE_RECT_SURFACE_FLAT, EXPERIMENT_CONFIG.SINGLE_RECT_SURFACE_WITH_SLOPE):
                    return [
                        Point(
                            x=self.fixed_ground_truth_tf.transform.translation.x + dx * x_axis[0] + dy * y_axis[0],
                            y=self.fixed_ground_truth_tf.transform.translation.y + dx * x_axis[1] + dy * y_axis[1],
                            z=self.fixed_ground_truth_tf.transform.translation.z + dx * x_axis[2] + dy * y_axis[2]
                        )
                        for dy, dx in [
                            (-self.gt_length / 2, -self.gt_width / 2),
                            (self.gt_length / 2, -self.gt_width / 2),
                            (self.gt_length / 2, self.gt_width / 2),
                            (-self.gt_length / 2, self.gt_width / 2),
                        ]
                    ]
                else:
                    raise ValueError(f"Unsupported experiment config {self.experiment_config} for mode {self.mode_of_computation}")
        elif self.mode_of_computation == ModeOfComputation.MARKERS_FOR_POSE_AND_DIMENSIONS:
            return [
                Point(
                    x=observation.position.x,
                    y=observation.position.y,
                    z=observation.position.z,
                )
                for _, observation in sorted(self.ground_truth.items(), key=lambda item: item[0])
            ]


def main(args=None):
    rclpy.init(args=args)
    node = GT_VisualizerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
