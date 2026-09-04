import json
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

class MarkerLayoutMode(Enum):
    CORNER_MARKERS = "corner_markers"
    MID_MARKER_IDS = "mid_marker_ids"
    

def _normalize(vector: np.ndarray, name: str) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 1e-9:
        raise ValueError(f"Cannot normalize zero-length {name}")
    return vector / norm


def plane_normal_from_points(point_array: np.ndarray) -> np.ndarray:
    centroid = np.mean(point_array, axis=0)
    covariance = np.cov((point_array - centroid).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if float(np.sort(eigenvalues)[1]) < 1e-9:
        raise ValueError("Marker points do not span a stable plane")
    z_axis = _normalize(eigenvectors[:, 0], "plane normal")
    if z_axis[2] < 0.0:
        z_axis = -z_axis
    return z_axis


def base_referenced_rotation(point_array: np.ndarray) -> np.ndarray:
    """Align z with the marker plane and x with projected base-frame +x."""
    z_axis = plane_normal_from_points(point_array)
    reference_axis = np.array([1.0, 0.0, 0.0])
    x_axis = reference_axis - np.dot(reference_axis, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        reference_axis = np.array([0.0, 1.0, 0.0])
        x_axis = reference_axis - np.dot(reference_axis, z_axis) * z_axis
    x_axis = _normalize(x_axis, "projected base reference axis")
    y_axis = _normalize(np.cross(z_axis, x_axis), "y axis")
    return np.column_stack((x_axis, y_axis, z_axis))


def pose_from_points(
    points: Sequence[Point],
    n_markers_per_surface: int,
) -> Tuple[Point, Quaternion]:
    """Compute a base-referenced plane pose from corner-marker positions."""
    point_array = np.array([[point.x, point.y, point.z] for point in points], dtype=np.float64)
    if point_array.shape[0] < n_markers_per_surface:
        raise ValueError(f"Need {n_markers_per_surface} marker points to compute ground truth pose")
    if point_array.shape[0] < 3:
        raise ValueError("At least three non-collinear marker points are required")

    centroid = np.mean(point_array, axis=0)
    rotation_matrix = base_referenced_rotation(point_array)
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
        self.declare_parameter("ee_to_camera_xyz", [0.0, 0.05639, -0.009525])
        self.declare_parameter("ee_pose_timeout_sec", 0.5)
        self.declare_parameter("marker_layout_mode", MarkerLayoutMode.CORNER_MARKERS.value)
        self.declare_parameter("n_markers_per_surface", 0)
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
        if self.n_markers_per_surface < 3:
            raise ValueError("n_markers_per_surface must be at least 3")
        self.capture_markers_in_single_frame = bool(
            self.get_parameter("capture_markers_in_single_frame").value
        )
        self.single_frame_marker_ids = self._marker_id_list_from_parameter("single_frame_marker_ids")
        if (
            self.single_frame_marker_ids
            and len(self.single_frame_marker_ids) != self.n_markers_per_surface
        ):
            raise ValueError(
                "single_frame_marker_ids must contain exactly "
                f"{self.n_markers_per_surface} IDs"
            )
        if len(set(self.single_frame_marker_ids)) != len(self.single_frame_marker_ids):
            raise ValueError("single_frame_marker_ids must not contain duplicates")

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

        if ArucoMarkers is None:
            self.get_logger().warn("aruco_interfaces is unavailable; ArUco ground truth is disabled")
        else:
            self.create_subscription(ArucoMarkers, self.aruco_topic, self.ground_truth_callback, 10)

        self.create_subscription(PoseStamped, self.ee_pose_topic, self.ee_callback, 2)
        self.create_subscription(PoseStamped, self.plane_topic, self.plane_callback, 10)

        self.get_logger().info(
            f"Created GT_Visualizer node; collecting {self.n_markers_per_surface} "
            f"markers using {self.marker_layout_mode.value}. The marker centroid and PCA "
            "normal define the origin and z-axis; base-frame +x defines the in-plane axes."
        )

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
        self.manual_marker_capture_complete = True
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
            if self.single_frame_marker_ids and marker_id not in self.single_frame_marker_ids:
                continue
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
            target_quaternion_to_euler = R.from_quat([
                pose_out.pose.orientation.x,
                pose_out.pose.orientation.y,
                pose_out.pose.orientation.z,
                pose_out.pose.orientation.w
            ]).as_euler('zyx', degrees=True)
            
            prompt = (
                f"Add marker {marker_id} pose to ground truth?\n"
                f"  {source_frame}: position=({pose_in.pose.position.x:.4f}, "
                f"{pose_in.pose.position.y:.4f}, "
                f"{pose_in.pose.position.z:.4f}), "
                f"orientation zyx=({quaternion_to_euler[0]:.1f}°, "
                f"{quaternion_to_euler[1]:.1f}°, "
                f"{quaternion_to_euler[2]:.1f}°)\n"
                f"  {pose_out.header.frame_id}: position=({pose_out.pose.position.x:.5f}, "
                f"{pose_out.pose.position.y:.5f}, "
                f"{pose_out.pose.position.z:.5f}), "
                f"orientation zyx=({target_quaternion_to_euler[0]:.1f}°, "
                f"{target_quaternion_to_euler[1]:.1f}°, "
                f"{target_quaternion_to_euler[2]:.1f}°)\n"
                "  Accept? [y/N]: "
            )
            try:
                consent = input(prompt).strip().lower()
            except EOFError:
                consent = ""

            if consent not in ("y", "yes"):
                self.get_logger().info(
                    f"Discarded the current observation of marker {marker_id}; "
                    "it may be offered again on a later detection."
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
                f"(target {self.n_markers_per_surface})"
            )

            if len(self.ground_truth) >= self.n_markers_per_surface:
                self.manual_marker_capture_complete = True
                self.ground_truth_computed = False
                self.fixed_ground_truth_tf = None
                self.get_logger().info(
                    f"Collected all {len(self.ground_truth)} configured corner markers: "
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
        if len(self.ground_truth) < self.n_markers_per_surface:
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
        ordered_points = self.ordered_ground_truth_points()
        centroid, orientation = pose_from_points(
            ordered_points,
            self.n_markers_per_surface,
        )

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
                "marker_ids": self.ordered_ground_truth_marker_ids(),
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
        if len(points) < 3:
            return

        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "ground_truth_surface"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = BuiltinDuration(sec=0)

        marker.points.extend(points)
        marker.points.append(points[0])

        marker.scale.x = 0.005
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.5
        marker.color.a = 0.8

        self.marker_pub.publish(marker)

    def ordered_ground_truth_points(self) -> List[Point]:
        marker_ids = self.ordered_ground_truth_marker_ids()
        return [
            Point(
                x=self.ground_truth[marker_id].position.x,
                y=self.ground_truth[marker_id].position.y,
                z=self.ground_truth[marker_id].position.z,
            )
            for marker_id in marker_ids
        ]

    def ordered_ground_truth_marker_ids(self) -> List[int]:
        if self.single_frame_marker_ids:
            return self.single_frame_marker_ids
        return sorted(self.ground_truth)


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
