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
    
class EXPERIMENT_CONFIG(Enum):
    SINGLE_RECT_SURFACE_FLAT = 1
    SINGLE_RECT_SURFACE_WITH_SLOPE = 2
    

def pose_from_points(points: Sequence[Point]) -> Tuple[Point, Quaternion]:
    """Compute a plane pose from marker corner positions in a common frame."""
    point_array = np.array([[point.x, point.y, point.z] for point in points], dtype=np.float64)
    if point_array.shape[0] < 4:
        raise ValueError("Need at least 4 marker points to compute ground truth pose")

    centroid = np.mean(point_array, axis=0)

    covariance = np.cov((point_array - centroid).T)
    _, eigenvectors = np.linalg.eigh(covariance)
    z_axis = eigenvectors[:, 0]
    z_axis /= np.linalg.norm(z_axis)

    # Preserve the original marker ordering convention: x points from marker 3 to marker 1.
    x_axis = point_array[0] - point_array[3]
    x_axis /= np.linalg.norm(x_axis)

    z_axis = z_axis - np.dot(z_axis, x_axis) * x_axis
    z_axis /= np.linalg.norm(z_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)

    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
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
        self.fixed_ground_truth_tf: Optional[TransformStamped] = None

        self.create_timer(0.5, self.publish_ground_truth)
        self.create_timer(1.0, self.publish_support_transforms)

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
                    self.length = 0.4890
                    self.width = 0.25
            elif self.experiment_config == EXPERIMENT_CONFIG.SINGLE_RECT_SURFACE_WITH_SLOPE:
                self.get_logger().info("Experiment Config: SINGLE_RECT_SURFACE_WITH_SLOPE - expecting a single rectangular surface with a slope.")
                self.n_surfaces = 1
                for i in range(self.n_surfaces):
                    self.n_sides = 4
                    self.length = 0.4890
                    self.width = 0.25
                    self.slope_angle_degrees = 30.0

        self.get_logger().info("Created GT_Visualizer node")

    def _parameter_vector(self, name: str, expected_length: int) -> List[float]:
        value = list(self.get_parameter(name).value)
        if len(value) != expected_length:
            raise ValueError(f"Parameter '{name}' must contain {expected_length} values")
        return [float(item) for item in value]

    def ground_truth_callback(self, msg) -> None:
        """Store the first transformed pose for each ArUco marker id."""
        source_frame = msg.header.frame_id or self.camera_frame

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
                f"{quaternion_to_euler[2]:.1f}°), [y/N]: "
            )
            try:
                consent = input(prompt).strip().lower()
            except EOFError:
                consent = ""

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
                f"({pose_out.pose.position.x:.3f}, "
                f"{pose_out.pose.position.y:.3f}, "
                f"{pose_out.pose.position.z:.3f})"
            )

    def plane_callback(self, msg: PoseStamped) -> None:
        self.plane_observations.append(msg.pose)

    def ee_callback(self, msg: PoseStamped) -> None:
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = msg.header.frame_id
        transform.child_frame_id = self.end_effector_frame
        transform.transform.translation.x = msg.pose.position.x
        transform.transform.translation.y = msg.pose.position.y
        transform.transform.translation.z = msg.pose.position.z
        transform.transform.rotation = msg.pose.orientation
        self.tf_broadcaster.sendTransform(transform)

    def publish_ground_truth(self) -> None:
        if len(self.ground_truth) < 4:
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
        ordered_points = [
            observation.position
            for _, observation in sorted(self.ground_truth.items(), key=lambda item: item[0])
        ]
        centroid, orientation = pose_from_points(ordered_points)

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
        
        self.ground_truth_computed = True
        self.get_logger().info(
            f"Computed {self.ground_truth_frame} from {len(ordered_points)} markers in {self.target_frame}. \
            Centroid: ({centroid.x:.3f}, {centroid.y:.3f}, {centroid.z:.3f}). \
            Orientation (euler): ({euler_ang_deg[0]:.3f}, {euler_ang_deg[1]:.3f}, {euler_ang_deg[2]:.3f})"
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
        if self.mode_of_computation == ModeOfComputation.MARKERS_FOR_POSE_GIVEN_DIMENSIONS:
            for i in range(self.n_surfaces):
                if self.experiment_config == EXPERIMENT_CONFIG.SINGLE_RECT_SURFACE_FLAT:
                    return [
                        Point(
                            x=self.fixed_ground_truth_tf.transform.translation.x + dx,
                            y=self.fixed_ground_truth_tf.transform.translation.y + dy,
                            z=self.fixed_ground_truth_tf.transform.translation.z
                        )
                        for dx, dy in [
                            (-self.length / 2, -self.width / 2),
                            (self.length / 2, -self.width / 2),
                            (self.length / 2, self.width / 2),
                            (-self.length / 2, self.width / 2),
                        ]
                    ]
                
                elif self.experiment_config == EXPERIMENT_CONFIG.SINGLE_RECT_SURFACE_WITH_SLOPE:
                    # ground truth is always in base_link. 
                    slope_radians = math.radians(self.slope_angle_degrees)
                    dz = math.tan(slope_radians) * self.length
                    return [
                        Point(
                            x=self.fixed_ground_truth_tf.transform.translation.x + dx,
                            y=self.fixed_ground_truth_tf.transform.translation.y + dy,
                            z=self.fixed_ground_truth_tf.transform.translation.z + (dz if dx > 0 else 0.0)
                        )
                        for dx, dy in [
                            (-self.length / 2, -self.width / 2),
                            (self.length / 2, -self.width / 2),
                            (self.length / 2, self.width / 2),
                            (-self.length / 2, self.width / 2),
                        ]
                    ]
                    
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