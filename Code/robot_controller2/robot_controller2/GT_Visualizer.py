import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import rclpy
from rclpy.node import Node
import tf2_ros
import geometry_msgs.msg
from rclpy.time import Time
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import tf2_geometry_msgs  # Import this for transforming geometry_msgs
import math
import random
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration
import mapbox_earcut as earcut
import numpy as np

try:
    from aruco_interfaces.msg import ArucoMarkers
except ImportError:
    ArucoMarkers = None
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
import geometry_msgs.msg
from geometry_msgs.msg import PointStamped
from tf2_ros import TransformException
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped
from tf2_ros import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation as Rscipy


from scipy.spatial.transform import Rotation as R  # for quaternion math

ground_truth = {}

used_gt_ids = set()

last_marker = None

used_plane_ids = set()

poses = []

fixed_marker = None

tf_poses = []

planes = []

def generate_unique_gt_id():
    gt_id = 0
    while gt_id in used_gt_ids:
        gt_id += 1
    used_plane_ids.add(gt_id)
    return gt_id

def euler_to_quaternion(roll, pitch, yaw):
    qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - math.cos(roll / 2) * math.sin(
        pitch / 2) * math.sin(yaw / 2)
    qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.cos(
        pitch / 2) * math.sin(yaw / 2)
    qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - math.sin(roll / 2) * math.sin(
        pitch / 2) * math.cos(yaw / 2)
    qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.sin(
        pitch / 2) * math.sin(yaw / 2)

    return [qx, qy, qz, qw]


def pose_from_points(points):
    """
    Compute pose from points:
      - x-axis points exactly from points[idx_x[0]] to points[idx_x[1]]
      - z-axis is plane normal (from PCA)
      - y-axis = z x x to make right-handed frame
    This is almost the same as the one from Reasoner.
    """
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

    # X axis exactly between two points
    #x_axis = pts[5] - pts[4]
    x_axis = pts[1] - pts[3]
    x_axis /= np.linalg.norm(x_axis)

    # Make z perpendicular to x
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


class GT_VisualizerNode(Node):

    def __init__(self):

        global ground_truth
        super().__init__('gt_visualizer_node')

        self.publisher = self.create_publisher(Marker, '/visualization_marker', 10)

        self.publisher_point = self.create_publisher(Point, '/ground_truth_corners', 10)

        self.publisher_centroid = self.create_publisher(PoseStamped, '/ground_truth_centroid', 10)

        self.br = tf2_ros.TransformBroadcaster(self)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.ground_truth_computed = False
        self.fixed_tf = None

        self.timer = self.create_timer(0.5, self.publish_gt)

        self.transform_timer = self.create_timer(1.0, self.send_transform)

        print("Created GT_Visualizer Node.")


        if ArucoMarkers:
            self.create_subscription(
                ArucoMarkers,
                "/aruco/markers",
                self.ground_truth_callback,
                10
            )
        self.subscription = self.create_subscription(
            PoseStamped,
            '/ee_pose',
            self.ee_callback,
            2
        )
        self.subscription = self.create_subscription(
            PoseStamped,
            '/plane_position',
            self.plane_callback,
            10
        )

    def ground_truth_callback(self, msg):
        """
        Update ground_truth only when a new marker_id is received.
        Existing markers are left unchanged.
        """
        global ground_truth, poses
        frame_id = msg.header.frame_id

        for marker_id, pose in zip(msg.marker_ids, msg.poses):
            if marker_id not in ground_truth:
                pose_in = PoseStamped()
                pose_in.header.stamp = Time(seconds=0).to_msg()
                pose_in.header.frame_id = 'camera_color_frame'

                pose_in.pose.position.x = pose.position.x
                pose_in.pose.position.y = pose.position.y
                pose_in.pose.position.z = pose.position.z
                pose_in.pose.orientation = pose.orientation
                print(pose_in.pose)
                try:
                    pose_out = self.tf_buffer.transform(
                        pose_in,
                        'eddie_base_link',
                        timeout=Duration(seconds=0.5)
                    )
                except TransformException as ex:
                    self.get_logger().warn(f'Could not transform Pose: {ex}')
                    return
                ground_truth[marker_id] = {
                    "position": pose_out.pose.position,
                    "orientation": pose_out.pose.orientation,
                    "frame_id": pose_out.header.frame_id
                }
                poses.append(pose_out.pose.position)
                print(pose_out)


    def plane_callback(self, msg):
        position = msg.pose.position
        orientation = msg.pose.orientation
        frame_id = msg.header.frame_id
        planes.append([position, orientation])


    def ee_callback(self, msg):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = msg.header.frame_id
        transform.child_frame_id = "end_effector"

        transform.transform.translation.x = msg.pose.position.x
        transform.transform.translation.y = msg.pose.position.y
        transform.transform.translation.z = msg.pose.position.z

        transform.transform.rotation = msg.pose.orientation

        self.br.sendTransform(transform)


    def publish_surface_marker(self):
        global last_marker

        # Build a new marker from the current ground truth
        new_marker = self.make_surface_marker()
        if not new_marker:
            return

        # Check if last_marker exists and has the same attributes
        if last_marker:
            desired_points = len(new_marker.points) == len(last_marker.points)
            same_ns = new_marker.ns == last_marker.ns
            same_id = new_marker.id == last_marker.id

            if desired_points and same_ns and same_id:
                # Nothing changed, reuse the old marker
                self.publisher.publish(last_marker)
                return

        # Otherwise, publish the new marker and save it
        new_marker.ns = "surface"
        new_marker.id = 0
        new_marker.action = Marker.ADD
        self.publisher.publish(new_marker)
        last_marker = new_marker



    def publish_gt(self):
        global ground_truth, poses, fixed_marker
        if not ground_truth:
            return

        if len(ground_truth) < 4:
            return

        #Compute fixed transform only once
        if not self.ground_truth_computed:
            all_poses = []

            orientation = None

            for marker_id in sorted(ground_truth.keys()):
                data = ground_truth[marker_id]
                position = data["position"]
                x, y, z = position.x, position.y, position.z

                print(f"Marker ID: {marker_id}", "position: ", position)
                all_poses.append([x, y, z])

            # all_poses is ordered by marker_id
            pos, ori = pose_from_points(all_poses)

            pos = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            ori = Quaternion(x=float(ori[0]), y=float(ori[1]), z=float(ori[2]), w=float(ori[3]))

            #Fill PoseStamped
            pose_in = PoseStamped()
            pose_in.header.stamp = Time(seconds=0).to_msg()
            pose_in.header.frame_id = 'camera_color_frame'

            pose_in.pose.position.x = pos.x
            pose_in.pose.position.y = pos.y
            pose_in.pose.position.z = pos.z
            pose_in.pose.orientation = ori
            #
            # try:
            #     pose_out = self.tf_buffer.transform(
            #         pose_in,
            #         'eddie_base_link',
            #         timeout=Duration(seconds=0.5)
            #     )
            # except TransformException as ex:
            #     self.get_logger().warn(f'Could not transform Pose: {ex}')
            #     return

            self.publisher_centroid.publish(pose_in)

            # Save the fixed transform
            self.fixed_tf = TransformStamped()
            self.fixed_tf.header.frame_id = 'eddie_base_link'
            self.fixed_tf.child_frame_id = 'ground_truth_object'
            self.fixed_tf.transform.translation.x = pose_in.pose.position.x
            self.fixed_tf.transform.translation.y = pose_in.pose.position.y
            self.fixed_tf.transform.translation.z = pose_in.pose.position.z
            self.fixed_tf.transform.rotation = pose_in.pose.orientation

            # Publish Plane Marker
            # fixed_marker = Marker()
            # fixed_marker.header.frame_id = "ground_truth_object"
            # fixed_marker.header.stamp = self.get_clock().now().to_msg()
            # fixed_marker.ns = "ground_truth_surface"
            # fixed_marker.id = 0
            # fixed_marker.type = Marker.TRIANGLE_LIST
            # fixed_marker.action = Marker.ADD
            # fixed_marker.pose.orientation.w = 1.0

            # # Hardcoded object dimensions
            # x_len = 0.6    #0.31  #0.34
            # y_len = 0.34   #0.98  #0.6
            #
            # # Define rectangle corners
            # p0 = Point(x=-x_len / 2, y=-y_len / 2, z=0.0)
            # p1 = Point(x=x_len / 2, y=-y_len / 2, z=0.0)
            # p2 = Point(x=x_len / 2, y=y_len / 2, z=0.0)
            # p3 = Point(x=-x_len / 2, y=y_len / 2, z=0.0)
            #
            # poses = [p0, p1, p2, p3]

            # poses = []
            #
            # for pose in all_poses:
            #     poses.append(Point(x=pose[0], y=pose[1], z=pose[2]))

            # for pose in poses:
            #     self.publisher_point.publish(pose)

            # Add rectangle as two triangles
            # fixed_marker.points.extend([p0, p1, p2])
            # fixed_marker.points.extend([p0, p2, p3])

            # fixed_marker.points.extend([poses[0], poses[1], poses[2]])
            # fixed_marker.points.extend([poses[0], poses[2], poses[3]])
            #
            # fixed_marker.scale.x = 1.0
            # fixed_marker.scale.y = 1.0
            # fixed_marker.scale.z = 1.0
            #
            # fixed_marker.color.r = 1.0
            # fixed_marker.color.g = 0.0
            # fixed_marker.color.b = 0.5
            # fixed_marker.color.a = 0.4

            self.ground_truth_computed = True


        if self.fixed_tf:
            self.fixed_tf.header.stamp = self.get_clock().now().to_msg()
            self.br.sendTransform(self.fixed_tf)
            # fixed_marker.header.stamp = self.get_clock().now().to_msg()
            # self.publisher.publish(fixed_marker)



    def send_transform(self):

        global poses, tf_poses

        t = geometry_msgs.msg.TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()

        t.header.frame_id = 'map'
        t.child_frame_id = 'eddie_base_link'

        t.transform.translation.x = 2.0
        t.transform.translation.y = 2.28
        t.transform.translation.z = 0.8

        quat = euler_to_quaternion(0, 0, 0)
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.br.sendTransform(t)

        t = geometry_msgs.msg.TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()

        t.header.frame_id = 'end_effector'
        t.child_frame_id = 'camera_color_frame'

        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.05639
        t.transform.translation.z = -0.00305

        quat = euler_to_quaternion(math.pi, math.pi, 0)
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.br.sendTransform(t)

        if self.fixed_tf and len(poses) >= 4:

            # out = []
            #
            # for i, point_msg in enumerate(poses):
            #     point_in = PointStamped()
            #     point_in.header.stamp = Time(seconds=0).to_msg()
            #     point_in.header.frame_id = 'camera_color_frame'
            #
            #     point_in.point.x = point_msg.x
            #     point_in.point.y = point_msg.y
            #     point_in.point.z = point_msg.z
            #
            #     try:
            #         point_out = self.tf_buffer.transform(
            #             point_in,
            #             'eddie_base_link',
            #             timeout=Duration(seconds=0.1)
            #         )
            #         out.append(point_out)
            #
            #     except TransformException as ex:
            #         self.get_logger().warn(
            #             f'Could not transform Point'
            #         )
            #         break
            #
            # list_out = []
            # for stamped_point in out:
            #     point = stamped_point.point
            #     x = float(point.x)  #
            #     y = float(point.y)
            #     z = float(point.z)
            #     list_out.append([x, y, z])
            #
            # tf_poses = list_out
            # print(tf_poses)


                    # Publish Plane Marker
            fixed_marker = Marker()
            fixed_marker.header.frame_id = "eddie_base_link"
            fixed_marker.header.stamp = self.get_clock().now().to_msg()
            fixed_marker.ns = "ground_truth_surface"
            fixed_marker.id = 0
            fixed_marker.type = Marker.TRIANGLE_LIST
            fixed_marker.action = Marker.ADD
            fixed_marker.pose.orientation.w = 1.0

            publish_poses = []

            for pose in poses:
                publish_poses.append(Point(x=pose.x, y=pose.y, z=pose.z))
                self.publisher_point.publish(Point(x=pose.x, y=pose.y, z=pose.z))


            fixed_marker.points.extend([publish_poses[0], publish_poses[1], publish_poses[2]])
            fixed_marker.points.extend([publish_poses[0], publish_poses[2], publish_poses[3]])

            fixed_marker.scale.x = 1.0
            fixed_marker.scale.y = 1.0
            fixed_marker.scale.z = 1.0

            fixed_marker.color.r = 1.0
            fixed_marker.color.g = 0.0
            fixed_marker.color.b = 0.5
            fixed_marker.color.a = 0.4


            fixed_marker.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(fixed_marker)




def main(args=None):
    rclpy.init(args=args)

    # Create the marker publisher node
    gt_publisher = GT_VisualizerNode()

    # Spin the node to keep it alive and publishing
    rclpy.spin(gt_publisher)

    # Clean up on shutdown
    gt_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
