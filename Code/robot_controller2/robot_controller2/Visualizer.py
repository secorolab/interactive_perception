import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped, TransformStamped
from geometry_msgs.msg import Quaternion

from builtin_interfaces.msg import Duration
import tf2_ros


class VisualizerNode(Node):

    def __init__(self):
        super().__init__('visualizer_node')

        # Publishers
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Internal state
        self.markers = []
        self.corners = []
        self.planes = []
        self._id_counters = {
            "marker": 0,
            "corner": 0,
            "plane": 0
        }

        # Subscriptions
        self.marker_sub = self.create_subscription(
            PoseStamped, '/marker_position', self.handle_marker_pose, 10
        )

        self.corner_sub = self.create_subscription(
            PoseStamped, '/corner_position', self.handle_corner_pose, 10
        )

        self.plane_sub = self.create_subscription(
            PoseStamped, '/plane_position', self.handle_plane_pose, 10
        )

        # Timers
        self.publish_timer = self.create_timer(1.0, self.publish_all)
        self.surface_timer = self.create_timer(0.5, self.publish_plane_surface)

    # =========================
    # Core Utilities
    # =========================

    def _next_id(self, category: str) -> int:
        current = self._id_counters[category]
        self._id_counters[category] += 1
        return current

    def _create_marker(self, ns, position, orientation, frame_id, color, scale=(0.05, 0.05, 0.05)):
        marker = Marker()

        marker.ns = ns
        marker.id = self._next_id(ns)

        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.lifetime = Duration(sec=0)

        marker.pose.position = position
        marker.pose.orientation = orientation

        marker.scale.x, marker.scale.y, marker.scale.z = scale
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color

        return marker

    def _broadcast_tf(self, marker, prefix, timestamp):
        transform = TransformStamped()

        transform.header.stamp = timestamp
        transform.header.frame_id = marker.header.frame_id
        transform.child_frame_id = f"{prefix}_{marker.id}"

        p = marker.pose.position
        q = marker.pose.orientation

        transform.transform.translation.x = p.x
        transform.transform.translation.y = p.y
        transform.transform.translation.z = p.z

        transform.transform.rotation = q

        self.tf_broadcaster.sendTransform(transform)

    def _publish_collection(self, collection, publish_tf=False, frame_prefix="frame"):
        now = self.get_clock().now().to_msg()

        for obj in collection:
            self.marker_pub.publish(obj)

            if publish_tf:
                self._broadcast_tf(obj, frame_prefix, now)

    # =========================
    # Callbacks
    # =========================

    def handle_marker_pose(self, msg: PoseStamped):
        marker = self._create_marker(
            ns="marker",
            position=msg.pose.position,
            orientation=msg.pose.orientation,
            frame_id=msg.header.frame_id,
            color=(0.0, 0.0, 1.0, 1.0)
        )
        self.markers.append(marker)

    def handle_corner_pose(self, msg: PoseStamped):
        corner = self._create_marker(
            ns="corner",
            position=msg.pose.position,
            orientation=msg.pose.orientation,
            frame_id=msg.header.frame_id,
            color=(0.0, 1.0, 1.0, 1.0)
        )
        self.corners.append(corner)

    def handle_plane_pose(self, msg: PoseStamped):
        plane = self._create_marker(
            ns="plane",
            position=msg.pose.position,
            orientation=msg.pose.orientation,
            frame_id=msg.header.frame_id,
            color=(1.0, 0.5, 0.0, 1.0)
        )
        self.planes.append(plane)

    # =========================
    # Publishing
    # =========================

    def publish_all(self):
        self._publish_collection(self.markers, publish_tf=True, frame_prefix="marker_frame")
        self._publish_collection(self.corners)
        self._publish_collection(self.planes, publish_tf=True, frame_prefix="plane_frame")

    # =========================
    # Plane Surface Generation
    # =========================

    def publish_plane_surface(self):
        if len(self.corners) < 3:
            return

        positions = [c.pose.position for c in self.corners]

        marker = Marker()
        marker.header.frame_id = "marker_frame_0"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "plane_surface"
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD

        # Fan triangulation
        points = []
        for i in range(1, len(positions) - 1):
            points.extend([positions[0], positions[i], positions[i + 1]])

        marker.points = points

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.4

        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        self.marker_pub.publish(marker)


# =========================
# Main
# =========================

def main(args=None):
    rclpy.init(args=args)

    node = VisualizerNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()







# import rclpy
# from rclpy.node import Node
# from visualization_msgs.msg import Marker
# from geometry_msgs.msg import Point
# from std_msgs.msg import ColorRGBA
# import rclpy
# from rclpy.node import Node
# import tf2_ros
# import geometry_msgs.msg
# from rclpy.time import Time
# from tf2_ros import TransformBroadcaster
# from geometry_msgs.msg import TransformStamped
# import tf2_geometry_msgs
# import math
# import random
# from geometry_msgs.msg import Quaternion
# from geometry_msgs.msg import PoseStamped
# from visualization_msgs.msg import Marker
# from geometry_msgs.msg import Point
# from builtin_interfaces.msg import Duration
# import mapbox_earcut as earcut
# import numpy as np
# from aruco_interfaces.msg import ArucoMarkers
# import numpy as np
# from geometry_msgs.msg import PointStamped
# from tf2_ros import TransformException



# # Set to keep track of marker IDs that are in use
# used_marker_ids = set()

# #All markers that are going to be published
# markers = []

# used_corner_ids = set()

# corners = []

# used_plane_ids = set()

# planes = []

# combined_list = []

# ground_truth = {}

# used_gt_ids = set()

# last_marker = None

# poses = []


# def publish_all_markers(marker_pub, broadcaster, time):
#     for marker in markers:
#         marker_pub.publish(marker)

#         position_tuple = (marker.pose.position.x, marker.pose.position.y, marker.pose.position.z)
#         transform_rotation = (marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w)
#         unique_target_frame = f"marker_frame_{marker.id}"
#         parent_frame = marker.header.frame_id

#         if unique_target_frame == "marker_frame_0":

#             transform = TransformStamped()

#             transform.header.stamp = time
#             transform.header.frame_id = parent_frame
#             transform.child_frame_id = unique_target_frame

#             transform.transform.translation.x = position_tuple[0]
#             transform.transform.translation.y = position_tuple[1]
#             transform.transform.translation.z = position_tuple[2]

#             quaternion = transform_rotation

#             quaternion_msg = Quaternion()
#             quaternion_msg.x = quaternion[0]
#             quaternion_msg.y = quaternion[1]
#             quaternion_msg.z = quaternion[2]
#             quaternion_msg.w = quaternion[3]

#             transform.transform.rotation = quaternion_msg

#             broadcaster.sendTransform(transform)


# def publish_all_corners(marker_pub, broadcaster, time):
#     for corner in corners:
#         marker_pub.publish(corner)


# def publish_all_planes(marker_pub, broadcaster, time):
#     for plane in planes:
#         marker_pub.publish(plane)

#         position_tuple = (plane.pose.position.x, plane.pose.position.y, plane.pose.position.z)
#         transform_rotation = (plane.pose.orientation.x, plane.pose.orientation.y, plane.pose.orientation.z, plane.pose.orientation.w)
#         unique_target_frame = f"plane_frame_{plane.id}"
#         parent_frame = plane.header.frame_id
#         transform = TransformStamped()

#         transform.header.stamp = time
#         transform.header.frame_id = parent_frame
#         transform.child_frame_id = unique_target_frame

#         transform.transform.translation.x = position_tuple[0]
#         transform.transform.translation.y = position_tuple[1]
#         transform.transform.translation.z = position_tuple[2]

#         quaternion = transform_rotation

#         quaternion_msg = Quaternion()
#         quaternion_msg.x = quaternion[0]
#         quaternion_msg.y = quaternion[1]
#         quaternion_msg.z = quaternion[2]
#         quaternion_msg.w = quaternion[3]


#         transform.transform.rotation = quaternion_msg


#         broadcaster.sendTransform(transform)

# def generate_unique_marker_id():
#     marker_id = 0
#     while marker_id in used_marker_ids:
#         marker_id += 1
#     used_marker_ids.add(marker_id)
#     return marker_id

# def generate_unique_corner_id():
#     corner_id = 0
#     while corner_id in used_corner_ids:
#         corner_id += 1
#     used_corner_ids.add(corner_id)
#     return corner_id

# def generate_unique_plane_id():
#     plane_id = 0
#     while plane_id in used_plane_ids:
#         plane_id += 1
#     used_plane_ids.add(plane_id)
#     return plane_id


# def create_marker(position, orientation, timestamp, frame_id):

#     # Generate a unique marker ID
#     marker_id = generate_unique_marker_id()

#     marker = Marker()
#     marker.ns = "marker"
#     marker.lifetime = Duration(sec=0)
#     marker.header.frame_id = frame_id
#     marker.header.stamp = timestamp
#     marker.id = marker_id
#     marker.type = Marker.SPHERE
#     marker.action = marker.ADD
#     #Cant be point object if marker is manually added
#     #marker.pose.position = Point(x=position[0], y=position[1], z=position[2])
#     marker.pose.position = position
#     marker.pose.orientation = orientation
#     marker.scale.x, marker.scale.y, marker.scale.z = (0.05, 0.05, 0.05)
#     marker.color.a, marker.color.r, marker.color.g, marker.color.b = (1.0, 0.0, 0.0, 1.0)
#     print("Adding marker...")

#     return marker

# def create_corner(position, orientation, timestamp, frame_id):

#     # Generate a unique marker ID
#     marker_id = generate_unique_corner_id()

#     marker = Marker()
#     marker.ns = "corner"
#     marker.lifetime = Duration(sec=0)
#     marker.header.frame_id = frame_id
#     marker.header.stamp = timestamp
#     marker.id = marker_id
#     marker.type = Marker.SPHERE
#     marker.action = marker.ADD
#     marker.pose.position = position
#     marker.pose.orientation = orientation
#     marker.scale.x, marker.scale.y, marker.scale.z = (0.05, 0.05, 0.05)
#     marker.color.a, marker.color.r, marker.color.g, marker.color.b = (1.0, 0.0, 1.0, 1.0)
#     print("Adding corner...")

#     return marker

# def create_plane(position, orientation, timestamp, frame_id):

#     # Generate a unique marker ID
#     marker_id = generate_unique_plane_id()

#     marker = Marker()
#     marker.ns = "plane"
#     marker.lifetime = Duration(sec=0)
#     marker.header.frame_id = frame_id
#     marker.header.stamp = timestamp
#     marker.id = marker_id
#     marker.type = Marker.SPHERE
#     marker.action = marker.ADD
#     marker.pose.position = position
#     marker.pose.orientation = orientation
#     marker.scale.x, marker.scale.y, marker.scale.z = (0.05, 0.05, 0.05)
#     marker.color.a, marker.color.r, marker.color.g, marker.color.b = (1.0, 1.0, 0.5, 0.0)
#     print("Adding plane...")

#     return marker



# class VisualizerNode(Node):

#     def __init__(self):

#         global ground_truth
#         super().__init__('visualizer_node')

#         self.publisher = self.create_publisher(Marker, '/visualization_marker', 10)

#         self.br = tf2_ros.TransformBroadcaster(self)

#         self.marker_timer = self.create_timer(1.0, self.publish_marker)

#         self.corner_timer = self.create_timer(1.0, self.publish_corner)

#         self.plane_timer = self.create_timer(1.0, self.publish_plane)

#         self.timer = self.create_timer(0.5, self.publish_plane_surface)


#         self.subscription = self.create_subscription(
#             PoseStamped,
#             '/marker_position',
#             self.marker_callback,
#             10
#         )

#         self.subscription = self.create_subscription(
#             PoseStamped,
#             '/corner_position',
#             self.corner_callback,
#             10
#         )

#         self.subscription = self.create_subscription(
#             PoseStamped,
#             '/plane_position',
#             self.plane_callback,
#             10
#         )


#     def publish_plane_surface(self):
#         global corners

#         if not isinstance(corners, list) or len(corners) < 4:
#             #self.get_logger().warn("Need at least 3 corner markers to form a plane")
#             return

#         # Extract points from marker poses
#         positions = []
#         for m in corners:
#             if hasattr(m, "pose") and hasattr(m.pose, "position"):
#                 positions.append(m.pose.position)

#         if len(positions) < 3:
#             self.get_logger().warn("Not enough valid positions in markers to form a plane")
#             return



#         marker = Marker()
#         marker.header.frame_id = "marker_frame_0"
#         marker.header.stamp = self.get_clock().now().to_msg()
#         marker.ns = "plane_surface"
#         marker.id = 0
#         marker.type = Marker.TRIANGLE_LIST
#         marker.action = Marker.ADD

#         # Fan triangulation
#         points = []
#         for i in range(1, len(positions) - 1):
#             p0 = positions[0]
#             p1 = positions[i]
#             p2 = positions[i + 1]
#             points.extend([p0, p1, p2])



#         marker.points = points

#         marker.color.r = 0.0
#         marker.color.g = 1.0
#         marker.color.b = 1.0
#         marker.color.a = 0.4

#         marker.scale.x = 1.0
#         marker.scale.y = 1.0
#         marker.scale.z = 3.0

#         self.publisher.publish(marker)


#     def marker_callback(self, msg):
#         position = msg.pose.position
#         orientation = msg.pose.orientation
#         frame_id = msg.header.frame_id
#         markers.append(create_marker(position, orientation, self.get_clock().now().to_msg(), frame_id))


#     def corner_callback(self, msg):
#         position = msg.pose.position
#         orientation = msg.pose.orientation
#         frame_id = msg.header.frame_id
#         corners.append(create_corner(position, orientation, self.get_clock().now().to_msg(), frame_id))


#     def plane_callback(self, msg):
#         position = msg.pose.position
#         orientation = msg.pose.orientation
#         frame_id = msg.header.frame_id
#         planes.append(create_plane(position, orientation, self.get_clock().now().to_msg(), frame_id))



#     def publish_marker(self):
#         publish_all_markers(self.publisher, self.br, self.get_clock().now().to_msg())

#     def publish_corner(self):
#         publish_all_corners(self.publisher, self.br, self.get_clock().now().to_msg())

#     def publish_plane(self):
#         publish_all_planes(self.publisher, self.br, self.get_clock().now().to_msg())


# def main(args=None):
#     rclpy.init(args=args)

#     # Create the marker publisher node
#     marker_publisher = VisualizerNode()

#     # Spin the node to keep it alive and publishing
#     rclpy.spin(marker_publisher)

#     # Clean up on shutdown
#     marker_publisher.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()
