import math

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
from rclpy.node import Node


def yaw_to_quaternion(yaw: float) -> Quaternion:
    half = yaw * 0.5
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


class InitialPosePublisher(Node):
    def __init__(self) -> None:
        super().__init__("initial_pose_publisher")

        self.declare_parameter("topic", "/initialpose")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("x", 11.138)
        self.declare_parameter("y", 6.0)
        self.declare_parameter("z", 0.0)
        self.declare_parameter("yaw", 0.0)
        self.declare_parameter(
            "covariance",
            [
                0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0685389,
            ],
        )

        self.pub = self.create_publisher(
            PoseWithCovarianceStamped,
            str(self.get_parameter("topic").value),
            10,
        )

        self._sent = False
        self.create_timer(2, self._publish_once)

    def _publish_once(self) -> None:
        if self._sent:
            return
        self._sent = True

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.get_parameter("frame_id").value)

        msg.pose.pose.position.x = float(self.get_parameter("x").value)
        msg.pose.pose.position.y = float(self.get_parameter("y").value)
        msg.pose.pose.position.z = float(self.get_parameter("z").value)

        yaw = float(self.get_parameter("yaw").value)
        msg.pose.pose.orientation = yaw_to_quaternion(yaw)

        cov = list(self.get_parameter("covariance").value)
        if len(cov) == 36:
            msg.pose.covariance = [float(v) for v in cov]

        self.pub.publish(msg)
        self.get_logger().info(
            f"Published initial pose once: x={msg.pose.pose.position.x:.3f}, "
            f"y={msg.pose.pose.position.y:.3f}, z={msg.pose.pose.position.z:.3f}, yaw={yaw:.3f}"
        )

        rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InitialPosePublisher()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == "__main__":
    main()
