import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import math


def quaternion_to_yaw(q: Quaternion):
    """Extract yaw angle from quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw):
    """Convert yaw → quaternion."""
    return Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0),
    )


class MeasurementNode(Node):

    def __init__(self):
        super().__init__("measurement_node")

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------
        self.declare_parameter("imu_topic", "/zed/zed_node/imu/data_raw")
        self.declare_parameter("vo_topic", "/vo/odom")
        self.declare_parameter("output_topic", "/measurement/odom")
        self.declare_parameter("alpha_orientation", 0.05)  # IMU orientation weight
        self.declare_parameter("alpha_velocity", 0.2)       # IMU angular vel weight

        imu_topic = self.get_parameter("imu_topic").value
        vo_topic = self.get_parameter("vo_topic").value
        output_topic = self.get_parameter("output_topic").value
        self.alpha_ori = self.get_parameter("alpha_orientation").value
        self.alpha_vel = self.get_parameter("alpha_velocity").value

        # ------------------------------------------------------------
        # Internal storage
        # ------------------------------------------------------------
        self.imu_msg = None
        self.vo_msg = None

        # ------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------
        self.imu_sub = self.create_subscription(
            Imu, imu_topic, self.imu_callback, 20
        )
        self.vo_sub = self.create_subscription(
            Odometry, vo_topic, self.vo_callback, 20
        )

        # ------------------------------------------------------------
        # Publisher
        # ------------------------------------------------------------
        self.pub = self.create_publisher(Odometry, output_topic, 10)

        self.get_logger().info("Measurement fusion node started.")

    # ------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------
    def imu_callback(self, msg):
        self.imu_msg = msg
        self.try_fuse()

    def vo_callback(self, msg):
        self.vo_msg = msg
        self.try_fuse()

    # ------------------------------------------------------------
    # Fusion Logic (Complementary measurement model)
    # ------------------------------------------------------------
    def try_fuse(self):
        if self.imu_msg is None or self.vo_msg is None:
            return

        imu = self.imu_msg
        vo = self.vo_msg

        # ------------------------------------------------------------
        # 1. Extract VO pose
        # ------------------------------------------------------------
        x = vo.pose.pose.position.x
        y = vo.pose.pose.position.y
        vo_yaw = quaternion_to_yaw(vo.pose.pose.orientation)

        # ------------------------------------------------------------
        # 2. Extract IMU orientation (only yaw used here)
        # ------------------------------------------------------------
        imu_yaw = quaternion_to_yaw(imu.orientation)

        # ------------------------------------------------------------
        # 3. Complementary filter for orientation
        # ------------------------------------------------------------
        fused_yaw = (
            (1.0 - self.alpha_ori) * vo_yaw +
            self.alpha_ori * imu_yaw
        )

        # ------------------------------------------------------------
        # 4. Linear velocity → use VO (best)
        # ------------------------------------------------------------
        vx = vo.twist.twist.linear.x
        vy = vo.twist.twist.linear.y

        # ------------------------------------------------------------
        # 5. Angular velocity → blended
        # ------------------------------------------------------------
        fused_angular_z = (
            (1.0 - self.alpha_vel) * vo.twist.twist.angular.z +
            self.alpha_vel * imu.angular_velocity.z
        )

        # ------------------------------------------------------------
        # Construct fused Odometry message
        # ------------------------------------------------------------
        fused = Odometry()
        fused.header.stamp = self.get_clock().now().to_msg()
        fused.header.frame_id = "odom"
        fused.child_frame_id = "base_link"

        # Pose
        fused.pose.pose.position.x = x
        fused.pose.pose.position.y = y
        fused.pose.pose.orientation = yaw_to_quaternion(fused_yaw)

        # Twist
        fused.twist.twist.linear.x = vx
        fused.twist.twist.linear.y = vy
        fused.twist.twist.angular.z = fused_angular_z

        # Publish
        self.pub.publish(fused)


def main(args=None):
    rclpy.init(args=args)
    node = MeasurementNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
