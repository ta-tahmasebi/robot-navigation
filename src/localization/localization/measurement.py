import math
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu


def quaternion_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    half = yaw * 0.5
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def fuse_angle(a: float, b: float, alpha: float) -> float:
    return normalize_angle(a + alpha * normalize_angle(b - a))


@dataclass(frozen=True)
class Topics:
    imu: str
    vo: str
    output: str


class MeasurementNode(Node):
    def __init__(self) -> None:
        super().__init__("measurement_node")

        self.declare_parameter("imu_topic", "/zed/zed_node/imu/data_raw")
        self.declare_parameter("vo_topic", "/vo/odom")
        self.declare_parameter("output_topic", "/measurement/odom")
        self.declare_parameter("alpha_orientation", 0.05)
        self.declare_parameter("alpha_velocity", 0.2)

        topics = Topics(
            imu=str(self.get_parameter("imu_topic").value),
            vo=str(self.get_parameter("vo_topic").value),
            output=str(self.get_parameter("output_topic").value),
        )
        self.alpha_ori = float(self.get_parameter("alpha_orientation").value)
        self.alpha_vel = float(self.get_parameter("alpha_velocity").value)

        self.latest_imu: Imu | None = None
        self.latest_vo: Odometry | None = None

        self.create_subscription(Imu, topics.imu, self.on_imu, 20)
        self.create_subscription(Odometry, topics.vo, self.on_vo, 20)

        self.pub = self.create_publisher(Odometry, topics.output, 10)


    def on_imu(self, msg: Imu) -> None:
        self.latest_imu = msg
        self.try_publish()

    def on_vo(self, msg: Odometry) -> None:
        self.latest_vo = msg
        self.try_publish()

    def try_publish(self) -> None:
        imu = self.latest_imu
        vo = self.latest_vo
        if imu is None or vo is None:
            return

        x = float(vo.pose.pose.position.x)
        y = float(vo.pose.pose.position.y)

        vo_yaw = quaternion_to_yaw(vo.pose.pose.orientation)
        imu_yaw = quaternion_to_yaw(imu.orientation)
        fused_yaw = fuse_angle(vo_yaw, imu_yaw, self.alpha_ori)

        vx = float(vo.twist.twist.linear.x)
        vy = float(vo.twist.twist.linear.y)
        wz_vo = float(vo.twist.twist.angular.z)
        wz_imu = float(imu.angular_velocity.z)
        fused_wz = (1.0 - self.alpha_vel) * wz_vo + self.alpha_vel * wz_imu

        fused = Odometry()
        fused.header.stamp = self.get_clock().now().to_msg()
        fused.header.frame_id = "odom"
        fused.child_frame_id = "base_link"

        fused.pose.pose.position.x = x
        fused.pose.pose.position.y = y
        fused.pose.pose.orientation = yaw_to_quaternion(fused_yaw)

        fused.twist.twist.linear.x = vx
        fused.twist.twist.linear.y = vy
        fused.twist.twist.angular.z = fused_wz

        self.pub.publish(fused)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MeasurementNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
