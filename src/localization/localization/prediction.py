import math
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64


def yaw_to_quaternion(yaw: float) -> Quaternion:
    half = yaw * 0.5
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Topics:
    left_w: str
    right_w: str
    odom: str


@dataclass
class DiffDriveParams:
    wheel_radius: float
    wheel_separation: float


class DiffDriveModel:
    def __init__(self, params: DiffDriveParams) -> None:
        self.params = params

    def wheel_to_body(self, w_l: float, w_r: float) -> tuple[float, float]:
        v_l = w_l * self.params.wheel_radius
        v_r = w_r * self.params.wheel_radius
        v = 0.5 * (v_r + v_l)
        w = (v_r - v_l) / self.params.wheel_separation
        return v, w


class OdometryPredictor(Node):
    def __init__(self) -> None:
        super().__init__("odometry_predictor")

        self.declare_parameter("wheel_radius", 0.10)
        self.declare_parameter("wheel_separation", 0.45)
        self.declare_parameter("left_angular_vel_topic", "/left_motor_angular_vel")
        self.declare_parameter("right_angular_vel_topic", "/right_motor_angular_vel")
        self.declare_parameter("odom_topic", "/prediction/odom")
        self.declare_parameter("prediction_rate", 50.0)

        params = DiffDriveParams(
            wheel_radius=float(self.get_parameter("wheel_radius").value),
            wheel_separation=float(self.get_parameter("wheel_separation").value),
        )
        topics = Topics(
            left_w=str(self.get_parameter("left_angular_vel_topic").value),
            right_w=str(self.get_parameter("right_angular_vel_topic").value),
            odom=str(self.get_parameter("odom_topic").value),
        )
        prediction_rate = float(self.get_parameter("prediction_rate").value)

        self.model = DiffDriveModel(params)

        self.w_l = 0.0
        self.w_r = 0.0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_time_s: float | None = None

        self.create_subscription(Float64, topics.left_w, self.on_left, 10)
        self.create_subscription(Float64, topics.right_w, self.on_right, 10)
        self.pub = self.create_publisher(Odometry, topics.odom, 10)

        period = 1.0 / max(prediction_rate, 1e-6)
        self.create_timer(period, self.on_timer)

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def on_left(self, msg: Float64) -> None:
        self.w_l = float(msg.data)

    def on_right(self, msg: Float64) -> None:
        self.w_r = float(msg.data)

    def on_timer(self) -> None:
        now = self.now_s()
        if self.last_time_s is None:
            self.last_time_s = now
            return

        dt = now - self.last_time_s
        self.last_time_s = now
        if dt <= 0.0:
            return

        v, w = self.model.wheel_to_body(self.w_l, self.w_r)

        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt
        self.yaw = normalize_angle(self.yaw + w * dt)

        stamp = self.get_clock().now().to_msg()

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = float(self.x)
        odom.pose.pose.position.y = float(self.y)
        odom.pose.pose.orientation = yaw_to_quaternion(self.yaw)

        odom.twist.twist.linear.x = float(v)
        odom.twist.twist.angular.z = float(w)

        self.pub.publish(odom)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OdometryPredictor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
