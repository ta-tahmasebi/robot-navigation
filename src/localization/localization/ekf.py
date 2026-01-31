import math
from dataclasses import dataclass

import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


def yaw_to_quaternion(yaw: float) -> Quaternion:
    half = yaw * 0.5
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


def quaternion_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Topics:
    prediction: str
    measurement: str
    output: str


class EKF:
    def __init__(self, x0: np.ndarray | None = None, P0: np.ndarray | None = None):
        self.x = np.zeros((5, 1)) if x0 is None else x0.astype(float).reshape(5, 1)
        self.P = np.eye(5) * 0.1 if P0 is None else P0.astype(float).reshape(5, 5)
        self.Q = np.eye(5) * 0.02
        self.R = np.eye(5) * 0.05
        self._I = np.eye(5)

    def predict(self, v: float, w: float, dt: float) -> None:
        x, y, yaw, _, _ = self.x.flatten()

        x_pred = np.zeros((5, 1))
        x_pred[0, 0] = x + v * math.cos(yaw) * dt
        x_pred[1, 0] = y + v * math.sin(yaw) * dt
        x_pred[2, 0] = normalize_angle(yaw + w * dt)
        x_pred[3, 0] = v
        x_pred[4, 0] = w

        F = np.eye(5)
        F[0, 2] = -v * math.sin(yaw) * dt
        F[1, 2] = v * math.cos(yaw) * dt
        F[0, 3] = math.cos(yaw) * dt
        F[1, 3] = math.sin(yaw) * dt
        F[2, 4] = dt

        self.P = F @ self.P @ F.T + self.Q
        self.x = x_pred

    def update(self, z: np.ndarray) -> None:
        z = z.astype(float).reshape(5, 1)
        y = z - self.x
        y[2, 0] = normalize_angle(float(y[2, 0]))

        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.x[2, 0] = normalize_angle(float(self.x[2, 0]))
        self.P = (self._I - K) @ self.P


class EKFNode(Node):
    STATE_DIM = 5

    def __init__(self) -> None:
        super().__init__("ekf_node")

        self.declare_parameter("prediction_topic", "/prediction/odom")
        self.declare_parameter("measurement_topic", "/measurement/odom")
        self.declare_parameter("output_topic", "/ekf/odom")

        topics = Topics(
            prediction=str(self.get_parameter("prediction_topic").value),
            measurement=str(self.get_parameter("measurement_topic").value),
            output=str(self.get_parameter("output_topic").value),
        )

        self.ekf = EKF()
        self.last_time_s: float | None = None

        self.pub = self.create_publisher(Odometry, topics.output, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(Odometry, topics.prediction, self.on_prediction, 20)
        self.create_subscription(Odometry, topics.measurement, self.on_measurement, 20)


    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def on_prediction(self, msg: Odometry) -> None:
        now = self.now_s()
        if self.last_time_s is None:
            self.last_time_s = now
            return

        dt = now - self.last_time_s
        self.last_time_s = now
        if dt <= 0.0:
            return

        v = float(msg.twist.twist.linear.x)
        w = float(msg.twist.twist.angular.z)

        self.ekf.predict(v=v, w=w, dt=dt)
        self.publish_state()

    def on_measurement(self, msg: Odometry) -> None:
        z = np.zeros((self.STATE_DIM, 1), dtype=float)
        z[0, 0] = float(msg.pose.pose.position.x)
        z[1, 0] = float(msg.pose.pose.position.y)
        z[2, 0] = float(quaternion_to_yaw(msg.pose.pose.orientation))
        z[3, 0] = float(msg.twist.twist.linear.x)
        z[4, 0] = float(msg.twist.twist.angular.z)

        self.ekf.update(z)
        self.publish_state()

    def publish_state(self) -> None:
        x, y, yaw, v, w = self.ekf.x.flatten()

        stamp = self.get_clock().now().to_msg()

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.orientation = yaw_to_quaternion(float(yaw))

        odom.twist.twist.linear.x = float(v)
        odom.twist.twist.angular.z = float(w)

        self.pub.publish(odom)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = 0.0
        t.transform.rotation = yaw_to_quaternion(float(yaw))

        self.tf_broadcaster.sendTransform(t)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EKFNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
