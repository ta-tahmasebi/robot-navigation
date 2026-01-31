import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import math


def yaw_to_quaternion(yaw):
    return Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0),
    )


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class EKFNode(Node):

    def __init__(self):
        super().__init__("ekf_node")

        # ---------------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------------
        self.declare_parameter("prediction_topic", "/prediction/odom")
        self.declare_parameter("measurement_topic", "/measurement/odom")
        self.declare_parameter("output_topic", "/ekf/odom")

        pred_topic = self.get_parameter("prediction_topic").value
        meas_topic = self.get_parameter("measurement_topic").value
        self.output_topic = self.get_parameter("output_topic").value

        # ---------------------------------------------------------------
        # EKF State Vector: [x, y, yaw, v, omega]
        # ---------------------------------------------------------------
        self.x = np.zeros((5, 1))
        # Covariance matrices
        self.P = np.eye(5) * 0.1                           # state covariance
        self.Q = np.eye(5) * 0.02                          # prediction noise
        self.R = np.eye(5) * 0.05                          # measurement noise

        self.last_time = None

        # ---------------------------------------------------------------
        # Subscribers
        # ---------------------------------------------------------------
        self.prediction_sub = self.create_subscription(
            Odometry, pred_topic, self.prediction_callback, 20
        )
        self.measurement_sub = self.create_subscription(
            Odometry, meas_topic, self.measurement_callback, 20
        )

        # ---------------------------------------------------------------
        # Publisher
        # ---------------------------------------------------------------
        self.pub = self.create_publisher(Odometry, self.output_topic, 10)

        # ---------------------------------------------------------------
        # TF Broadcaster
        # ---------------------------------------------------------------
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info("EKF node started.")

    # ---------------------------------------------------------------
    # Prediction Step
    # ---------------------------------------------------------------
    def prediction_callback(self, msg: Odometry):
        now = self.get_clock().now().nanoseconds / 1e9

        if self.last_time is None:
            self.last_time = now
            return

        dt = now - self.last_time
        self.last_time = now

        # Extract prediction inputs:
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z

        # Unpack state
        x, y, yaw, _, _ = self.x.flatten()

        # State prediction model
        x_pred = np.zeros((5, 1))
        x_pred[0] = x + v * math.cos(yaw) * dt
        x_pred[1] = y + v * math.sin(yaw) * dt
        x_pred[2] = yaw + w * dt
        x_pred[3] = v
        x_pred[4] = w

        # Jacobian of state transition model
        F = np.eye(5)
        F[0, 2] = -v * math.sin(yaw) * dt
        F[1, 2] = v * math.cos(yaw) * dt
        F[0, 3] = math.cos(yaw) * dt
        F[1, 3] = math.sin(yaw) * dt
        F[2, 4] = dt

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q

        # Update state
        self.x = x_pred

        # Publish result
        self.publish_state()

    # ---------------------------------------------------------------
    # Measurement Update
    # ---------------------------------------------------------------
    def measurement_callback(self, msg: Odometry):
        # Extract measurements
        z = np.zeros((5, 1))
        z[0] = msg.pose.pose.position.x
        z[1] = msg.pose.pose.position.y
        z[2] = quaternion_to_yaw(msg.pose.pose.orientation)
        z[3] = msg.twist.twist.linear.x
        z[4] = msg.twist.twist.angular.z

        # Observation model: identity (direct measurement)
        H = np.eye(5)

        # Innovation
        y = z - H @ self.x

        # Normalize angle residual
        y[2] = (y[2] + math.pi) % (2 * math.pi) - math.pi

        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I = np.eye(5)
        self.P = (I - K @ H) @ self.P

        # Publish fused state
        self.publish_state()

    # ---------------------------------------------------------------
    # Publish Odometry + TF
    # ---------------------------------------------------------------
    def publish_state(self):
        x, y, yaw, v, w = self.x.flatten()

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.orientation = yaw_to_quaternion(yaw)

        odom.twist.twist.linear.x = float(v)
        odom.twist.twist.angular.z = float(w)

        self.pub.publish(odom)
        
        # ---- TF ----
        t = TransformStamped()
        t.header.stamp = odom.header.stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"

        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = 0.0
        t.transform.rotation = yaw_to_quaternion(yaw)

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
