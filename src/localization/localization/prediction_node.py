import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import math


def yaw_to_quaternion(yaw):
    """Convert yaw angle (rad) to quaternion."""
    return Quaternion(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0),
    )


class OdometryPredictor(Node):

    def __init__(self):
        super().__init__("odometry_predictor")

        # -------------------------
        # Load Parameters
        # -------------------------
        self.declare_parameter("wheel_radius", 0.10)
        self.declare_parameter("wheel_separation", 0.45)
        self.declare_parameter("left_angular_vel_topic", "/left_motor_angular_vel")
        self.declare_parameter("right_angular_vel_topic", "/right_motor_angular_vel")
        self.declare_parameter("odom_topic", "/prediction/odom")
        self.declare_parameter("prediction_rate", 50.0)

        self.wheel_radius = self.get_parameter("wheel_radius").value
        self.wheel_separation = self.get_parameter("wheel_separation").value
        left_topic = self.get_parameter("left_angular_vel_topic").value
        right_topic = self.get_parameter("right_angular_vel_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        prediction_rate = self.get_parameter("prediction_rate").value

        # -------------------------
        # Internal state
        # -------------------------
        self.left_angular_vel = 0.0
        self.right_angular_vel = 0.0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_time = self.get_clock().now().nanoseconds / 1e9

        # -------------------------
        # Subscribers
        # -------------------------
        self.left_sub = self.create_subscription(
            Float64, left_topic, self.left_callback, 10
        )
        self.right_sub = self.create_subscription(
            Float64, right_topic, self.right_callback, 10
        )

        # -------------------------
        # Publisher
        # -------------------------
        self.odom_pub = self.create_publisher(Odometry, odom_topic, 10)

        # -------------------------
        # Timer for prediction update loop
        # -------------------------
        self.period = 1.0 / prediction_rate
        self.timer = self.create_timer(self.period, self.update_prediction)

        self.get_logger().info("Odometry Predictor node started.")

    # ---------------------------------------------------------------
    # Callback functions
    # ---------------------------------------------------------------
    def left_callback(self, msg):
        self.left_angular_vel = msg.data

    def right_callback(self, msg):
        self.right_angular_vel = msg.data

    # ---------------------------------------------------------------
    # Prediction update step
    # ---------------------------------------------------------------
    def update_prediction(self):
        current_time = self.get_clock().now().nanoseconds / 1e9
        dt = current_time - self.last_time
        self.last_time = current_time

        # Wheel linear speeds (m/s)
        v_l = self.left_angular_vel * self.wheel_radius
        v_r = self.right_angular_vel * self.wheel_radius

        # Differential-drive motion model
        v = (v_r + v_l) / 2.0                       # forward linear velocity
        w = (v_r - v_l) / self.wheel_separation    # angular velocity        

        # Integrate pose
        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt
        self.yaw += w * dt

        # -----------------------------------------------------------
        # Publish Odometry
        # -----------------------------------------------------------
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        # Pose
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.orientation = yaw_to_quaternion(self.yaw)

        # Velocity
        odom_msg.twist.twist.linear.x = v
        odom_msg.twist.twist.angular.z = w

        self.odom_pub.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)
    node = OdometryPredictor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
