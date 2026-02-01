#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Path

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


def normalize_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class PIDPathFollower(Node):
    def __init__(self):
        super().__init__('pid_path_follower')

        self.kp = 2.0
        self.ki = 0.0
        self.kd = 0.4

        self.max_linear = 0.35
        self.max_angular = 1.2

        self.lookahead_dist = 0.60
        self.goal_tolerance = 0.20
        self.slowdown_dist = 0.70

        self.integral_limit = 1.0

        self.have_pose = False
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0

        self.path_points = []

        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = self.get_clock().now()

        self._last_wait_state = None
        self._goal_announced = False

        qos_amcl = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        qos_path = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.sub_amcl = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.on_amcl_pose, qos_amcl
        )
        self.sub_path = self.create_subscription(
            Path, '/planned_path', self.on_path, qos_path
        )
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        self.timer = self.create_timer(0.05, self.control_step)

    def _log_wait_state(self, state: str | None):
        """Log only when wait state changes."""
        if state == self._last_wait_state:
            return
        self._last_wait_state = state

        if state == "amcl":
            self.get_logger().warn("Waiting for /amcl_pose...")
        elif state == "path":
            self.get_logger().warn("Waiting for /planned_path...")
        else:
            pass

    def on_amcl_pose(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.pose_x = float(p.x)
        self.pose_y = float(p.y)
        self.pose_yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        self.have_pose = True

    def on_path(self, msg: Path):
        pts = [(float(ps.pose.position.x), float(ps.pose.position.y)) for ps in msg.poses]
        self.path_points = pts

        if len(pts) == 1:
            self.get_logger().warn("Path has only 1 point.")
        elif len(pts) == 0:
            self.get_logger().warn("Received empty path.")

    def control_step(self):
        if not self.have_pose:
            self.pub_cmd.publish(Twist())
            self._log_wait_state("amcl")
            return

        if len(self.path_points) == 0:
            self.pub_cmd.publish(Twist())
            self._log_wait_state("path")
            return

        self._log_wait_state(None)

        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds / 1e9
        self.prev_time = now
        if dt <= 0.0:
            dt = 1e-3

        x, y, yaw = self.pose_x, self.pose_y, self.pose_yaw
        goal_x, goal_y = self.path_points[-1]
        dist_to_goal = math.hypot(goal_x - x, goal_y - y)

        if dist_to_goal <= self.goal_tolerance:
            self.pub_cmd.publish(Twist())
            if not self._goal_announced:
                self.get_logger().info("Goal reached.")
                self._goal_announced = True
            self.prev_error = 0.0
            self.integral = 0.0
            return
        else:
            self._goal_announced = False

        tx, ty = self.get_lookahead_target(x, y, self.lookahead_dist)

        desired_yaw = math.atan2(ty - y, tx - x)
        error = normalize_angle(desired_yaw - yaw)

        self.integral += error * dt
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)

        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        angular = self.kp * error + self.ki * self.integral + self.kd * derivative
        angular = max(min(angular, self.max_angular), -self.max_angular)

        near_goal_scale = min(1.0, dist_to_goal / self.slowdown_dist)
        heading_scale = max(0.0, 1.0 - (abs(error) / 1.2))
        linear = self.max_linear * near_goal_scale * heading_scale
        linear = max(0.05, linear)

        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self.pub_cmd.publish(cmd)

    def get_lookahead_target(self, x: float, y: float, lookahead: float):
        pts = self.path_points
        if len(pts) == 1:
            return pts[0]

        closest_idx = 0
        closest_dist = float('inf')
        for i, (px, py) in enumerate(pts):
            d = math.hypot(px - x, py - y)
            if d < closest_dist:
                closest_dist = d
                closest_idx = i

        acc = 0.0
        i = closest_idx
        while i < len(pts) - 1 and acc < lookahead:
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            acc += math.hypot(x2 - x1, y2 - y1)
            i += 1

        return pts[i]


def main(args=None):
    rclpy.init(args=args)
    node = PIDPathFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
