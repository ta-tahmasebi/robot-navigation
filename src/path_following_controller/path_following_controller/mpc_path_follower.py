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


class MPCPathFollower(Node):
    def __init__(self):
        super().__init__('mpc_path_follower')

        self.v_min = 0.05
        self.v_max = 0.35
        self.w_max = 1.2
        self.dt = 0.10
        self.N = 15
        self.v_samples = 5
        self.w_samples = 13
        self.goal_tolerance = 0.20
        self.slowdown_dist = 0.80
        self.w_pos = 6.0
        self.w_yaw = 1.5
        self.w_u = 0.15
        self.w_du = 0.25
        self.have_pose = False
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0
        self.path_points = []
        self.last_u = (0.0, 0.0)

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

        self.timer = self.create_timer(0.10, self.control_step)

    def _log_wait_state(self, state: str | None):
        if state == self._last_wait_state:
            return
        self._last_wait_state = state
        if state == "amcl":
            self.get_logger().warn("Waiting for /amcl_pose...")
        elif state == "path":
            self.get_logger().warn("Waiting for /planned_path...")

    def on_amcl_pose(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.pose_x = float(p.x)
        self.pose_y = float(p.y)
        self.pose_yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        self.have_pose = True

    def on_path(self, msg: Path):
        self.path_points = [(float(ps.pose.position.x), float(ps.pose.position.y)) for ps in msg.poses]
        if len(self.path_points) == 0:
            self.get_logger().warn("Received empty path.")
        elif len(self.path_points) == 1:
            self.get_logger().warn("Path has only 1 point.")

    def control_step(self):
        # waiting
        if not self.have_pose:
            self.pub_cmd.publish(Twist())
            self._log_wait_state("amcl")
            return
        if len(self.path_points) == 0:
            self.pub_cmd.publish(Twist())
            self._log_wait_state("path")
            return
        self._log_wait_state(None)

        x0, y0, th0 = self.pose_x, self.pose_y, self.pose_yaw

        goal_x, goal_y = self.path_points[-1]
        dist_goal = math.hypot(goal_x - x0, goal_y - y0)

        if dist_goal <= self.goal_tolerance:
            self.pub_cmd.publish(Twist())
            if not self._goal_announced:
                self.get_logger().info("Goal reached.")
                self._goal_announced = True
            self.last_u = (0.0, 0.0)
            return
        self._goal_announced = False
        ref = self.build_reference_horizon(x0, y0, dist_goal)
        v_cmd, w_cmd = self.solve_mpc_sampling(x0, y0, th0, ref, dist_goal)

        cmd = Twist()
        cmd.linear.x = float(v_cmd)
        cmd.angular.z = float(w_cmd)
        self.pub_cmd.publish(cmd)

        self.last_u = (v_cmd, w_cmd)

    def solve_mpc_sampling(self, x0, y0, th0, ref, dist_goal):
        near_goal_scale = min(1.0, dist_goal / self.slowdown_dist)
        v_max_eff = max(self.v_min, self.v_max * near_goal_scale)
        v_grid = self.linspace(self.v_min, v_max_eff, self.v_samples)
        w_grid = self.linspace(-self.w_max, self.w_max, self.w_samples)

        best_cost = float('inf')
        best_u = (0.0, 0.0)

        last_v, last_w = self.last_u

        for v in v_grid:
            for w in w_grid:
                x, y, th = x0, y0, th0
                cost = 0.0

                dv0 = v - last_v
                dw0 = w - last_w
                cost += self.w_du * (dv0 * dv0 + dw0 * dw0)

                for k in range(self.N):
                    rx, ry, rth = ref[k]
                    ex = rx - x
                    ey = ry - y
                    epos2 = ex * ex + ey * ey
                    eth = normalize_angle(rth - th)

                    cost += self.w_pos * epos2 + self.w_yaw * (eth * eth)
                    cost += self.w_u * (v * v + 0.5 * w * w)
                    x, y, th = self.sim_step(x, y, th, v, w, self.dt)
                    if k == self.N - 1:
                        exT = ref[-1][0] - x
                        eyT = ref[-1][1] - y
                        cost += 2.0 * self.w_pos * (exT * exT + eyT * eyT)

                if cost < best_cost:
                    best_cost = cost
                    best_u = (v, w)

        return best_u

    def sim_step(self, x, y, th, v, w, dt):
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        th = normalize_angle(th + w * dt)
        return x, y, th

    def build_reference_horizon(self, x, y, dist_goal):
        pts = self.path_points
        if len(pts) == 1:
            gx, gy = pts[0]
            yaw = math.atan2(gy - y, gx - x)
            return [(gx, gy, yaw)] * self.N

        closest_idx = self.closest_waypoint_index(x, y)
        near_goal_scale = min(1.0, dist_goal / self.slowdown_dist)
        v_nom = max(self.v_min, self.v_max * near_goal_scale)
        ds_target = max(0.05, v_nom * self.dt)

        ref = []
        idx = closest_idx
        seg_progress = 0.0

        while len(ref) < self.N:
            if idx >= len(pts) - 1:
                gx, gy = pts[-1]
                yaw = math.atan2(gy - y, gx - x) if len(ref) == 0 else ref[-1][2]
                ref.append((gx, gy, yaw))
                continue

            x1, y1 = pts[idx]
            x2, y2 = pts[idx + 1]
            seg_len = math.hypot(x2 - x1, y2 - y1)

            if seg_len < 1e-6:
                idx += 1
                continue

            seg_progress += ds_target
            while seg_progress > seg_len and idx < len(pts) - 2:
                seg_progress -= seg_len
                idx += 1
                x1, y1 = pts[idx]
                x2, y2 = pts[idx + 1]
                seg_len = math.hypot(x2 - x1, y2 - y1)
                if seg_len < 1e-6:
                    seg_len = 1e-6

            t = max(0.0, min(1.0, seg_progress / seg_len))
            rx = x1 + t * (x2 - x1)
            ry = y1 + t * (y2 - y1)
            rth = math.atan2(y2 - y1, x2 - x1)

            ref.append((rx, ry, rth))

        return ref

    def closest_waypoint_index(self, x, y):
        pts = self.path_points
        best_i = 0
        best_d = float('inf')
        for i, (px, py) in enumerate(pts):
            d = math.hypot(px - x, py - y)
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    @staticmethod
    def linspace(a, b, n):
        if n <= 1:
            return [b]
        step = (b - a) / (n - 1)
        return [a + i * step for i in range(n)]


def main(args=None):
    rclpy.init(args=args)
    node = MPCPathFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
