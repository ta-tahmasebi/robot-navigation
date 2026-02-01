from __future__ import annotations

import math
import os
import shutil
import subprocess
import threading
import time
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Path


class CFG:
    DT: float = 0.0
    REAL_TIME_SLEEP: bool = False
    MAX_EP_STEPS: int = 1500

    POSE_WAIT_TIMEOUT: float = 1.5
    PATH_WAIT_TIMEOUT: float = 3.0
    STALE_POSE_SEC: float = 1.0
    STALE_PATH_SEC: float = 5.0

    GOAL_TOL: float = 0.5 
    MAX_CTE_TERMINATE: float = 2.0

    W_MAX: float = 1.0

    V_MIN_STALL: float = 0.05 

    W_PROGRESS: float = 8.0             # reward per meter of positive progress
    W_CTE2: float = 2.0                 # penalty for cte^2
    W_HEADING: float = 0.35             # penalty for |heading_err|
    W_W2: float = 0.35                  # penalty for w_norm^2 (discourage hard turns)
    W_SMOOTH: float = 0.20              # penalty for delta action squared
    W_TIME: float = 0.03                # per-step time penalty (finish fast)
    STALL_PEN: float = 0.30             # penalty if v_norm too small
    GOAL_BONUS: float = 35.0            # reward for reaching goal
    OFFPATH_PENALTY: float = 12.0       # penalty for leaving path

    pass

_DEFAULT_RESET_POSE = (0.0, 0.0, 0.9, 0.0)  # x, y, z, yaw

class _RosNodeHolder:
    def __init__(self, node_name: str = "gazebo_env_async_node"):
        if not rclpy.ok():
            rclpy.init(args=None)

        self.node: Node = Node(node_name)
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

    def _spin(self) -> None:
        try:
            rclpy.spin(self.node)
        except Exception:
            pass

    def shutdown(self) -> None:
        try:
            self.node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


def _yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _wrap_to_pi(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


class GazeboEnv(gym.Env):

    metadata = {"render.modes": []}

    def __init__(
        self,
        world_name: str = "depot",
        cmd_topic: str = "/cmd_vel",
        amcl_pose_topic: str = "/amcl_pose",
        path_topic: str = "/planned_path",
        odom_topic: Optional[str] = None,
        robot_name_in_gz: str = "robot",
        reset_pose_xyzyaw: Tuple[float, float, float, float] = _DEFAULT_RESET_POSE,
        ign_bin: str = "ign",
        dt: float = CFG.DT,
        max_episode_steps: int = CFG.MAX_EP_STEPS,
        real_time_sleep: bool = CFG.REAL_TIME_SLEEP,
        frozen_path_topic: str = "/frozen_path",
        frozen_path_frame_id: str = "map",
        initialpose_topic: str = "/initialpose",
        initialpose_frame_id: str = "map",
        initialpose_xyyaw: Tuple[float, float, float] = (11.0, 6.0, 0.0),
        initialpose_cov_diag: Tuple[float, float, float] = (0.25, 0.25, 0.25),
        goal_tol: float = CFG.GOAL_TOL,
        lock_first_path_forever: bool = True,
    ):
        super().__init__()

        if (amcl_pose_topic is None or amcl_pose_topic == "") and odom_topic:
            amcl_pose_topic = odom_topic

        self._ros = _RosNodeHolder()

        self._world = world_name
        self._cmd_topic = cmd_topic
        self._amcl_pose_topic = amcl_pose_topic
        self._path_topic = path_topic

        self._robot_name_in_gz = robot_name_in_gz
        self._reset_pose = reset_pose_xyzyaw
        self._ign_bin = ign_bin

        self._dt = float(dt)
        self._max_episode_steps = int(max_episode_steps)
        self._real_time_sleep = bool(real_time_sleep)
        self._goal_tol = float(goal_tol)

        self._lock_first_path_forever = bool(lock_first_path_forever)

        self._frozen_path_topic = frozen_path_topic
        self._frozen_path_frame_id = frozen_path_frame_id

        self._initialpose_topic = initialpose_topic
        self._initialpose_frame_id = initialpose_frame_id
        self._initialpose_xyyaw = initialpose_xyyaw
        self._initialpose_cov_diag = initialpose_cov_diag

        self._step_count = 0

        self._cmd_pub = self._ros.node.create_publisher(Twist, self._cmd_topic, 10)

        qos_initialpose = QoSProfile(depth=1)
        qos_initialpose.reliability = QoSReliabilityPolicy.RELIABLE
        qos_initialpose.history = QoSHistoryPolicy.KEEP_LAST
        qos_initialpose.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self._initialpose_pub = self._ros.node.create_publisher(
            PoseWithCovarianceStamped, self._initialpose_topic, qos_initialpose
        )

        qos_frozen = QoSProfile(depth=1)
        qos_frozen.reliability = QoSReliabilityPolicy.RELIABLE
        qos_frozen.history = QoSHistoryPolicy.KEEP_LAST
        qos_frozen.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self._frozen_path_pub = self._ros.node.create_publisher(Path, self._frozen_path_topic, qos_frozen)

        qos_pose = QoSProfile(depth=5)
        qos_pose.reliability = QoSReliabilityPolicy.RELIABLE
        qos_pose.history = QoSHistoryPolicy.KEEP_LAST
        qos_pose.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        self._pose_lock = threading.Lock()
        self._last_pose: Optional[np.ndarray] = None
        self._pose_wall_ts: float = 0.0

        self._pose_sub = self._ros.node.create_subscription(
            PoseWithCovarianceStamped, self._amcl_pose_topic, self._pose_cb, qos_pose
        )

        qos_path = QoSProfile(depth=5)
        qos_path.reliability = QoSReliabilityPolicy.RELIABLE
        qos_path.history = QoSHistoryPolicy.KEEP_LAST
        qos_path.durability = QoSDurabilityPolicy.VOLATILE

        self._path_lock = threading.Lock()
        self._latest_path_xy: Optional[np.ndarray] = None
        self._path_wall_ts: float = 0.0

        self._path_sub = self._ros.node.create_subscription(Path, self._path_topic, self._path_cb, qos_path)

        self._global_path_xy: Optional[np.ndarray] = None
        self._global_cumlen: Optional[np.ndarray] = None
        self._global_total_len: float = 0.0

        self._prev_action = np.array([0.0, 0.0], dtype=np.float32)
        self._prev_s_m = 0.0
        self._last_obs = np.zeros(3, dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        obs_low = np.array([-1e6, -1e6, -math.pi], dtype=np.float32)
        obs_high = np.array([1e6, 1e6, math.pi], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

    def _pose_cb(self, msg: PoseWithCovarianceStamped) -> None:
        with self._pose_lock:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            yaw = _yaw_from_quat(msg.pose.pose.orientation)
            self._last_pose = np.array([x, y, yaw], dtype=np.float32)
            self._pose_wall_ts = time.time()

    def _path_cb(self, msg: Path) -> None:
        pts = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]
        if len(pts) < 2:
            return
        arr = np.asarray(pts, dtype=np.float32)
        with self._path_lock:
            self._latest_path_xy = arr
            self._path_wall_ts = time.time()

    def _publish_cmdvel(self, v_cmd: float, w_cmd: float) -> None:
        msg = Twist()
        msg.linear.x = float(v_cmd)
        msg.angular.z = float(w_cmd)
        self._cmd_pub.publish(msg)

    def _publish_initialpose(self) -> None:
        x, y, yaw = self._initialpose_xyyaw
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)

        cov_xx, cov_yy, cov_yaw = self._initialpose_cov_diag

        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self._initialpose_frame_id
        msg.header.stamp = self._ros.node.get_clock().now().to_msg()

        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = float(qz)
        msg.pose.pose.orientation.w = float(qw)

        cov = [0.0] * 36
        cov[0] = float(cov_xx)
        cov[7] = float(cov_yy)
        cov[35] = float(cov_yaw)
        msg.pose.covariance = cov

        for _ in range(3):
            self._initialpose_pub.publish(msg)
            time.sleep(0.02)

    def _publish_frozen_path(self, path_xy: np.ndarray) -> None:
        msg = Path()
        msg.header.frame_id = self._frozen_path_frame_id
        msg.header.stamp = self._ros.node.get_clock().now().to_msg()
        poses = []
        for (x, y) in path_xy:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            poses.append(ps)
        msg.poses = poses

        for _ in range(3):
            msg.header.stamp = self._ros.node.get_clock().now().to_msg()
            self._frozen_path_pub.publish(msg)
            time.sleep(0.02)

    def _wait_for_pose(self, timeout: float) -> Optional[np.ndarray]:
        t0 = time.time()
        while (time.time() - t0) < timeout:
            with self._pose_lock:
                if self._last_pose is not None and (time.time() - self._pose_wall_ts) <= CFG.STALE_POSE_SEC:
                    return self._last_pose.copy()
            time.sleep(0.005)
        with self._pose_lock:
            if self._last_pose is None:
                return None
            if (time.time() - self._pose_wall_ts) > CFG.STALE_POSE_SEC:
                return None
            return self._last_pose.copy()

    def _wait_for_latest_path(self, timeout: float) -> Optional[np.ndarray]:
        t0 = time.time()
        while (time.time() - t0) < timeout:
            with self._path_lock:
                if (
                    self._latest_path_xy is not None
                    and len(self._latest_path_xy) >= 2
                    and (time.time() - self._path_wall_ts) <= CFG.STALE_PATH_SEC
                ):
                    return self._latest_path_xy.copy()
            time.sleep(0.01)
        with self._path_lock:
            if self._latest_path_xy is None:
                return None
            if (time.time() - self._path_wall_ts) > CFG.STALE_PATH_SEC:
                return None
            return self._latest_path_xy.copy()

    def _gazebo_set_pose(self, x: float, y: float, z: float, yaw: float) -> bool:
        ign = shutil.which(self._ign_bin)
        if ign is None:
            print(f"[GazeboEnv] reset failed: '{self._ign_bin}' not found on PATH")
            return False

        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)

        service = f"/world/{self._world}/set_pose"
        req = (
            f'name: "{self._robot_name_in_gz}" '
            f'position: {{x: {x}, y: {y}, z: {z}}} '
            f'orientation: {{x: 0.0, y: 0.0, z: {qz}, w: {qw}}}'
        )

        cmd = [
            ign, "service",
            "-s", service,
            "--reqtype", "ignition.msgs.Pose",
            "--reptype", "ignition.msgs.Boolean",
            "--timeout", "2000",
            "--req", req,
        ]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ,
                timeout=5.0,
            )
            if proc.returncode == 0:
                return True
            stderr = proc.stderr.decode("utf-8", errors="ignore")
            print(f"[GazeboEnv] ign set_pose failed rc={proc.returncode} stderr:\n{stderr}")
            return False
        except Exception as e:
            print(f"[GazeboEnv] ign set_pose exception: {e}")
            return False

    @staticmethod
    def _precompute_cumlen(path_xy: np.ndarray) -> Tuple[np.ndarray, float]:
        diffs = path_xy[1:] - path_xy[:-1]
        seglen = np.sqrt(np.sum(diffs * diffs, axis=1))
        cum = np.zeros((len(path_xy),), dtype=np.float32)
        cum[1:] = np.cumsum(seglen, dtype=np.float32)
        total = float(cum[-1])
        return cum, total

    def _project_onto_polyline(
        self, pos_xy: np.ndarray, path_xy: np.ndarray, cumlen: np.ndarray
    ) -> Tuple[float, float, float]:
        best_d2 = float("inf")
        best_s = 0.0
        best_tang = 0.0

        p = pos_xy.astype(np.float32)

        for i in range(len(path_xy) - 1):
            a = path_xy[i]
            b = path_xy[i + 1]
            ab = b - a
            ab2 = float(np.dot(ab, ab))
            if ab2 < 1e-12:
                continue
            t = float(np.dot(p - a, ab) / ab2)
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            proj = a + t * ab
            d = p - proj
            d2 = float(np.dot(d, d))
            if d2 < best_d2:
                best_d2 = d2
                seg_len = float(math.sqrt(ab2))
                best_s = float(cumlen[i] + t * seg_len)
                best_tang = float(math.atan2(float(ab[1]), float(ab[0])))

        return best_s, float(math.sqrt(best_d2)), best_tang

    def _ensure_global_path(self) -> bool:
        if self._global_path_xy is not None and self._global_cumlen is not None:
            return True
        latest_path = self._wait_for_latest_path(timeout=CFG.PATH_WAIT_TIMEOUT)
        if latest_path is None:
            return False
        self._global_path_xy = latest_path.copy()
        self._global_cumlen, self._global_total_len = self._precompute_cumlen(self._global_path_xy)
        self._publish_frozen_path(self._global_path_xy)
        print(f"[GazeboEnv] Captured GLOBAL path once: N={len(self._global_path_xy)} len={self._global_total_len:.2f}m")
        return True

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        assert self.action_space.contains(action), f"Action out of bounds: {action}"

        self._step_count += 1

        v_norm = float(action[0])
        w_norm = float(action[1])
        v_cmd = v_norm
        w_cmd = w_norm * CFG.W_MAX

        self._publish_cmdvel(v_cmd, w_cmd)

        if self._real_time_sleep and self._dt > 0:
            time.sleep(self._dt)

        pose = self._wait_for_pose(timeout=0.0)
        if pose is None or self._global_path_xy is None or self._global_cumlen is None:
            obs = self._last_obs.copy()
            reward = -1.0
            terminated = False
            truncated = bool(self._step_count >= self._max_episode_steps)
            info = {"missing": True}
            return obs, float(reward), terminated, truncated, info

        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])

        obs = np.array([x, y, yaw], dtype=np.float32)
        self._last_obs = obs
        pos_xy = np.array([x, y], dtype=np.float32)
        s_m, cte, tang_yaw = self._project_onto_polyline(pos_xy, self._global_path_xy, self._global_cumlen)
        dprog_m = float(s_m - self._prev_s_m)
        self._prev_s_m = s_m
        dprog_pos = float(max(0.0, dprog_m))

        heading_err = _wrap_to_pi(tang_yaw - yaw)
        da = action - self._prev_action
        smooth_pen = float(np.dot(da, da))
        self._prev_action = action.copy()
        goal_xy = self._global_path_xy[-1]
        dist_goal = float(math.hypot(float(goal_xy[0] - x), float(goal_xy[1] - y)))
        reached_goal = dist_goal <= self._goal_tol
        off_path = cte > CFG.MAX_CTE_TERMINATE

        terminated = bool(reached_goal or off_path)
        truncated = bool(self._step_count >= self._max_episode_steps)

        stall = v_norm < CFG.V_MIN_STALL
        reward = (
            CFG.W_PROGRESS * dprog_pos
            - CFG.W_CTE2 * (cte * cte)
            - CFG.W_HEADING * abs(heading_err)
            - CFG.W_W2 * (w_norm * w_norm)
            - CFG.W_SMOOTH * smooth_pen
            - CFG.W_TIME
        )
        if stall:
            reward -= CFG.STALL_PEN
        if reached_goal:
            reward += CFG.GOAL_BONUS
        if off_path:
            reward -= CFG.OFFPATH_PENALTY

        info = {
            "s_m": s_m,
            "dprog_m": dprog_m,
            "dprog_pos": dprog_pos,
            "cte": cte,
            "heading_err": heading_err,
            "dist_goal": dist_goal,
            "stall": stall,
            "smooth_pen": smooth_pen,
            "reached_goal": reached_goal,
            "off_path": off_path,
            "fixed_path_len_m": self._global_total_len,
            "v_cmd": v_cmd,
            "w_cmd": w_cmd,
        }
        return obs, float(reward), terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._step_count = 0
        self._prev_action[:] = 0.0
        self._prev_s_m = 0.0

        self._publish_cmdvel(0.0, 0.0)
        time.sleep(0.05)

        x, y, z, yaw = self._reset_pose
        ok = self._gazebo_set_pose(x, y, z, yaw)
        if not ok:
            print("[GazeboEnv] Warning: set_pose failed. Continuing anyway.")

        self._publish_initialpose()
        time.sleep(0.15)

        pose = self._wait_for_pose(timeout=CFG.POSE_WAIT_TIMEOUT)

        if self._lock_first_path_forever:
            got = self._ensure_global_path()
            if got and self._global_path_xy is not None:
                self._publish_frozen_path(self._global_path_xy)
        else:
            latest_path = self._wait_for_latest_path(timeout=CFG.PATH_WAIT_TIMEOUT)
            if latest_path is not None:
                self._global_path_xy = latest_path.copy()
                self._global_cumlen, self._global_total_len = self._precompute_cumlen(self._global_path_xy)
                self._publish_frozen_path(self._global_path_xy)

        if pose is None or self._global_path_xy is None or self._global_cumlen is None:
            obs = np.zeros(3, dtype=np.float32)
            self._last_obs = obs
            return obs, {"missing": True}

        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        obs = np.array([x, y, yaw], dtype=np.float32)
        self._last_obs = obs

        pos_xy = np.array([x, y], dtype=np.float32)
        s_m, cte, tang_yaw = self._project_onto_polyline(pos_xy, self._global_path_xy, self._global_cumlen)
        self._prev_s_m = s_m

        return obs, {
            "s0_m": s_m,
            "cte0": cte,
            "tang0_yaw": tang_yaw,
            "fixed_path_len_m": self._global_total_len,
            "frozen_path_topic": self._frozen_path_topic,
            "global_path_locked": self._lock_first_path_forever,
        }

    def close(self) -> None:
        try:
            self._publish_cmdvel(0.0, 0.0)
            time.sleep(0.02)
        except Exception:
            pass
        try:
            self._ros.shutdown()
        except Exception:
            pass
