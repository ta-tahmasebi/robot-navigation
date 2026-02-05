import math
import heapq
from typing import Dict, Tuple, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path

from interface_srvices.srv import PlanPath


class AStarPlanner(Node):
    def __init__(self):
        super().__init__('a_star_planner')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('amcl_pose_topic', '/amcl_pose')
        self.declare_parameter('path_topic', '/planned_path')
        self.declare_parameter('lethal_threshold', 50)
        self.declare_parameter('allow_unknown', False)

        self.declare_parameter('replan_on_pose_update', True)
        self.declare_parameter('min_replan_period_sec', 0.5)
        self.declare_parameter('min_pose_delta_m', 0.10)

        self.declare_parameter('auto_start', True)
        self.declare_parameter('auto_goal_x', 28.0)
        self.declare_parameter('auto_goal_y', 1.0)

        self.declare_parameter('robot_radius_cells', 10)

        self.declare_parameter('publish_period_sec', 1.0)

        self.map_topic = str(self.get_parameter('map_topic').value)
        self.amcl_pose_topic = str(self.get_parameter('amcl_pose_topic').value)
        self.path_topic = str(self.get_parameter('path_topic').value)

        self.lethal_threshold = int(self.get_parameter('lethal_threshold').value)
        self.allow_unknown = bool(self.get_parameter('allow_unknown').value)

        self.replan_on_pose_update = bool(self.get_parameter('replan_on_pose_update').value)
        self.min_replan_period_sec = float(self.get_parameter('min_replan_period_sec').value)
        self.min_pose_delta_m = float(self.get_parameter('min_pose_delta_m').value)

        self.auto_start = bool(self.get_parameter('auto_start').value)
        self.auto_goal_x = float(self.get_parameter('auto_goal_x').value)
        self.auto_goal_y = float(self.get_parameter('auto_goal_y').value)
        self.auto_goal_set = False

        self.robot_radius_cells = int(self.get_parameter('robot_radius_cells').value)

        self.publish_period_sec = float(self.get_parameter('publish_period_sec').value)

        self.map_msg: Optional[OccupancyGrid] = None
        self.map_data: List[int] = []
        self.width = 0
        self.height = 0
        self.resolution = 0.0
        self.origin_x = 0.0
        self.origin_y = 0.0

        self.inflated_occ: Optional[List[bool]] = None
        self.inflation_offsets: List[Tuple[int, int]] = []
        self._build_inflation_offsets()

        self.current_pose_map: Optional[PoseStamped] = None
        self.active_goal: Optional[PoseStamped] = None

        self.last_plan_time_sec: float = 0.0
        self.last_pose_for_plan: Optional[PoseStamped] = None

        self.last_path: Optional[Path] = None
        self.publish_enabled = False

        map_qos = QoSProfile(depth=1)
        map_qos.reliability = QoSReliabilityPolicy.RELIABLE
        map_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        amcl_qos = QoSProfile(depth=1)
        amcl_qos.reliability = QoSReliabilityPolicy.RELIABLE
        amcl_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.on_map,
            map_qos
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.amcl_pose_topic,
            self.on_amcl_pose,
            amcl_qos
        )

        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.plan_srv = self.create_service(PlanPath, 'plan_path', self.on_plan_path)

        self.add_on_set_parameters_callback(self.on_params)

        self.publish_timer = self.create_timer(self.publish_period_sec, self.publish_last_path)

    def publish_last_path(self):
        if not self.publish_enabled:
            return
        if self.last_path is None:
            return
        self.last_path.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(self.last_path)

    def on_params(self, params):
        need_reinflate = False
        for p in params:
            if p.name == 'robot_radius_cells':
                try:
                    v = int(p.value)
                except Exception:
                    return SetParametersResult(successful=False, reason='robot_radius_cells must be an integer')
                if v < 0:
                    return SetParametersResult(successful=False, reason='robot_radius_cells must be >= 0')
                self.robot_radius_cells = v
                need_reinflate = True

        if need_reinflate:
            self._build_inflation_offsets()
            if self.map_msg is not None:
                self._inflate_from_current_map()
                if self.current_pose_map is not None and self.active_goal is not None:
                    path = self.plan(self.current_pose_map, self.active_goal)
                    if path is not None:
                        self.last_path = path
                        self.last_plan_time_sec = self.get_clock().now().nanoseconds / 1e9
                        self.last_pose_for_plan = self.current_pose_map

        return SetParametersResult(successful=True)

    def _build_inflation_offsets(self):
        r = max(0, int(self.robot_radius_cells))
        offs: List[Tuple[int, int]] = []
        rr = r * r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= rr:
                    offs.append((dx, dy))
        self.inflation_offsets = offs

    def _inflate_from_current_map(self):
        if self.map_msg is None:
            self.inflated_occ = None
            return

        inflated = [False] * (self.width * self.height)

        for y in range(self.height):
            base = y * self.width
            for x in range(self.width):
                v = int(self.map_data[base + x])
                occ = (v >= self.lethal_threshold) or (v < 0 and not self.allow_unknown)
                if not occ:
                    continue
                for dx, dy in self.inflation_offsets:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        inflated[ny * self.width + nx] = True

        self.inflated_occ = inflated

    def on_map(self, msg: OccupancyGrid):
        self.map_msg = msg
        self.width = int(msg.info.width)
        self.height = int(msg.info.height)
        self.resolution = float(msg.info.resolution)
        self.origin_x = float(msg.info.origin.position.x)
        self.origin_y = float(msg.info.origin.position.y)
        self.map_data = list(msg.data)

        self._inflate_from_current_map()
        self.maybe_set_auto_goal()

    def on_amcl_pose(self, msg: PoseWithCovarianceStamped):
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose = msg.pose.pose
        self.current_pose_map = ps

        self.maybe_set_auto_goal()

        if not self.replan_on_pose_update:
            return
        if self.active_goal is None:
            return
        if self.map_msg is None or self.inflated_occ is None:
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        if (now_sec - self.last_plan_time_sec) < self.min_replan_period_sec:
            return

        if self.last_pose_for_plan is not None:
            dx = self.current_pose_map.pose.position.x - self.last_pose_for_plan.pose.position.x
            dy = self.current_pose_map.pose.position.y - self.last_pose_for_plan.pose.position.y
            if math.hypot(dx, dy) < self.min_pose_delta_m:
                return

        path = self.plan(self.current_pose_map, self.active_goal)
        if path is not None:
            self.last_path = path
            self.last_plan_time_sec = now_sec
            self.last_pose_for_plan = self.current_pose_map

    def maybe_set_auto_goal(self):
        if not self.auto_start:
            return
        if self.auto_goal_set:
            return
        if self.map_msg is None or self.inflated_occ is None:
            return

        g = PoseStamped()
        g.header.frame_id = 'map'
        g.header.stamp = self.get_clock().now().to_msg()
        g.pose.position.x = float(self.auto_goal_x)
        g.pose.position.y = float(self.auto_goal_y)
        g.pose.position.z = 0.0
        g.pose.orientation.x = 0.0
        g.pose.orientation.y = 0.0
        g.pose.orientation.z = 0.0
        g.pose.orientation.w = 1.0

        self.active_goal = g
        self.auto_goal_set = True

    def on_plan_path(self, request: PlanPath.Request, response: PlanPath.Response):
        if self.map_msg is None or self.inflated_occ is None:
            response.path = Path()
            response.success = False
            response.message = 'No map received yet.'
            return response

        if self.current_pose_map is None:
            response.path = Path()
            response.success = False
            response.message = 'No AMCL pose received yet.'
            return response

        self.publish_enabled = True

        goal = request.goal
        if goal.header.frame_id == '':
            goal.header.frame_id = 'map'

        self.active_goal = goal
        self.auto_goal_set = True

        path = self.plan(self.current_pose_map, goal)
        if path is None:
            response.path = Path()
            response.success = False
            response.message = 'Failed to find a collision-free path.'
            return response

        self.last_path = path
        self.last_plan_time_sec = self.get_clock().now().nanoseconds / 1e9
        self.last_pose_for_plan = self.current_pose_map

        response.path = path
        response.success = True
        response.message = 'OK'
        return response

    def world_to_grid(self, wx: float, wy: float) -> Optional[Tuple[int, int]]:
        mx = int(math.floor((wx - self.origin_x) / self.resolution))
        my = int(math.floor((wy - self.origin_y) / self.resolution))
        if mx < 0 or my < 0 or mx >= self.width or my >= self.height:
            return None
        return mx, my

    def grid_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        wx = self.origin_x + (mx + 0.5) * self.resolution
        wy = self.origin_y + (my + 0.5) * self.resolution
        return wx, wy

    def is_free(self, mx: int, my: int) -> bool:
        if self.inflated_occ is None:
            return False
        return not self.inflated_occ[my * self.width + mx]

    def plan(self, start_pose: PoseStamped, goal_pose: PoseStamped) -> Optional[Path]:
        s = self.world_to_grid(start_pose.pose.position.x, start_pose.pose.position.y)
        g = self.world_to_grid(goal_pose.pose.position.x, goal_pose.pose.position.y)
        if s is None or g is None:
            return None

        sx, sy = s
        gx, gy = g

        if not self.is_free(sx, sy) or not self.is_free(gx, gy):
            return None

        def h(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return math.hypot(b[0] - a[0], b[1] - a[1])

        moves = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)),
        ]

        start = (sx, sy)
        goal = (gx, gy)

        open_heap: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (h(start, goal), start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        closed = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                return self.build_path(came_from, goal_pose, current)

            closed.add(current)
            cx, cy = current

            for ox, oy, step in moves:
                nx = cx + ox
                ny = cy + oy
                if nx < 0 or ny < 0 or nx >= self.width or ny >= self.height:
                    continue
                if not self.is_free(nx, ny):
                    continue

                # prevent corner-cutting through obstacles
                if ox != 0 and oy != 0:
                    if not self.is_free(cx + ox, cy) or not self.is_free(cx, cy + oy):
                        continue

                nbr = (nx, ny)
                tentative = g_score[current] + step
                if tentative < g_score.get(nbr, float('inf')):
                    came_from[nbr] = current
                    g_score[nbr] = tentative
                    f = tentative + h(nbr, goal)
                    heapq.heappush(open_heap, (f, nbr))

        return None

    def build_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        goal_pose: PoseStamped,
        current: Tuple[int, int]
    ) -> Path:
        cells = [current]
        while current in came_from:
            current = came_from[current]
            cells.append(current)
        cells.reverse()

        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()

        poses: List[PoseStamped] = []
        for mx, my in cells:
            wx, wy = self.grid_to_world(mx, my)
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(wx)
            ps.pose.position.y = float(wy)
            ps.pose.position.z = 0.0
            ps.pose.orientation.x = 0.0
            ps.pose.orientation.y = 0.0
            ps.pose.orientation.z = 0.0
            ps.pose.orientation.w = 1.0
            poses.append(ps)

        path.poses = poses
        return path


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
