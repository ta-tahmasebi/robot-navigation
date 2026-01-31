#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "Usage: $0 <goal_x> <goal_y> [robot_radius_cells]"
  exit 1
fi

GOAL_X="$1"
GOAL_Y="$2"
RADIUS="${3:-}"

NODE_NAME="/a_star_planner_node"
SERVICE_NAME="/plan_path"
SERVICE_TYPE="interface_srvices/srv/PlanPath"

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set +u
if [ -n "${ZSH_VERSION:-}" ]; then
  source "$WS_DIR/install/setup.zsh"
elif [ -n "${BASH_VERSION:-}" ]; then
  source "$WS_DIR/install/setup.bash"
else
  . "$WS_DIR/install/setup.sh"
fi
set -u

if [ -n "$RADIUS" ]; then
  ros2 param set "$NODE_NAME" robot_radius_cells "$RADIUS" >/dev/null
fi

REQ="{goal: {header: {frame_id: 'map'}, pose: {position: {x: ${GOAL_X}, y: ${GOAL_Y}, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}"

OUT="$(ros2 service call "$SERVICE_NAME" "$SERVICE_TYPE" "$REQ" 2>&1 || true)"

if echo "$OUT" | grep -Eq "success(=|:)[[:space:]]*True"; then
  echo "successful"
else
  MSG="$(echo "$OUT" | sed -n "s/.*message=\x27\([^']*\)\x27.*/\1/p" | tail -n 1)"
  if [ -n "$MSG" ]; then
    echo "$MSG"
  else
    echo "$OUT"
  fi
fi

