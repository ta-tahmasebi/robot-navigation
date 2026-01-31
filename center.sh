#!/usr/bin/env bash

set -e
WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${ZSH_VERSION:-}" ]; then
  source "$WS_DIR/install/setup.zsh"
elif [ -n "${BASH_VERSION:-}" ]; then
  source "$WS_DIR/install/setup.bash"
else
  . "$WS_DIR/install/setup.sh"
fi

ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{
  header: {frame_id: 'map'},
  pose: {
    pose: {
      position: {x: 11.0, y: 6.0, z: 0.0},
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
    },
    covariance: [
      0.25,0,0,0,0,0,
      0,0.25,0,0,0,0,
      0,0,0.0,0,0,0,
      0,0,0,0.0,0,0,
      0,0,0,0,0.0,0,
      0,0,0,0,0,0.25
    ]
  }
}"