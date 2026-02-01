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

ros2 launch robot_description gazebo.launch.py
