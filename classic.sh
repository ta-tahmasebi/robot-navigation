#!/usr/bin/env bash

set -euo pipefail

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

MODE="${1:-}"

usage() {
  echo "Usage: $0 {pid|mpc}"
  exit 2
}

if [ -z "$MODE" ]; then
  usage
fi

case "$MODE" in
  pid)
    exec ros2 run path_following_controller pid_path_follower
    ;;
  mpc)
    exec ros2 run path_following_controller mpc_path_follower
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown mode: '$MODE'"
    usage
    ;;
esac
