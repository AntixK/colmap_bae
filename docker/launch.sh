#!/bin/bash
# Launch the COLMAP Docker container in interactive mode with a shell.
# Usage: ./launch.sh [host_directory]
# If no directory is provided, the current directory is used.

set -e

# Resolve the host directory to mount.
if [ $# -ge 1 ]; then
    HOST_DIR=$(realpath "$1")
else
    HOST_DIR=$(pwd)
fi

if [ ! -d "$HOST_DIR" ]; then
    echo "Error: Directory '$HOST_DIR' does not exist."
    exit 1
fi

# Select the Docker image: prefer local build, fall back to official.
if docker image inspect colmap:latest >/dev/null 2>&1; then
    COLMAP_IMAGE="colmap:latest"
else
    echo "Local image not found, pulling colmap/colmap:latest..."
    docker pull colmap/colmap:latest
    COLMAP_IMAGE="colmap/colmap:latest"
fi

# Base Docker arguments.
DOCKER_ARGS=(
    -it --rm
    -v "${HOST_DIR}:/working"
    -w /working
)

# Enable GPU if available.
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    DOCKER_ARGS+=(--gpus all)
fi

echo "Launching interactive shell with $COLMAP_IMAGE"
echo "  Mounted: $HOST_DIR -> /working"
docker run "${DOCKER_ARGS[@]}" "$COLMAP_IMAGE" bash
