#!/bin/bash
set -e

export ROS_DOMAIN_ID=76
source /opt/ros/humble/setup.bash
source ~/dobot_quad_navinterface/install/setup.bash

BAG_PLAY_DIR="$1"
ODOM_BAG_DIR="$2"
MAP_DIR="${3:-${ODOM_BAG_DIR}_map}"
DURATION="${4:-60}"
LIDAR_EXTRINSIC_YAML="${5:-/tmp/device_static_params/devices/robot_dog_v1/lidar/extrinsic/laser_link_to_base_link.yaml}"

# The launch file now loads base_link->lidar extrinsics from device_static_params.
# Callers must copy the submodule into the container first:
#   docker cp third_party/device_static_params dobot_build:/tmp/device_static_params
if [ ! -f "$LIDAR_EXTRINSIC_YAML" ]; then
    echo "Error: LiDAR extrinsic YAML not found at $LIDAR_EXTRINSIC_YAML"
    echo "Please copy device_static_params into the container before running SLAM:"
    echo "  docker cp <host_path>/third_party/device_static_params dobot_build:/tmp/device_static_params"
    exit 1
fi

# Resolve bag directory
if [ ! -f "$BAG_PLAY_DIR/metadata.yaml" ]; then
    found_bag=$(find "$BAG_PLAY_DIR" -maxdepth 2 -name metadata.yaml -print -quit 2>/dev/null)
    if [ -n "$found_bag" ]; then
        BAG_PLAY_DIR="$(dirname "$found_bag")"
        echo "Resolved bag directory to: $BAG_PLAY_DIR"
    else
        echo "Error: no metadata.yaml found under $BAG_PLAY_DIR"
        exit 1
    fi
fi

# Remove output dirs so ros2 bag record accepts them
rm -rf "$ODOM_BAG_DIR"
rm -rf "$MAP_DIR"

# Clean up stale SLAM / TF / recorder processes from previous runs to avoid
# multiple nodes publishing on /lightning/odometry with conflicting timestamps.
echo "[0/3] Cleaning up stale SLAM/TF processes..."
pkill -9 -f "run_slam_online" 2>/dev/null || true
pkill -9 -f "static_transform_publisher.*base_link.*laser_link" 2>/dev/null || true
pkill -9 -f "ros2 bag record.*lightning/odometry" 2>/dev/null || true
sleep 2

# Use a unique log directory based on output dir to avoid races
LOG_DIR="$(dirname "$ODOM_BAG_DIR")/logs"
mkdir -p "$LOG_DIR"

echo "[1/3] Starting SLAM node..."
ros2 launch lightning run_slam_online_tf.launch.py \
  config:="/tmp/slam_run/default_dobot_headless.yaml" \
  lidar_extrinsic_yaml:="$LIDAR_EXTRINSIC_YAML" > "$LOG_DIR/slam.log" 2>&1 &
SLAM_PID=$!
sleep 4

echo "[2/3] Starting odometry recording..."
ros2 bag record -o "$ODOM_BAG_DIR" /lightning/odometry > "$LOG_DIR/record.log" 2>&1 &
RECORD_PID=$!
sleep 2

echo "[3/3] Playing bag for ${DURATION}s (using timeout)..."
timeout ${DURATION}s ros2 bag play "$BAG_PLAY_DIR" --topics /livox_Lidar_node/xyz/pointcloud /livox_Lidar_node/imu/raw_data || true

echo "Bag playback finished or timed out. Saving map before stopping SLAM..."
# Save map while SLAM is still alive; add a 10s timeout so a dead node does not hang forever.
timeout 10s ros2 service call /lightning/save_map lightning/srv/SaveMap "{map_id: '$MAP_DIR'}" >/dev/null 2>&1 || {
    echo "Warning: failed to save map (service call timed out or SLAM not responding)."
}

echo "Stopping processes..."
# Send SIGINT to recorder to close sqlite3 cleanly
kill -INT $RECORD_PID 2>/dev/null || true
sleep 2
# Give SLAM a graceful shutdown window, then force-kill if needed
kill -TERM $SLAM_PID 2>/dev/null || true
for i in {1..10}; do
    if ! kill -0 $SLAM_PID 2>/dev/null; then
        break
    fi
    sleep 1
done
kill -9 $SLAM_PID 2>/dev/null || true
wait 2>/dev/null || true

echo "Converting odometry to JSON..."
python3 /root/dobot_quad_mapping/script/bag_to_odom_json.py "$ODOM_BAG_DIR"

echo "Done. Odom bag: $ODOM_BAG_DIR"
