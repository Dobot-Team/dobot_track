# MCAP Processor

A Python project for reading, analyzing and extracting data from ROS2 MCAP bags without requiring a full ROS2 installation.

## Features

- Read ROS2 MCAP bags (single `.mcap` file or rosbag2 directory)
- Decode standard ROS2 messages and custom message types via `mcap-ros2-support`
- Inspect topic statistics (message counts, durations, frequencies)
- Extract images (`sensor_msgs/Image`, `sensor_msgs/CompressedImage`)
- Extract IMU data to CSV/JSON
- Extract Livox/custom point clouds to CSV/NumPy
- **Time-synchronize VLA samples** using a camera topic as reference
- **Visualize synced samples** (image + pointcloud + IMU + offset distribution)
- Modular design for adding custom extractors

## Quick Start

```bash
cd ~/embodvla/tools/mcap_processor
source venv/bin/activate

# Inspect the example bag
python scripts/01_inspect_bag.py

# Extract first 50 images from all image topics
python scripts/02_extract_images.py --max-messages 50

# Extract IMU data to CSV
python scripts/03_extract_imu.py

# Extract point clouds
python scripts/04_extract_pointcloud.py --max-messages 10

# Synchronize VLA samples around the color camera
python scripts/05_sync_vla_samples.py --max-samples 20

# Use a custom sync configuration
python scripts/05_sync_vla_samples.py --config config/sync_config.yaml

# Batch process multiple bags (one per line in the file)
python scripts/05_sync_vla_samples.py \
  --batch-file /path/to/bag_list.txt \
  --output-dir output/synced_vla_samples \
  --max-samples 20

# Visualize synced samples
python scripts/06_visualize_synced_samples.py --random --plot-offsets --plot-imu
```

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Or install as editable package
pip install -e .
```

## Project Structure

```
mcap_processor/
├── src/mcap_processor/      # Core library
│   ├── core.py              # BagReader
│   ├── analyzer.py          # BagAnalyzer / statistics
│   ├── utils.py             # Helpers
│   └── extractors/          # Extractors for specific message types
├── scripts/                 # Runnable examples
├── config/                  # Example configuration
├── ../../tests/test_mcap_reader.py  # Repository-level integration tests
├── data/                    # Place input bags here
└── output/                  # Extraction outputs
```

Run the integration test from the repository root:

```bash
MCAP_EXAMPLE_BAG=/path/to/bag pytest tests/test_mcap_reader.py
```

## Example Bag

The default example bag is:

```
/mnt/oss/data-collection/20260630/N-01-035-004-1782801171966089189/dobot_bag_0000
```

It is a Dobot robot recording containing topics such as:

- `/lower/state`, `/lower/cmd`
- `/camera/camera/color/image_raw`
- `/camera/camera/imu`
- `/livox_Lidar_node/xyz/pointcloud`
- `/livox_Lidar_node/imu/raw_data`
- Depth and compressed image topics

## VLA Sample Time Synchronization

The script `scripts/05_sync_vla_samples.py` builds training samples using `/camera/camera/color/image_raw` as the time reference. For each reference frame it finds the nearest:

- **extra cameras**:
  - `/camera/camera/infra1/image_rect_raw`
  - `/camera/camera/infra2/image_rect_raw`
  - `/camera/camera0/image_compressed`
  - `/camera/camera2/image_compressed`
- point cloud (`/livox_Lidar_node/xyz/pointcloud`)
- camera IMU (`/camera/camera/imu`)
- Livox IMU (`/livox_Lidar_node/imu/raw_data`)
- lower state (`/lower/state`)
- lower command (`/lower/cmd`)

Each valid sample is saved as:

```
output/synced_vla_samples/sample_<timestamp_ns>/
├── image.png -> reference color image
├── camera_camera_infra1_image_rect_raw.png
├── camera_camera_infra2_image_rect_raw.png
├── camera_camera0_image_compressed.jpg
├── camera_camera2_image_compressed.jpg
├── pointcloud.npy -> symlink to raw pointcloud
├── metadata.json
├── camera_camera_imu.json
├── livox_Lidar_node_imu_raw_data.json
├── lower_state.json
└── lower_cmd.json
```

A sample is dropped if no required sensor is found within its window, or if any required extra camera frame is missing within its configured window. Because cameras have different timestamps and frequencies, the number of multi-camera synced samples may be smaller than the number of reference frames.

### Batch processing

To process many bags at once, prepare a text file with one bag directory per line:

```text
/mnt/oss/data-collection/20260630/N-01-035-004-1782801171966089189/dobot_bag_0000
/mnt/oss/data-collection/20260630/N-01-035-004-1782801171966089189/dobot_bag_0001
/mnt/oss/data-collection/20260630/N-01-035-004-1782801171966089189/dobot_bag_0002
```

Then run:

```bash
python scripts/05_sync_vla_samples.py \
  --batch-file bag_list.txt \
  --output-dir output/synced_vla_samples \
  --image-output-dir output/images \
  --pointcloud-output-dir output/pointclouds \
  --imu-output-dir output/imu \
  --max-samples 20
```

In batch mode, outputs are placed in per-bag subdirectories so different bags never overwrite each other:

```
output/synced_vla_samples/
├── dobot_bag_0000/
│   ├── index.json
│   └── sample_<timestamp_ns>/
├── dobot_bag_0001/
│   ├── index.json
│   └── sample_<timestamp_ns>/
└── ...
```

The same applies to `--image-output-dir`, `--pointcloud-output-dir`, and `--imu-output-dir`.

### Configuration

Sync topics and time windows are configurable via `config/sync_config.yaml`:

```yaml
ref_topic: /camera/camera/color/image_raw
extra_camera_topics:
  - topic: /camera/camera/infra1/image_rect_raw
    window_ms: 30
  - topic: /camera/camera0/image_compressed
    window_ms: 15
sensor_topics:
  - topic: /livox_Lidar_node/xyz/pointcloud
    window_ms: 100
    required: true
  - topic: /camera/camera/imu
    window_ms: 20
    required: false
```

Run with `--config config/sync_config.yaml`. Without `--config`, the script uses sensible defaults.

## Visualizing Synced Samples

`scripts/06_visualize_synced_samples.py` generates static images for sanity checks:

```bash
# Random sample + offset distribution + IMU plots
python scripts/06_visualize_synced_samples.py --random --plot-offsets --plot-imu

# Generate N random samples at once
python scripts/06_visualize_synced_samples.py --num-random 10 --plot-offsets --plot-imu

# Specific sample by index
python scripts/06_visualize_synced_samples.py --sample-index 100 --plot-imu

# Only offset distribution
python scripts/06_visualize_synced_samples.py --plot-offsets
```

Output images are saved to `output/visualizations/`:
- `sample_<timestamp>.png` — camera image, pointcloud top-down/side view, sync metadata
- `offset_distribution.png` — histogram of absolute time offsets per topic
- `sample_<timestamp>_imu.png` — angular velocity and linear acceleration for both IMUs

## Adding a Custom Extractor

1. Create a new file under `src/mcap_processor/extractors/` inheriting from `BaseExtractor`.
2. Implement the `extract()` method.
3. Add a corresponding script under `scripts/`.

## Notes

- `mcap-ros2-support` decodes messages using schemas embedded in the MCAP file. If a custom message type fails to decode, the raw bytes are still accessible via the lower-level `mcap` API.
- For large bags, use `--max-messages` for quick validation before running full extraction.
