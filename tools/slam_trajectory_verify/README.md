# SLAM 轨迹真值可视化

本目录包含一套脚本，用于：

1. 在 Docker 容器 (`dobot_build`) 中运行 `dobot_quad_mapping` 的 LiDAR SLAM，获取 `base_link` 轨迹真值；
2. 从 MCAP rosbag 中同步抽取彩色相机图像；
3. 将每张图像与最近时间戳的 SLAM 位姿匹配，逐帧渲染“相机画面 + 2D 轨迹”叠加图，方便人工校验轨迹是否正确。

## 前置条件

- 当前主机已有一个名为 `dobot_build` 的 Docker 容器（镜像 `docker.cnb.cool/gpf2025/slam:demo`），且容器内已构建好 `dobot_quad_navinterface` 工作空间。
- 容器内可以访问 `/mnt/oss/data-collection/`（已与主机共享）。
- 项目虚拟环境已安装 `mcap`、`mcap-ros2-support`：
  ```bash
  cd /root/embodvla
  .venv/bin/pip install mcap mcap-ros2-support
  ```

## 快速使用

### 1. 运行 SLAM 获取轨迹

```bash
cd /root/embodvla

# 1. 把 device_static_params（含 base_link->lidar 外参）复制到容器
docker cp third_party/device_static_params dobot_build:/tmp/device_static_params

# 2. 将脚本复制到容器并执行
docker cp tools/slam_trajectory_verify/run_slam_in_docker.sh dobot_build:/tmp/run_slam_in_docker.sh
docker exec dobot_build chmod +x /tmp/run_slam_in_docker.sh

# 参数：<bag_dir> <odom_bag_dir_in_container> <map_dir_in_container> <duration_seconds> [lidar_extrinsic_yaml]
docker exec dobot_build bash /tmp/run_slam_in_docker.sh \
  /mnt/oss/data-collection/20260630/N-01-035-004-1782801171966089189/dobot_bag_0000 \
  /tmp/slam_run/odom_bag \
  /tmp/slam_run/map \
  60

# 复制结果到主机
docker cp dobot_build:/tmp/slam_run/odom_bag.json outputs/slam_trajectory_verify/slam_output/
docker cp dobot_build:/tmp/slam_run/map outputs/slam_trajectory_verify/slam_output/
```

运行结束后会得到：

- `odom_bag.json`：`base_link` 在 `odom` 下的位姿序列（x, y, z, yaw, roll, pitch, quaternion, velocity, covariance）。
- `map/`：拼接后的点云地图。

### 坐标系与 base_link→lidar 外参说明

整个链路中的坐标变换如下：

1. **LIO 内部估计**: `dobot_quad_mapping` 的 LIO 估计出 `laser_link` 在 `odom` 下的位姿 `T_odom_laser`。
2. **静态 TF**: `run_slam_online_tf.launch.py` 从 `device_static_params` 读取 `base_link → laser_link` 外参 YAML 并发布为静态 TF。当前使用文件：
   ```text
   third_party/device_static_params/devices/robot_dog_v1/lidar/extrinsic/laser_link_to_base_link.yaml
   ```
   其中：
   - 平移：`x = -0.06855 m`，`y = 0.0005 m`，`z = 0.14788 m`
   - 旋转：四元数 `[qw=0.7071, qx=0, qy=0, qz=0.7071]`，即绕 `z` 轴 **+90°**
   - 含义：lidar 位于 base_link **后方 6.86 cm、上方 14.79 cm、基本居中**；lidar 的 `x` 轴指向 base_link 的 **左侧**，`y` 轴指向 base_link 的 **后方**。
3. **SLAM 节点转换**: 当配置 `odom_child_frame_id: base_link` 时，SLAM 节点在 `slam.cc` 中自动查询上述 TF，并做转换：
   - `T_odom_base = T_odom_laser · T_base_laser^{-1}`
   - 也就是用 lidar 位姿左乘 `laser_link → base_link` 的逆变换，得到 **base_link 在 odom 下的位姿**。
4. **JSON 输出**: `bag_to_odom_json.py` 直接把 `/lightning/odometry` 的 pose 写入 `odom_bag.json`，不再做额外坐标变换。
5. **可视化转换**: `extract_and_visualize.py` 将 odom 下的 base_link 位姿进一步转到**本体坐标系**（以初始帧或当前帧为参考），其中 `x` 向前、`y` 向左。

因此，**`odom_bag.json` 已经是 base_link 轨迹，不是 laser_link 轨迹**；base_link→lidar 的外参在 SLAM 节点内部已经被正确使用。只要 `device_static_params` 中的外参与车上 lidar 的实际安装一致，最终轨迹就是准确的 base_link 轨迹。

> 验证 TF 是否加载成功，可在容器内查看 SLAM 日志：
> ```bash
> docker exec dobot_build grep "static TF base_link->laser_link" /tmp/slam_run/logs/slam.log
> ```
> 正常应输出加载的平移/旋转值；若提示 `Failed to lookup ... using identity`，则转换失效，odometry 会退化为 laser_link 轨迹。
>
> 如需更换机型或外参，可覆盖 launch 参数：
> ```bash
> ros2 launch lightning run_slam_online_tf.launch.py lidar_extrinsic_yaml:=/path/to/laser_link_to_base_link.yaml
> ```

### 2. 抽取图像并逐帧可视化

```bash
cd /root/embodvla
.venv/bin/python tools/slam_trajectory_verify/extract_and_visualize.py \
  --bag-dir /mnt/oss/data-collection/20260630/N-01-035-004-1782801171966089189/dobot_bag_0000 \
  --traj-json outputs/slam_trajectory_verify/slam_output/odom_bag.json
```

输出：

- `outputs/slam_trajectory_verify/images/`：抽出的彩色相机原图。
- `outputs/slam_trajectory_verify/frames/`：每帧的叠加图，命名格式 `frame_{idx}_{timestamp_ns}.png`。
- `outputs/slam_trajectory_verify/trajectory_video.mp4`：由帧合成的 MP4 视频（见下文）。

可用 `--max-frames N` 先快速渲染少量帧验证效果。

### 3. 合成视频

```bash
.venv/bin/python tools/slam_trajectory_verify/frames_to_video.py \
  --frames-dir outputs/slam_trajectory_verify/frames \
  --output outputs/slam_trajectory_verify/trajectory_video.mp4 \
  --fps 20
```

默认 20 fps，可用 `--fps` 调整。输出为 H.264 MP4，可用普通播放器直接播放。

## 输出示例

单帧叠加图包含三栏：

- **左侧**：当前时刻彩色相机画面。
- **中间**：完整 SLAM 2D 轨迹在**初始 base_link 本体坐标系**下的投影（浅蓝线），固定起点（绿圆）、终点（红 X），以及随帧移动的当前位姿（亮绿圆点）。坐标定义：`x` 向前，`y` 向左（右手系）。
- **右侧**：以**当前帧为本体坐标系**的未来 3s 轨迹预览。`x` 向前（朝上），`y` 向左（朝左），与主视角一致；当前位置固定在原点（亮绿圆点），蓝色曲线为未来轨迹。坐标轴范围在所有帧中固定不变，便于观察车体运动趋势。

## 实测案例

### 案例 1：`20260630/N-01-035-004-1782801171966089189/dobot_bag_0000`

- 播放时长：60s
- SLAM 轨迹：913 个位姿，58.5s，本体坐标系下从 `(0, 0)` 移动到 `(29.79, -24.65)`；
- 同步图像：684 帧彩色相机画面；
- 输出：`outputs/slam_trajectory_verify/trajectory_video.mp4`。

### 案例 2：`20260630/N-01-035-004-1782802143984650483/dobot_bag_0000`

- 播放时长：60s
- SLAM 轨迹：420 个位姿，56.1s，本体坐标系下从 `(0, 0)` 移动到 `(38.92, -21.30)`；
- 同步图像：618 帧彩色相机画面；
- 输出：`outputs/slam_trajectory_verify_1782802143/trajectory_video.mp4`。

## 文件说明

| 文件 | 说明 |
|------|------|
| `run_slam_in_docker.sh` | 在 `dobot_build` 容器内运行 SLAM 并导出 JSON；启动前会清理旧的 SLAM/TF/录制进程，保存地图带 10s 超时，避免残留节点导致 `/lightning/odometry` 时间戳混乱 |
| `extract_and_visualize.py` | 抽图、时间戳匹配、逐帧可视化 |
| `frames_to_video.py` | 将渲染好的帧合成为 MP4 视频 |
| `README.md` | 本文档 |
