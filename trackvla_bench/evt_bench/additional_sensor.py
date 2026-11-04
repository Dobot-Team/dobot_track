#!/usr/bin/env python3

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union
import math
import numpy as np
from gym import spaces

from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    RGBSensor,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.tasks.nav.nav import PointGoalSensor
from hydra.core.config_store import ConfigStore
import habitat_sim

from dataclasses import dataclass
from habitat.config.default_structured_configs import LabSensorConfig
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector

if TYPE_CHECKING:
    from omegaconf import DictConfig

from habitat.tasks.rearrange.utils import UsesArticulatedAgentInterface


@dataclass
class MainHumanoidDetectorSensorConfig(LabSensorConfig):
    human_id: int = 0
    human_pixel_threshold: int = 60000
    return_image: bool = False
    is_return_image_bbox: bool = True

    type: str = "MainHumanoidDetectorSensor"

@dataclass
class OtherHumanoidDetectorSensorConfig(LabSensorConfig):
    human_id: int = 0
    human_pixel_threshold: int = 60000
    return_image: bool = False
    is_return_image_bbox: bool = True

    type: str = "OtherHumanoidDetectorSensor"


@registry.register_sensor
class MainHumanoidDetectorSensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._human_id = config.human_id
        self._human_pixel_threshold = config.human_pixel_threshold
        self._return_image = config.return_image
        self._is_return_image_bbox = config.is_return_image_bbox
        self._first_init = True

        # Check the observation size
        jaw_panoptic_shape = None
        for key in self._sim.sensor_suite.observation_spaces.spaces:
            if "articulated_agent_jaw_panoptic" in key:
                jaw_panoptic_shape = (
                    self._sim.sensor_suite.observation_spaces.spaces[key].shape
                )

        # Set the correct size
        if jaw_panoptic_shape is not None:
            self._height = jaw_panoptic_shape[0]
            self._width = jaw_panoptic_shape[1]
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return "main_humanoid_detector_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR
        
    def _get_observation_space(self, *args, **kwargs):
        return spaces.Dict({
            "facing": spaces.Box(
                shape=(1,),
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                dtype=np.float32
            ),
            "box": spaces.Box(
                shape=(4,),
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                dtype=np.float32
            ),
            "mask": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self._height, self._width, 1),
                dtype=np.float32
            ),
            "pixel_count": spaces.Box(
                shape=(1,),
                low=0,
                high=float(self._height * self._width), # 最大值为全图像素总数
                dtype=np.float32
            )
        })

    def _get_bbox(self, img):
        """Simple function to get the bounding box, assuming that only one object of interest in the image"""
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    def get_observation(self, observations, episode, *args, **kwargs):
        if self._first_init:
            self._human_id = episode.info["main_human_semantic_id"]
            self._first_init = False
        self._human_id = episode.info["main_human_semantic_id"]
        use_k = f"agent_{self.agent_id}_articulated_agent_jaw_panoptic"
        if use_k in observations:
            panoptic = observations[use_k]
        else:
            if self._return_image:
                return{
                    "facing": np.zeros((1,), dtype=np.float32),
                    "box": np.zeros((4,), dtype=np.float32),
                    "mask": np.zeros(
                        (self._height, self._width, 1), dtype=np.float32
                    ),
                    "pixel_count": np.zeros((1,), dtype=np.float32),
                } 
            else:
                return {
                    "facing": np.zeros((1,), dtype=np.float32),
                    "box": np.zeros((4,), dtype=np.float32),
                    "mask": np.zeros(
                        (self._height, self._width, 1), dtype=np.float32
                    ),
                    "pixel_count": np.zeros((1,), dtype=np.float32),
                }
        result = {
            "facing": np.zeros((1,), dtype=np.float32),
            "box": np.zeros((4,), dtype=np.float32),
            "mask": panoptic,
            "pixel_count": np.zeros((1,), dtype=np.float32),
        }

        tgt_mask = np.float32(panoptic == self._human_id)
        result["mask"] = np.expand_dims(tgt_mask, axis=-1)
        
        human_pixel_count = np.sum(tgt_mask)
        result["pixel_count"] = np.array([human_pixel_count], dtype=np.float32)
        
        if self._is_return_image_bbox and np.sum(tgt_mask) > 0:
            rmin, rmax, cmin, cmax = self._get_bbox(tgt_mask)
            result["box"] = np.array([cmin, rmin, cmax, rmax], dtype=np.float32)

        human_pixel_count = np.sum(panoptic == self._human_id)
        if (
            self._human_pixel_threshold < human_pixel_count < self._height * self._width * 0.3
        ):
            result["facing"] = np.ones((1,), dtype=np.float32)
        
        return result
     
@registry.register_sensor
class OtherHumanoidDetectorSensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._all_human_ids = list(range(1000, 1101))
        self._human_id = config.human_id
        self._human_pixel_threshold = config.human_pixel_threshold
        self._return_image = config.return_image
        self._is_return_image_bbox = config.is_return_image_bbox

        # Check the observation size
        jaw_panoptic_shape = None
        for key in self._sim.sensor_suite.observation_spaces.spaces:
            if "articulated_agent_jaw_panoptic" in key:
                jaw_panoptic_shape = (
                    self._sim.sensor_suite.observation_spaces.spaces[key].shape
                )

        # Set the correct size
        if jaw_panoptic_shape is not None:
            self._height = jaw_panoptic_shape[0]
            self._width = jaw_panoptic_shape[1]
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return "other_humanoid_detector_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR
        
    def _get_observation_space(self, *args, **kwargs):
        return spaces.Dict({
            "facing": spaces.Box(
                shape=(1,),
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                dtype=np.float32
            ),
            "block": spaces.Box(
                shape=(1,),
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                dtype=np.float32
            ),
            "box": spaces.Box(
                shape=(4,),
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                dtype=np.float32
            )
        })

    def _get_bbox(self, img):
        """Simple function to get the bounding box, assuming that only one object of interest in the image"""
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    def get_observation(self, observations, episode, *args, **kwargs):
        self._human_id = episode.info["main_human_semantic_id"]
        use_k = f"agent_{self.agent_id}_articulated_agent_jaw_panoptic"
        if use_k in observations:
            panoptic = observations[use_k]
        else:
            return {
                "facing": np.zeros((1,), dtype=np.float32),
                "block": np.zeros((1,), dtype=np.float32),
                "box": np.zeros((4,), dtype=np.float32)
            }
        result = {
            "facing": np.zeros((1,), dtype=np.float32),
            "block": np.zeros((1,), dtype=np.float32),
            "box": np.zeros((4,), dtype=np.float32)
        }

        for other_human_id in self._all_human_ids:
            if other_human_id == self._human_id:
                continue
            tgt_mask = np.float32(panoptic == other_human_id)
            
            if self._is_return_image_bbox and np.sum(tgt_mask) > 0:
                rmin, rmax, cmin, cmax = self._get_bbox(tgt_mask)
                result["box"] = np.array([rmin, cmin, rmax, cmax], dtype=np.float32)

            human_pixel_count = np.sum(panoptic == other_human_id)
            if (
                human_pixel_count > self._human_pixel_threshold
            ):
                result["facing"] = np.ones((1,), dtype=np.float32)
            if (
                human_pixel_count > self._height * self._width * 0.2
            ):
                result["block"] = np.ones((1,), dtype=np.float32)
        
        return result
    
@dataclass
class HumanPolarSensorConfig(LabSensorConfig):
    type: str = "HumanPolarSensor"
    # 既然只针对 Spot，这里默认值直接写死，YAML 里不写也没关系
    agent_id: int = 1         
    target_agent_idx: int = 0 
    camera_sensor_uuid: str = "agent_1_articulated_agent_jaw_rgb"

@registry.register_sensor
class HumanPolarSensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._target_agent_idx = config.target_agent_idx
        self._camera_sensor_uuid = config.camera_sensor_uuid
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return "human_polar_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        # 返回 [rho, phi]
        return spaces.Box(
            shape=(2,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        # 1. 获取目标 Human (Agent 0) 的位置
        # 使用 agents_mgr 获取 ArticulatedAgent 是安全的
        target_agent = self._sim.agents_mgr[self._target_agent_idx].articulated_agent
        target_world_pos = target_agent.base_pos

        # 2. 获取 Spot (Agent 1) 的底层对象
        # [关键修复]：绕过 self._sim.agents，直接访问 C++ 后端 (habitat_sim)
        # RearrangeSim 将底层 sim 放在 self._sim.habitat_sim 中
        backend_sim = self._sim
        
        # 直接从后端列表获取 Agent 1，这里一定存在
        spot_agent_backend = backend_sim.agents[0]

        # 3. 获取 Jaw 相机传感器
        # 直接尝试用配置的名字获取
        sensor_obj = spot_agent_backend._sensors.get(self._camera_sensor_uuid)
        
        if sensor_obj is None:
             raise ValueError(
                f"HumanPolarSensor Error: Cannot find jaw sensor on Agent 1.\n"
                f"Available backend sensors: {list(spot_agent_backend._sensors.keys())}"
            )

        # 4. 获取变换矩阵
        # 后端 sensor 对象直接有 .node 属性
        camera_trans = sensor_obj.node.absolute_transformation()
        inv_camera_trans = camera_trans.inverted()

        # 5. 坐标转换 (World -> Camera Local)
        rel_pos = inv_camera_trans.transform_point(target_world_pos)

        # 6. 计算极坐标
        # Habitat 相机坐标系: -Z=前, +X=右
        rho = np.hypot(rel_pos.x, rel_pos.z)
        phi = np.arctan2(rel_pos.x, -rel_pos.z)

        return np.array([rho, phi], dtype=np.float32)

@dataclass
class AgentVelocitySensorConfig(LabSensorConfig):
    type: str = "AgentVelocitySensor"

@registry.register_sensor
class AgentVelocitySensor(UsesArticulatedAgentInterface, Sensor):
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return "agent_velocity_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        # 返回 6维 向量: 
        # [Linear_X, Linear_Y, Linear_Z, Angular_X, Angular_Y, Angular_Z]
        return spaces.Box(
            shape=(6,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        agent = self._sim.agents_mgr[self.agent_id].articulated_agent
        
        # 1. 获取世界坐标系下的速度 (World Frame)
        # linear_velocity 和 angular_velocity 是 magnum.Vector3 类型
        if not hasattr(agent, "base_velocity"):
            return np.zeros(3, dtype=np.float32)
        
        lin_vel_world = agent.base_velocity
        ang_vel_world = agent.base_angular_velocity

        # 2. 获取机器人当前的旋转矩阵的逆矩阵
        # 用于将世界坐标向量旋转到机器人局部坐标系
        # inverted() 得到逆变换，rotation() 提取旋转部分
        inv_rotation = agent.base_transformation.inverted().rotation()

        # 3. 转换到局部坐标系 (Local Frame)
        # transform_vector 只旋转向量，不平移
        lin_vel_local = inv_rotation.transform_vector(lin_vel_world)
        ang_vel_local = inv_rotation.transform_vector(ang_vel_world)

        # 4. 组合结果
        # Habitat 中: -Z 是前方, X 是右方, Y 是上方
        # 典型的移动机器人:
        # lin_vel_local.z -> 前进/后退速度 (负值通常是前进，取决于坐标定义，建议print确认)
        # lin_vel_local.x -> 横向平移速度 (Slide)
        # ang_vel_local.y -> 偏航角速度 (Yaw rate)
        
        # velocity_tensor = np.array([
        #     lin_vel_local.x, lin_vel_local.y, lin_vel_local.z,
        #     ang_vel_local.x, ang_vel_local.y, ang_vel_local.z
        # ], dtype=np.float32)
        
        velocity_tensor = np.array([
            lin_vel_local.z, lin_vel_local.x, ang_vel_local.y,
        ], dtype=np.float32)

        return velocity_tensor

cs = ConfigStore.instance()

cs.store(
    package="habitat.task.lab_sensors.agent_velocity_sensor",
    group="habitat/task/lab_sensors",
    name="agent_velocity_sensor",
    node=AgentVelocitySensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.human_polar_sensor",
    group="habitat/task/lab_sensors",
    name="human_polar_sensor",
    node=HumanPolarSensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.main_humanoid_detector_sensor",
    group="habitat/task/lab_sensors",
    name="main_humanoid_detector_sensor",
    node=MainHumanoidDetectorSensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.other_humanoid_detector_sensor",
    group="habitat/task/lab_sensors",
    name="other_humanoid_detector_sensor",
    node=OtherHumanoidDetectorSensorConfig,
)