import habitat
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from habitat.tasks.nav.nav import NavigationEpisode
from habitat_sim.gfx import LightInfo, LightPositionModel
from tqdm import tqdm
import os
import os.path as osp
import imageio
import json
import math
import habitat_sim
from habitat.utils.geometry_utils import quaternion_rotate_vector
import magnum as mn
import random



# -------------------------------------------------------------------------
# Oracle A* Agent 类
# 主要负责管理视频缓存，不再负责计算PID，因为导航交给Oracle了
# -------------------------------------------------------------------------
class OracleAStarAgent:
    def __init__(self, result_path, target_id=None):
        print("Initialize Oracle A* agent")
        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)
        self.target_id = target_id
        self.rgb_list = []
        self.mask_list = []
        self.reset()

    def reset(self, episode: NavigationEpisode = None, success: bool = False):
        # 保存视频逻辑与原版保持一致
        if len(self.rgb_list) != 0 and episode is not None:
            if success:
                scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split('.')[0]
                save_dir = os.path.join(self.result_path, scene_key)
                os.makedirs(save_dir, exist_ok=True)
                output_video_path = os.path.join(save_dir, "{}.mp4".format(episode.episode_id))
                # 假设 FPS 为 10，如果你的 Oracle 动作步长不同，需调整
                imageio.mimsave(output_video_path, self.rgb_list, fps=10)
                print(f"Successfully saved the episode video with episode id {episode.episode_id}")
        
        if len(self.mask_list) != 0 and episode is not None:
            if success:
                scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split('.')[0]
                save_dir = os.path.join(self.result_path, scene_key)
                os.makedirs(save_dir, exist_ok=True)
                output_video_path = os.path.join(save_dir, "{}_mask.mp4".format(episode.episode_id))
                # 假设 FPS 为 10，如果你的 Oracle 动作步长不同，需调整
                imageio.mimsave(output_video_path, self.mask_list, fps=10)
                print(f"Successfully saved the episode mask with episode id {episode.episode_id}")
        
        self.rgb_list = []
        self.mask_list = []

    def update_obs(self, observations, mask):
        # 仅用于记录图像，不再计算动作
        if "agent_1_articulated_agent_jaw_rgb" in observations:
            rgb = observations["agent_1_articulated_agent_jaw_rgb"]
            rgb_ = rgb[:, :, :3]
            rgb_uint8 = rgb_.astype(np.uint8)
            self.rgb_list.append(rgb_uint8)
        
        if "mask" in mask:
            img = mask["mask"]
            img_ = img[:, :, 0]
            mask_uint8 = (np.clip(img_, 0, 1) * 255).astype(np.uint8)
            self.mask_list.append(mask_uint8)

def evaluate_agent(
    config, dataset_split, save_path, target_id=None, max_episodes=None
) -> None:
    agent_wrapper = OracleAStarAgent(save_path, target_id)
    action_name = "agent_1_oracle_follow_action"
    with habitat.TrackEnv(config=config, dataset=dataset_split) as env:
        sim = env.sim
        agent_wrapper.reset()
        
        num_candidates = len(env.episodes)
        target_episodes = min(max_episodes or num_candidates, num_candidates)
        completed_episodes = 0
        progress = tqdm(total=target_episodes)

        for _ in range(num_candidates):
            if completed_episodes >= target_episodes:
                break
            try:
                obs = env.reset()
            except RuntimeError as error:
                if "No collision-free movable robot start found" not in str(error):
                    raise
                failed_episode = env.current_episode
                failure = {
                    "scene_id": failed_episode.scene_id,
                    "episode_id": failed_episode.episode_id,
                    "reason": str(error),
                }
                failure_path = os.path.join(
                    save_path, f"skipped_robot_starts_{os.getpid()}.jsonl"
                )
                os.makedirs(save_path, exist_ok=True)
                with open(failure_path, "a") as failure_file:
                    failure_file.write(json.dumps(failure) + "\n")
                print(
                    f"Skipping episode {failed_episode.episode_id}: {error}"
                )
                agent_wrapper.reset()
                continue
            current_ep_id = env.current_episode.episode_id

            scene_key = osp.splitext(
                osp.basename(env.current_episode.scene_id)
            )[0].split('.')[0]
            save_dir = os.path.join(save_path, scene_key)
            result_path = os.path.join(save_dir, f"{current_ep_id}.json")
            info_path = os.path.join(save_dir, f"{current_ep_id}_info.json")
            video_path = os.path.join(save_dir, f"{current_ep_id}.mp4")
            if all(os.path.exists(path) for path in (result_path, info_path, video_path)):
                print(f"Episode {current_ep_id} already collected; skipping.")
                agent_wrapper.reset()
                completed_episodes += 1
                progress.update(1)
                continue
            
            light_setup = [
                LightInfo(vector=[10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[-10.0, -2.0, 0.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[0.0, -2.0, 10.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
                LightInfo(vector=[0.0, -2.0, -10.0, 0.0], color=[1.0, 1.0, 1.0], model=LightPositionModel.Global),
            ]
            sim.set_light_setup(light_setup)

            result = {}
            record_infos = []
            
            try:
                instruction = env.current_episode.info.get('instruction', None)
            except Exception:
                instruction = None

            finished = False
            status = 'Normal'
            
            humanoid_agent_main = sim.agents_mgr[0].articulated_agent
            robot_agent = sim.agents_mgr[1].articulated_agent # 获取 Agent 1 (机器人) 句柄

            iter_step = 0 
            followed_step = 0
            too_far_count = 0
            pixel_count_zero = 0
            stop_count = 0
            
            DT = 1/10
            
            history_pos = 0

            episode_info = sim.ep_info.info
            num_goals_for_other_human = episode_info["num_goals_for_other_human"]

            # ==========================================
            action_list = [
                "agent_0_humanoid_navigate_action", 
                "agent_1_oracle_follow_action", # 默认动作
                "agent_2_oracle_nav_randcoord_action_obstacle", 
                "agent_3_oracle_nav_randcoord_action_obstacle", 
                "agent_4_oracle_nav_randcoord_action_obstacle", 
                "agent_5_oracle_nav_randcoord_action_obstacle",
            ]

            # 随机决定是否激活 agent_6, 7, 8
            if random.random() < 0.5:
                action_list.append("agent_6_oracle_nav_randcoord_action_obstacle")
                if random.random() < 0.4:
                    action_list.append("agent_7_oracle_nav_randcoord_action_obstacle")
                    if random.random() < 0.3:
                        action_list.append("agent_8_oracle_nav_randcoord_action_obstacle")
            
            # 将列表转换为元组，并在整个 Episode 中固定下来
            action_tuple = tuple(action_list)
            action_args = {}
            # ==========================================
            
            
            while not env.episode_over:
                record_info = {}
                
                # 更新视频帧
                obs = sim.get_sensor_observations()
                
                detector = env.task._get_observations(env.current_episode)
                polar_info = detector.get("agent_1_human_polar_sensor")
                mask = detector.get("agent_1_main_humanoid_detector_sensor")
                agent_wrapper.update_obs(obs,mask)
                

                
                # 获取当前距离和碰撞状态
                # 注意：info通常在step后更新，但在step前我们需要判断是否要倒车
                # 我们可以用上一帧的 metrics 或者直接算距离
                curr_dist = np.linalg.norm(robot_agent.base_pos - humanoid_agent_main.base_pos)

                action_args = {}
                

                action_dict = {
                    "action": action_tuple,
                    "action_args": action_args
                }
                
                iter_step += 1
                env.step(action_dict)
                # --- Step 环境 ---
                
                record_info["robot_pos"] = list(robot_agent.base_pos)
                record_info["humanoid_pos"] = list(humanoid_agent_main.base_pos)
                
                trans_matrix = robot_agent.base_transformation 
                rot_matrix = trans_matrix.rotation()
                rotation_quat = mn.Quaternion.from_matrix(rot_matrix)
                rot_list = [
                    rotation_quat.scalar,    # w
                    rotation_quat.vector.x,  # x
                    rotation_quat.vector.y,  # y
                    rotation_quat.vector.z   # z
                ]
                record_info["robot_rot"] = rot_list
                
                trans_matrix = humanoid_agent_main.base_transformation 
                rot_matrix = trans_matrix.rotation()
                rotation_quat = mn.Quaternion.from_matrix(rot_matrix)
                rot_list = [
                    rotation_quat.scalar,    # w
                    rotation_quat.vector.x,  # x
                    rotation_quat.vector.y,  # y
                    rotation_quat.vector.z   # z
                ]
                record_info["humanoid_rot"] = rot_list
                
                detector = env.task._get_observations(env.current_episode)
                polar_info = detector.get("agent_1_human_polar_sensor")
                mask = detector.get("agent_1_main_humanoid_detector_sensor")
                record_info["pixel_count"] = float(mask["pixel_count"][0])
                
                if record_info["pixel_count"] <= 1:
                    pixel_count_zero += 1
                    if pixel_count_zero > 16:
                        status = 'Pixel Lost'
                        finished = False
                        break
                else:
                    pixel_count_zero = 0
                
                if polar_info is not None:
                    distance = polar_info[0]
                    angle_rad = polar_info[1] # 弧度
                    angle_deg = np.degrees(angle_rad) # 角度
                    record_info["angle_rad"] = float(angle_rad)
                    record_info["angle_deg"] = float(angle_deg)
                    record_info["distance_2d"] = float(distance)
                    # print(f"Distance: {distance:.2f}m, Angle: {angle_deg:.2f} degrees")

                if action_name in env.task.actions:
                    action_instance = env.task.actions[action_name]
                    
                    oracle_velocity = action_instance.last_computed_velocity
                    
                    # print(f"Oracle GT Velocity: {oracle_velocity}")
                    record_info["base_velocity"] = oracle_velocity.tolist()
                else:
                    print(f"Warning: Action {action_name} not found!")

                # --- 记录 Metrics ---
                info = env.get_metrics()
                
                # 统计逻辑 (保持一致)
                if info['human_following'] == 1.0:
                    followed_step += 1
                    too_far_count = 0
                
                dist_to_human = float(info["distance_to_leader"])
                if dist_to_human > 4.0:
                    too_far_count += 1
                    if too_far_count > 25:
                        status = 'Lost'
                        finished = False
                        break
                
                if robot_agent.base_pos == history_pos:
                    stop_count += 1
                    if stop_count > 50:
                        status = 'Stop'
                        finished = False
                        break
                history_pos = robot_agent.base_pos
                
                record_info["step"] = iter_step
                record_info["dis_to_human"] = float(dist_to_human)
                record_info["facing"] = info['human_following']
                
                record_infos.append(record_info)

                if info['human_collision'] == 1.0:
                    status = 'Collision'
                    finished = False
                    break

            # --- 结束处理 ---
            print(f"Finished episode id: {env.current_episode.episode_id}, Status: {status}")
            info = env.get_metrics()

            if env.episode_over:
                finished = True

            result['finish'] = finished
            result['status'] = status
            if iter_step < 300:
                result['success'] = info['human_following_success'] and info['human_following']
            else:
                result['success'] = info['human_following']
            
            result['following_rate'] = followed_step / iter_step if iter_step > 0 else 0
            result['following_step'] = followed_step
            result['total_step'] = iter_step
            result['collision'] = info['human_collision']
            if instruction is not None:
                result['instruction'] = instruction

            if True:
                save_dir = os.path.join(save_path, scene_key)
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, "{}_info.json".format(env.current_episode.episode_id)), "w") as f:
                    json.dump(record_infos, f, indent=2)
                with open(os.path.join(save_dir, "{}.json".format(env.current_episode.episode_id)), "w") as f:
                    json.dump(result, f, indent=2)

            agent_wrapper.reset(env.current_episode, success=True)
            # agent_wrapper.reset(env.current_episode, success=result['success'])
            completed_episodes += 1
            progress.update(1)

        progress.close()
        if completed_episodes < target_episodes:
            raise RuntimeError(
                f"Collected only {completed_episodes}/{target_episodes} episodes "
                f"after exhausting {num_candidates} candidates"
            )
