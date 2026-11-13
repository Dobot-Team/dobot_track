import sys
import os
import numpy as np
import habitat
from habitat.config.default import get_config
from habitat.core.env import Env
from habitat.utils.visualizations.utils import images_to_video
import json
import math
import habitat_sim
from habitat.utils.geometry_utils import quaternion_rotate_vector
import argparse
import gc

# -------------------------------------------------------------------------
# 1. 路径与模块设置
# -------------------------------------------------------------------------
custom_code_path = "/data0/dsj/code/OpenTrackVLA/evt_bench"
if custom_code_path not in sys.path:
    sys.path.append(custom_code_path)

try:
    import additional_action 
    import additional_sensor
    import additional_metric
    import rearrange_sim_v2
    print("✅ 自定义模块导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    exit(1)

# -------------------------------------------------------------------------
# 2. 加载配置
# -------------------------------------------------------------------------
config_path = "/data0/dsj/code/OpenTrackVLA/habitat-lab/habitat/config/benchmark/nav/track/track_train_stt.yaml"
if not os.path.exists(config_path):
    print(f"❌ 找不到配置文件: {config_path}")
    exit(1)

config = get_config(config_path)

# -------------------------------------------------------------------------
# 3. 辅助函数
# -------------------------------------------------------------------------
def get_heading_angle(rotation_quat):
    heading_vector = quaternion_rotate_vector(rotation_quat, np.array([0, 0, -1]))
    phi = math.atan2(heading_vector[0], -heading_vector[2])
    return phi

# -------------------------------------------------------------------------
# 4. 核心运行逻辑 (整合版)
# -------------------------------------------------------------------------
def run_dataset(start_idx, end_idx):
    print(f"🚀 进程启动 | 范围: {start_idx} - {end_idx}")
    
    root_output_dir = "./stt"
    os.makedirs(root_output_dir, exist_ok=True)

    # --- 步骤 1: 预读取并排序任务列表 (保证多进程一致性) ---
    print("⏳ 初始化临时环境以获取任务列表...")
    temp_env = Env(config=config)
    
    # 强制排序：确保所有 GPU 拿到的任务顺序一模一样
    all_episodes = sorted(
        temp_env.episodes, 
        key=lambda x: (x.scene_id, int(x.episode_id) if x.episode_id.isdigit() else x.episode_id)
    )
    
    total_available_episodes = len(all_episodes)
    if end_idx == -1 or end_idx > total_available_episodes:
        end_idx = total_available_episodes
    
    if start_idx >= total_available_episodes:
        print("⚠️ 任务范围无效，进程退出。")
        temp_env.close()
        return

    # 切分属于本进程的任务
    my_episodes = all_episodes[start_idx:end_idx]
    print(f"✅ 任务分配完成: 本进程处理 {len(my_episodes)} 个任务")
    
    temp_env.close()
    del temp_env
    gc.collect()

    # --- 步骤 2: 循环处理 ---
    # 频率设置
    CTRL_FREQ = 50
    RECORD_FREQ = 10 
    INTERVAL = CTRL_FREQ // RECORD_FREQ 
    DT = 1.0 / RECORD_FREQ 
    ROBOT_AGENT_ID = 1
    
    env = None # 环境句柄

    for i, current_ep_data in enumerate(my_episodes):
        # 提取信息
        ep_id = current_ep_data.episode_id
        scene_id_full = current_ep_data.scene_id
        scene_name = os.path.basename(scene_id_full).split('.')[0]
        
        # 路径准备
        scene_save_dir = os.path.join(root_output_dir, scene_name)
        result_json_path = os.path.join(scene_save_dir, f"{ep_id}.json")
        
        # [跳过机制]
        if os.path.exists(result_json_path):
            print(f"[{i + 1}/{len(my_episodes)}] Scene: {scene_name} | ID: {ep_id} ⏩ 已完成，跳过。")
            continue

        try:
            # 环境懒加载 / 重建
            if env is None:
                env = Env(config=config)
            
            # [关键] 强制注入当前任务，防止乱序
            env.episodes = [current_ep_data]

            # Reset
            observations = env.reset()
            instruction = current_ep_data.info.get("instruction", "Follow the target.")
            print(f"[{i + 1}/{len(my_episodes)}] Scene: {scene_name} | ID: {ep_id} ▶️  运行中...")

            episode_step_info = [] 
            images = []
            
            # 初始化状态
            prev_record_pos = np.array(env.sim.get_agent_state(ROBOT_AGENT_ID).position).copy()
            prev_record_rot = env.sim.get_agent_state(ROBOT_AGENT_ID).rotation
            prev_yaw = get_heading_angle(prev_record_rot)
            
            following_step_count = 0
            collision_flag = 0.0
            max_steps = config.habitat.environment.max_episode_steps

            # Step 循环
            for step_i in range(max_steps):
                target_pos = env.sim.get_agent_state(0).position
                action_dict = {
                    "action": ("agent_0_humanoid_navigate_action", "agent_1_oracle_follow_action", "agent_2_oracle_nav_randcoord_action_obstacle",
                               "agent_3_oracle_nav_randcoord_action_obstacle", "agent_4_oracle_nav_randcoord_action_obstacle", 
                               "agent_6_oracle_nav_randcoord_action_obstacle", "agent_7_oracle_nav_randcoord_action_obstacle", 
                               "agent_5_oracle_nav_randcoord_action_obstacle", "agent_8_oracle_nav_randcoord_action_obstacle",
                               ),
                }
                observations = env.step(action_dict)
                metrics = env.get_metrics()

                # 10Hz 记录逻辑
                if step_i % INTERVAL == 0 and step_i > 0:
                    curr_state = env.sim.get_agent_state(ROBOT_AGENT_ID)
                    curr_pos = np.array(curr_state.position).copy()
                    curr_rot = curr_state.rotation
                    
                    # [保留你的正确速度计算逻辑]
                    disp_world = curr_pos - prev_record_pos
                    inv_rot = prev_record_rot.inverse()
                    disp_local = quaternion_rotate_vector(inv_rot, disp_world)
                    d_forward = -disp_local[2]
                    d_lateral = disp_local[0]
                    v_forward = d_forward / DT
                    v_lateral = d_lateral / DT
                    curr_yaw = get_heading_angle(curr_rot)
                    diff_yaw = curr_yaw - prev_yaw
                    if diff_yaw > math.pi: diff_yaw -= 2 * math.pi
                    elif diff_yaw < -math.pi: diff_yaw += 2 * math.pi
                    v_yaw = diff_yaw / DT
                    base_velocity_local = [float(v_forward), float(v_lateral), float(v_yaw)]
                    
                    prev_record_pos = curr_pos
                    prev_record_rot = curr_rot
                    prev_yaw = curr_yaw

                    dis_to_human = metrics.get("distance_to_leader", -1.0)
                    if dis_to_human == -1.0: dis_to_human = np.linalg.norm(curr_pos - np.array(target_pos))
                    
                    sensor_key = "agent_1_main_humanoid_detector_sensor"
                    facing_val = float(observations[sensor_key]["facing"][0]) if sensor_key in observations else 0.0
                    
                    if metrics.get("human_following", 0.0) > 0.5: following_step_count += 1
                    if metrics.get("human_collision", 0.0) > 0.5: collision_flag = 1.0

                    step_info = {
                        "step": step_i,
                        "dis_to_human": float(dis_to_human),
                        "facing": float(facing_val),
                        "base_velocity": base_velocity_local
                    }
                    episode_step_info.append(step_info)

                    # ---------------------------------------------------------
                    # [关键修复] 像你的脚本一样，先判断 Key 是否存在，避免崩溃
                    # ---------------------------------------------------------
                    if "agent_1_articulated_agent_jaw_rgb" in observations:
                        images.append(observations["agent_1_articulated_agent_jaw_rgb"])
                    # ---------------------------------------------------------

                if env.episode_over:
                    break
            
            # --- 保存数据 ---
            os.makedirs(scene_save_dir, exist_ok=True)
            
            with open(os.path.join(scene_save_dir, f"{ep_id}_info.json"), "w") as f:
                json.dump(episode_step_info, f, indent=2)
            
            recorded_steps = len(episode_step_info)
            summary_info = {
                "finish": True,
                "status": "Normal",
                "success": metrics.get("human_following_success", 0.0),
                "following_rate": following_step_count / recorded_steps if recorded_steps > 0 else 0.0,
                "following_step": following_step_count,
                "total_step": recorded_steps,
                "collision": collision_flag,
                "instruction": instruction
            }
            with open(result_json_path, "w") as f:
                json.dump(summary_info, f, indent=2)

            if len(images) > 0:
                images_to_video(images, output_dir=scene_save_dir, video_name=f"{ep_id}", fps=RECORD_FREQ)
                print(f"   -> 数据已保存: {scene_save_dir}/{ep_id}.mp4")
            
            del images
            images = []

            # --- 内存管理 ---
            # 每 5 个 episode 彻底重置环境，防止内存溢出
            if i > 0 and i % 5 == 0:
                env.close()
                env = None
                gc.collect()

        except Exception as e:
            print(f"❌ 错误 (ID: {ep_id}): {e}")
            import traceback
            traceback.print_exc()
            # 出错后销毁环境，下次循环重建
            if env:
                env.close()
                env = None
            gc.collect()

    if env:
        env.close()
    print("✅ 任务结束")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    args = parser.parse_args()

    run_dataset(args.start, args.end)