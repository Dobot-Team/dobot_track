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
import gc
import torch.multiprocessing as mp  # 使用 torch 的多进程封装通常更稳定
from habitat.config import read_write
import itertools # 记得在文件最上方添加这个

# -------------------------------------------------------------------------
# 1. 路径与模块设置
# -------------------------------------------------------------------------
custom_code_path = "/data0/dsj/code/OpenTrackVLA/evt_bench"
config_path = "/data0/dsj/code/OpenTrackVLA/habitat-lab/habitat/config/benchmark/nav/track/track_train_stt.yaml"

if custom_code_path not in sys.path:
    sys.path.append(custom_code_path)

# 尝试导入自定义模块（放在 worker 内部导入也可以，但放在这里检查更安全）
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
# 2. 辅助函数
# -------------------------------------------------------------------------
def get_heading_angle(rotation_quat):
    """从四元数计算 Yaw (Heading) 角度，单位弧度"""
    heading_vector = quaternion_rotate_vector(rotation_quat, np.array([0, 0, -1]))
    phi = math.atan2(heading_vector[0], -heading_vector[2])
    return phi

# -------------------------------------------------------------------------
# 3. Worker 函数 (每个进程运行的逻辑)
# -------------------------------------------------------------------------
def run_dataset_worker(rank, gpu_list, world_size):
    target_gpu = gpu_list[rank]
    print(f"🚀 启动进程 {rank}/{world_size}, 绑定 GPU: {target_gpu}")

    if not os.path.exists(config_path):
        return

    config = get_config(config_path)
    with read_write(config):
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = target_gpu
        config.habitat.simulator.habitat_sim_v0.gpu_gpu = False

    # 1. 初始化环境
    try:
        env = Env(config=config)
    except Exception as e:
        print(f"❌ 进程 {rank} 初始化失败: {e}")
        return
    
    # -----------------------------------------------------------
    # [核心修复 1]: 获取属于当前 GPU 的任务列表
    # 我们把所有 episode 提取出来，存在 Python 列表里，完全脱离 Env 的自动控制
    # -----------------------------------------------------------
    all_episodes = env.episodes 
    
    # 计算当前进程需要跑的 indices (全局 index)
    my_indices = list(range(rank, len(all_episodes), world_size))
    # 取出对应的 episode 对象
    my_tasks = [all_episodes[i] for i in my_indices]

    print(f"Proc {rank} 分配任务数: {len(my_tasks)} (Indices: {my_indices[:5]}...)")
    
    # 创建输出目录
    data_output_dir = "./output_data"
    os.makedirs(data_output_dir, exist_ok=True)

    # 频率设置
    CTRL_FREQ = 50
    RECORD_FREQ = 10 
    INTERVAL = CTRL_FREQ // RECORD_FREQ 
    DT = 1.0 / RECORD_FREQ 
    ROBOT_AGENT_ID = 1

    # -----------------------------------------------------------
    # [核心修复 2]: 遍历分配给自己的任务列表
    # -----------------------------------------------------------
    for i, current_task_ep in enumerate(my_tasks):
        
        # 获取这个 episode 在全局数据集中原本的 index，用于文件命名
        ep_index = my_indices[i]

        # -------------------------------------------------------
        # [至关重要]: 强制指定环境只包含这一个 episode
        # 这样 env.reset() 必定加载这个特定的场景，绝不会错
        # -------------------------------------------------------
        env.episodes = [current_task_ep]
        # 重置迭代器，确保它指向这唯一的 episode
        env.episode_iterator = itertools.cycle(env.episodes) 
        
        # 现在 reset，得到的一定是 current_task_ep
        observations = env.reset()
        
        # --- 下面的逻辑基本不用变，除了 ep_index 已经是正确的全局ID了 ---
        current_ep = env.current_episode
        ep_id = current_ep.episode_id
        instruction = current_ep.info.get("instruction", "Follow the target.")
        
        print(f"[GPU {target_gpu} | Proc {rank}] Processing Ep_Index: {ep_index} (ID: {ep_id})")

        episode_step_info = [] 
        images = []
        
        prev_record_pos = np.array(env.sim.get_agent_state(ROBOT_AGENT_ID).position).copy()
        prev_record_rot = env.sim.get_agent_state(ROBOT_AGENT_ID).rotation
        prev_yaw = get_heading_angle(prev_record_rot)
        
        following_step_count = 0
        collision_flag = 0.0
        max_steps = config.habitat.environment.max_episode_steps

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
            
            if step_i % INTERVAL == 0 and step_i > 0:
                curr_state = env.sim.get_agent_state(ROBOT_AGENT_ID)
                curr_pos = np.array(curr_state.position).copy()
                curr_rot = curr_state.rotation
                
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
                if dis_to_human == -1.0: 
                     dis_to_human = np.linalg.norm(curr_pos - np.array(target_pos))
                
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

                # if "agent_1_articulated_agent_jaw_rgb" in observations:
                images.append(observations["agent_1_articulated_agent_jaw_rgb"])

            if env.episode_over:
                break
        
        ep_save_dir = os.path.join(data_output_dir, f"ep_{ep_index}")
        os.makedirs(ep_save_dir, exist_ok=True)
        
        with open(os.path.join(ep_save_dir, f"{ep_index}_info.json"), "w") as f:
            json.dump(episode_step_info, f, indent=2)
        
        recorded_steps = len(episode_step_info)
        summary_info = {
            "finish": True,
            "status": "Normal",
            "success": metrics.get("human_following_success"),
            "following_rate": following_step_count / recorded_steps if recorded_steps > 0 else 0.0,
            "following_step": following_step_count,
            "total_step": recorded_steps,
            "collision": collision_flag,
            "instruction": instruction
        }
        
        with open(os.path.join(ep_save_dir, f"{ep_index}.json"), "w") as f:
            json.dump(summary_info, f, indent=2)

        if len(images) > 0:
            images_to_video(images, output_dir=ep_save_dir, video_name=f"track_ep_{ep_index}", fps=RECORD_FREQ)
            del images
            images = []
            gc.collect()

        # --- 内存保护逻辑 ---
        # 即使这里重启 Env，也不会影响下一个任务，因为我们在循环开始时会重新把 my_tasks[i+1] 塞进去
        if i > 0 and i % 1 == 0:
            print(f"🔄 [Process {rank}] 定期重启环境...")
            env.close()
            del env
            gc.collect()
            env = Env(config=config)
            # 注意：新环境的 episodes 列表是完整的，但不要紧，
            # 下一次循环开始时，我们会执行 env.episodes = [next_task]

    env.close()
    print(f"✅ [Process {rank}] 任务完成")
# -------------------------------------------------------------------------
# 4. Main 入口
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # 配置要使用的 GPU 列表，例如 [0, 1, 2, 3]
    # 你可以根据实际机器情况修改这里，或者通过 argparse 传入
    AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7] 
    
    # 也可以自动检测
    # import torch
    # num_gpus = torch.cuda.device_count()
    # AVAILABLE_GPUS = list(range(num_gpus))
    
    WORLD_SIZE = len(AVAILABLE_GPUS)
    print(f"🔥 检测到 {WORLD_SIZE} 个 GPU，准备启动并行采集...")

    # [关键] 设置启动方法为 spawn
    # Habitat 和 PyTorch 在 fork 模式下混合使用 OpenGL/CUDA 极易死锁或崩溃
    mp.set_start_method("spawn", force=True)

    processes = []
    for rank in range(WORLD_SIZE):
        p = mp.Process(
            target=run_dataset_worker, 
            args=(rank, AVAILABLE_GPUS, WORLD_SIZE)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        
    print("🎉 所有进程已结束，数据采集完成。")