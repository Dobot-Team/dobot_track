import sys
import os
import numpy as np
import habitat
from habitat.config.default import get_config
from habitat.core.env import Env
from habitat.utils.visualizations.utils import images_to_video
import json
import math
import habitat_sim # 必须导入，用于四元数计算
from habitat.utils.geometry_utils import quaternion_rotate_vector
import gc
# -------------------------------------------------------------------------
# 1. 路径与模块设置
# -------------------------------------------------------------------------
custom_code_path = "/data0/dsj/code/trackopen/evt_bench"
if custom_code_path not in sys.path:
    sys.path.append(custom_code_path)

try:
    import additional_action
    import additional_sensor
    import additional_metric
    import rearrange_sim_v2
    # import humanoid_controller
    print("✅ 自定义模块导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    exit(1)

# -------------------------------------------------------------------------
# 2. 加载配置
# -------------------------------------------------------------------------
config_path = "/data0/dsj/code/trackopen/habitat-lab/habitat/config/benchmark/nav/track/track_train_stt.yaml"
if not os.path.exists(config_path):
    print(f"❌ 找不到配置文件: {config_path}")
    exit(1)

config = get_config(config_path)

# -------------------------------------------------------------------------
# 3. 运行整个数据集
# -------------------------------------------------------------------------
def get_heading_angle(rotation_quat):
    """从四元数计算 Yaw (Heading) 角度，单位弧度"""
    # Habitat 中 -Z 是前方。我们要看 -Z 向量旋转后的朝向。
    # 或者直接用 quaternion 转换。这里用一个简单的向量投影法。
    heading_vector = quaternion_rotate_vector(rotation_quat, np.array([0, 0, -1]))
    # atan2(x, z) or atan2(-x, -z) depending on coord sys. 
    # Habitat: X is Right, Z is Back. Forward is -Z.
    phi = math.atan2(heading_vector[0], -heading_vector[2])
    return phi

def run_dataset():
    print("正在初始化 Habitat 环境...")
    
    data_output_dir = "./output_data"
    os.makedirs(data_output_dir, exist_ok=True)

    with habitat.TrackEnv(config=config) as env:
        total_episodes = len(env.episodes)
        
        # 频率设置
        CTRL_FREQ = 40
        RECORD_FREQ = 40
        INTERVAL = CTRL_FREQ // RECORD_FREQ 
        DT = 1.0 / RECORD_FREQ # 0.1秒
        
        ROBOT_AGENT_ID = 1

        for ep_index in range(total_episodes):
            observations = env.reset()
            if(ep_index%8!=1):
                continue
            current_ep = env.current_episode
            ep_id = current_ep.episode_id
            instruction = current_ep.info.get("instruction", "Follow the target.")
            
            print(f"\n[{ep_index + 1}/{total_episodes}] ID: {ep_id}")

            episode_step_info = [] 
            images = []
            
            # --- 初始化状态变量 ---
            # 1. 记录初始位置 (World Pos)
            prev_record_pos = np.array(env.sim.get_agent_state(ROBOT_AGENT_ID).position).copy()
            # 2. 记录初始旋转 (World Rotation Quaternion)
            prev_record_rot = env.sim.get_agent_state(ROBOT_AGENT_ID).rotation
            # 3. 记录初始 Yaw
            prev_yaw = get_heading_angle(prev_record_rot)
            
            following_step_count = 0
            collision_flag = 0.0
            max_steps = config.habitat.environment.max_episode_steps

            for step_i in range(max_steps):
                # --- A. Action (保持不变) ---
                target_pos = env.sim.get_agent_state(0).position
                
                action_dict = {
                    "action": ("agent_0_humanoid_navigate_action", "agent_1_oracle_follow_action", "agent_2_oracle_nav_randcoord_action_obstacle",
                               "agent_3_oracle_nav_randcoord_action_obstacle", "agent_4_oracle_nav_randcoord_action_obstacle", 
                               "agent_6_oracle_nav_randcoord_action_obstacle", "agent_7_oracle_nav_randcoord_action_obstacle", 
                               "agent_5_oracle_nav_randcoord_action_obstacle", "agent_8_oracle_nav_randcoord_action_obstacle",
                               ),
                }

                # --- B. Step ---
                observations = env.step(action_dict)
                
                # --- C. 记录逻辑 (10Hz) ---
                metrics = env.get_metrics()
                if step_i % INTERVAL == 0 and step_i > 0:
                    
                    # 获取当前状态
                    curr_state = env.sim.get_agent_state(ROBOT_AGENT_ID)
                    curr_pos = np.array(curr_state.position).copy()
                    curr_rot = curr_state.rotation
                    
                    # -------------------------------------------------------
                    # [核心修改] 计算 Local Velocity [Forward, Lateral, Yaw]
                    # -------------------------------------------------------
                    
                    # 1. 计算世界坐标位移向量
                    disp_world = curr_pos - prev_record_pos
                    
                    # 2. 计算局部坐标位移向量 (Local Displacement)
                    # 逻辑：将世界位移 乘以 机器人旋转的逆矩阵
                    # 这里的 prev_record_rot 是上一帧的朝向，我们用它来投影这一帧的移动
                    # (或者用 curr_rot 也可以，只要 dt 很小差距不大，通常用 prev 更符合控制逻辑)
                    inv_rot = prev_record_rot.inverse()
                    disp_local = quaternion_rotate_vector(inv_rot, disp_world)
                    
                    # Habitat 局部坐标系定义:
                    # X 轴: Right (右)
                    # Y 轴: Up (上)
                    # Z 轴: Back (后) -> 所以前方是 -Z
                    
                    # 3. 提取分量并计算线速度 (m/s)
                    d_forward = -disp_local[2] # 负Z 是向前
                    d_lateral = disp_local[0]  # X 是向右
                    
                    v_forward = d_forward / DT
                    v_lateral = d_lateral / DT
                    
                    # 4. 计算角速度 (Yaw Rate, rad/s)
                    curr_yaw = get_heading_angle(curr_rot)
                    
                    # 处理角度跳变 (例如从 3.14 变到 -3.14)
                    diff_yaw = curr_yaw - prev_yaw
                    if diff_yaw > math.pi: diff_yaw -= 2 * math.pi
                    elif diff_yaw < -math.pi: diff_yaw += 2 * math.pi
                    
                    v_yaw = diff_yaw / DT

                    # 5. 组合最终向量 [v_fwd, v_lat, v_yaw]
                    # 这就是 processing script 想要的数据格式！
                    base_velocity_local = [float(v_forward), float(v_lateral), float(v_yaw)]
                    
                    # -------------------------------------------------------

                    # 更新历史状态
                    prev_record_pos = curr_pos
                    prev_record_rot = curr_rot
                    prev_yaw = curr_yaw

                    # 获取其他数据
                    dis_to_human = metrics.get("distance_to_leader", -1.0)
                    if dis_to_human == -1.0: 
                         dis_to_human = np.linalg.norm(curr_pos - np.array(target_pos))
                    
                    sensor_key = "agent_1_main_humanoid_detector_sensor"
                    facing_val = float(observations[sensor_key]["facing"][0]) if sensor_key in observations else 0.0
                    
                    if metrics.get("human_following", 0.0) > 0.5: following_step_count += 1
                    if metrics.get("human_collision", 0.0) > 0.5: collision_flag = 1.0

                    # 存入列表
                    step_info = {
                        "step": step_i,
                        "dis_to_human": float(dis_to_human),
                        "facing": float(facing_val),
                        "base_velocity": base_velocity_local # 这里的格式现在对了！
                    }
                    episode_step_info.append(step_info)

                    if "agent_1_articulated_agent_jaw_rgb" in observations:
                        images.append(observations["agent_1_articulated_agent_jaw_rgb"])

                if env.episode_over:
                    break
            
            # --- 保存逻辑 (保持不变) ---
            ep_save_dir = os.path.join(data_output_dir, f"ep_{ep_index}")
            os.makedirs(ep_save_dir, exist_ok=True)
            
            with open(os.path.join(ep_save_dir, f"{ep_index}_info.json"), "w") as f:
                json.dump(episode_step_info, f, indent=2)
            
            # 注意：这里的 total_step 指的是“记录下来的帧数”，即 10Hz 下的步数
            recorded_steps = len(episode_step_info)
            summary_info = {
                "finish": True,
                "status": "Normal",
                "success": metrics.get("human_following_success"),
                "following_rate": following_step_count / recorded_steps if recorded_steps > 0 else 0.0,
                "following_step": following_step_count,
                "total_step": recorded_steps, # 这是 10Hz 的总步数
                "collision": collision_flag,
                "instruction": instruction
            }
            
            with open(os.path.join(ep_save_dir, f"{ep_index}.json"), "w") as f:
                json.dump(summary_info, f, indent=2)

            if len(images) > 0:
                # fps=10 对应 10Hz 的记录频率，这样视频播放速度才是正常的
                images_to_video(images, output_dir=ep_save_dir, video_name=f"track_ep_{ep_index}", fps=RECORD_FREQ)
                print(f"   -> 数据与视频已保存 (FPS={RECORD_FREQ})")
                print("   -> 清理内存...")
                del images      # 删除引用
                images = []     # 重置为空
                gc.collect()    # 强制 Python 进行垃圾回收  
                     
            # 每 20 个 episode 重启一次环境，防止 C++ 后端内存泄漏
            if ep_index > 0 and ep_index % 20 == 0:
                print("🔄 定期重启环境以释放资源...")
                env.close()
                del env
                gc.collect()
                env = Env(config=config)
                
if __name__ == "__main__":
    try:
        run_dataset()
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()