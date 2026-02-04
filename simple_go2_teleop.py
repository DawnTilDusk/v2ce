from isaacsim import SimulationApp
import argparse

# 创建参数解析器
parser = argparse.ArgumentParser(description="Go2 Teleop")
# 添加 headless 参数
parser.add_argument("--headless", action="store_true", default=False, help="Force headless mode")
args_cli = parser.parse_args()

# 强制设置 headless 为 False 以显示界面
args_cli.headless = False

# 启动仿真 (使用 SimulationApp 以确保加载所有核心扩展)
simulation_app = SimulationApp({"headless": args_cli.headless})

# 步骤2：导入核心API
import numpy as np
import carb
import sys
import os
import torch
import time

# --------------------------------------------------------------------------------
# [CRITICAL] 强制注入 Conda 环境的 site-packages 路径
# 解决 Isaac Sim 找不到外部安装库 (如 einops, isaaclab) 的问题
# --------------------------------------------------------------------------------
# 更新为 v2ce 环境的路径
CONDA_SITE_PACKAGES = "/home/fishyu/anaconda3/envs/v2ce/lib/python3.10/site-packages"
if CONDA_SITE_PACKAGES not in sys.path:
    print(f"[INFO] Injecting Conda site-packages: {CONDA_SITE_PACKAGES}")
    sys.path.append(CONDA_SITE_PACKAGES)

# 确保能找到 isaac-go2-ros2 项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import omni.appwindow
import carb

# --------------------------------------------------------------------------------
# [FIX] 手动设置 Nucleus 服务器路径
# 解决 NUCLEUS_ASSET_ROOT_DIR 为 None 导致的 USD 加载失败问题
# --------------------------------------------------------------------------------
settings = carb.settings.get_settings()
if settings.get("/persistent/isaac/asset_root/cloud") is None:
    # 使用 NVIDIA 官方提供的 S3 镜像地址作为后备
    fallback_url = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5"
    print(f"[INFO] Setting Nucleus asset root to fallback URL: {fallback_url}")
    settings.set("/persistent/isaac/asset_root/cloud", fallback_url)

from omni.isaac.core import World

# 导入 RL 相关模块
from go2.go2_env import Go2RSLEnvCfg
import go2.go2_ctrl as go2_ctrl

def main():
    # 3. 配置 RL 环境参数
    # 我们使用 Go2RSLEnvCfg 配置，这会告诉 go2_ctrl 创建什么样的环境
    # (包含机器人、地形、传感器等)
    env_cfg = Go2RSLEnvCfg()
    env_cfg.scene.num_envs = 1 # 强制单机器人
    env_cfg.decimation = 4     # 控制频率分频 (sim_dt * decimation = control_dt)
    
    # 4. 加载 RL 策略与环境
    # 关键点：go2_ctrl.get_rsl_flat_policy 内部调用了 gym.make
    # 由于 SimulationApp 已经启动，Isaac Lab (基于 Omniverse) 会检测到现有的 App 并复用
    print(f"[INFO] Initializing RL Environment and Policy...")
    
    # 初始化全局速度命令张量 (用于键盘控制)
    go2_ctrl.init_base_vel_cmd(env_cfg.scene.num_envs)
    
    # 获取环境和策略
    # env 是 RslRlVecEnvWrapper 包装的 Isaac Lab 环境
    # policy 是加载了权重的 ActorCritic 网络
    try:
        env, policy = go2_ctrl.get_rsl_flat_policy(env_cfg)
    except Exception as e:
        print(f"[ERROR] Failed to load RL policy: {e}")
        print("[HINT] Make sure 'isaaclab' and 'rsl_rl' are installed in your conda environment.")
        simulation_app.close()
        return

    # 5. 设置键盘控制
    # 订阅键盘事件，更新 go2_ctrl 内部的 base_vel_cmd_input 张量
    _input = carb.input.acquire_input_interface()
    app_window = omni.appwindow.get_default_app_window()
    if app_window:
        _keyboard = app_window.get_keyboard()
        _input.subscribe_to_keyboard_events(_keyboard, go2_ctrl.sub_keyboard_event)
        print("[INFO] Keyboard control enabled.")
    else:
        print("[WARN] No App Window found, keyboard control disabled.")

    print("-" * 80)
    print("Go2 RL Teleop Started")
    print("Controls:")
    print("  W/S: Forward/Backward Velocity")
    print("  A/D: Left/Right Velocity")
    print("  Z/C: Rotate Left/Right")
    print("-" * 80)

    # 6. 仿真循环
    obs, _ = env.reset()
    
    while simulation_app.is_running():
        # 推理：根据当前观测计算动作
        with torch.inference_mode():
            actions = policy(obs)
            
            # 步进：执行动作，获取下一帧观测
            # env.step 会处理物理步进、渲染步进以及奖励计算(这里不需要)
            obs, _, _, _ = env.step(actions)
            
        # 注意：不需要手动调用 world.step(render=True)，因为 env.step 内部已经包含了
        
    # 7. 清理
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
