
from isaacsim import SimulationApp
import argparse
import sys
import os

# Create argument parser
parser = argparse.ArgumentParser(description="Go2 Teleop with V2CE")
parser.add_argument("--headless", action="store_true", default=False, help="Force headless mode")
parser.add_argument("--device", type=str, default="cuda:0", help="Device for V2CE inference (default: cuda:0)")
args_cli = parser.parse_args()

# Start Simulation (Force headless=False for visualization unless specified)
simulation_app = SimulationApp({"headless": args_cli.headless})

import numpy as np
import carb
import torch
import cv2
import matplotlib.pyplot as plt

# Inject Conda site-packages
CONDA_SITE_PACKAGES = "/home/fishyu/anaconda3/envs/v2ce/lib/python3.10/site-packages"
if CONDA_SITE_PACKAGES not in sys.path:
    print(f"[INFO] Injecting Conda site-packages: {CONDA_SITE_PACKAGES}")
    sys.path.append(CONDA_SITE_PACKAGES)

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import omni.appwindow
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core import World
from omni.isaac.sensor import Camera

# Import RL modules
from go2.go2_env import Go2RSLEnvCfg
import go2.go2_ctrl as go2_ctrl

# Import V2CE modules
from v2ce_inference import V2CEPredictor
from env.obstacle_manager_fabric import DynamicObstacleManager

def find_robot_prim_path(world):
    """
    Find the first valid robot prim path in the scene.
    Isaac Lab usually spawns robots at /World/envs/env_0/Robot
    """
    possible_paths = [
        "/World/envs/env_0/Robot",
        "/World/Unitree_Go2",
        "/World/Robot"
    ]
    
    stage = world.stage
    for path in possible_paths:
        if stage.GetPrimAtPath(path).IsValid():
            return path
            
    # Fallback: search for any prim with "base" in its name which is a common root link name
    # But better to just return the most likely one if not found and let it fail later
    return "/World/envs/env_0/Robot"

def main():
    # 1. Configure RL Environment
    env_cfg = Go2RSLEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.decimation = 4
    
    # 2. Load Policy and Env
    print(f"[INFO] Initializing RL Environment and Policy...")
    go2_ctrl.init_base_vel_cmd(env_cfg.scene.num_envs)
    
    try:
        env, policy = go2_ctrl.get_rsl_flat_policy(env_cfg)
    except Exception as e:
        print(f"[ERROR] Failed to load RL policy: {e}")
        simulation_app.close()
        return

    # 3. Get World and Setup V2CE
    world = World.instance()
    
    # Find robot and attach camera
    robot_prim_path = find_robot_prim_path(world)
    print(f"[INFO] Found robot at: {robot_prim_path}")
    
    # Camera setup (Front camera)
    camera_path = f"{robot_prim_path}/base/front_cam"
    # Ensure the prim exists or create it? 
    # Isaac Lab's Go2 USD should have "base/front_cam" XForm. If not, we might need to create it.
    # But usually standard Unitree USDs have it.
    
    print(f"[INFO] Attaching camera to: {camera_path}")
    my_camera = Camera(prim_path=camera_path, resolution=(640, 480))
    my_camera.initialize() # Initialize explicitly
    
    # Obstacle Manager
    print("[INFO] Initializing Dynamic Obstacle Manager...")
    obs_manager = DynamicObstacleManager(world, num_objects=20, area_size=10.0)
    
    # V2CE Predictor
    print("[INFO] Initializing V2CE Predictor...")
    model_path = os.path.join(current_dir, 'weights', 'v2ce_3d.pt')
    target_fps = 30.0 # Match simulation physics if possible
    
    v2ce_predictor = V2CEPredictor(
        model_path=model_path, 
        device=args_cli.device, 
        fps=int(target_fps),
        height=260,
        width=346
    )

    # 4. Setup Keyboard
    _input = carb.input.acquire_input_interface()
    app_window = omni.appwindow.get_default_app_window()
    if app_window:
        _keyboard = app_window.get_keyboard()
        _input.subscribe_to_keyboard_events(_keyboard, go2_ctrl.sub_keyboard_event)
        print("[INFO] Keyboard control enabled (W/A/S/D).")
    
    # 5. Visualization Setup (Matplotlib)
    if not args_cli.headless:
        print("[INFO] Initializing Matplotlib Visualization...")
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        empty_img_v2ce = np.zeros((260, 346, 3), dtype=np.uint8)
        empty_img_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        
        im1 = ax1.imshow(empty_img_v2ce)
        ax1.set_title("V2CE Event Frame")
        ax1.axis('off')
        
        im2 = ax2.imshow(empty_img_rgb)
        ax2.set_title("RGB Input")
        ax2.axis('off')
        
        plt.show()
    else:
        print("[INFO] Headless mode enabled. Visualization disabled.")

    # 6. Main Loop
    print("-" * 80)
    print("Go2 RL Teleop + V2CE Started")
    print("-" * 80)
    
    obs, _ = env.reset()
    step_count = 0
    
    while simulation_app.is_running():
        # A. Camera & V2CE
        rgba_image = my_camera.get_rgba()
        event_frame = None
        bgr_image = None
        
        if rgba_image is not None and rgba_image.size > 0:
            if len(rgba_image.shape) == 1:
                rgba_image = rgba_image.reshape((480, 640, 4))
            
            rgb_image = rgba_image[:, :, :3]
            if rgb_image.dtype == np.float32:
                rgb_image = (rgb_image * 255).astype(np.uint8)
                
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            # Run V2CE
            event_frame = v2ce_predictor.predict(bgr_image)
            
            # Update Visualization
            if not args_cli.headless:
                try:
                    if event_frame is not None:
                        im1.set_data(cv2.cvtColor(event_frame, cv2.COLOR_BGR2RGB))
                    
                    im2.set_data(rgb_image) # Already RGB
                    
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                except Exception:
                    pass

        # B. RL Control
        with torch.inference_mode():
            actions = policy(obs)
            # Step the environment (physics)
            obs, _, _, _ = env.step(actions)
            
        # C. Update Obstacles
        # env.step handles world.step, but we might need to manually update obstacle manager logic
        # if it depends on world time or needs explicit update call
        obs_manager.update(1.0/target_fps) # Assuming dt
        
        step_count += 1
        
    if not args_cli.headless:
        pass
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
