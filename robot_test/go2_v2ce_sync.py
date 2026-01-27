from isaacsim import SimulationApp

# 步骤1：初始化 SimulationApp (Headless = True 用于加速训练，但这里因为要同步，所以速度由推理决定)
# 为了调试方便，建议先设为 False，跑通后再改为 True
simulation_app = SimulationApp({"headless": False})

# 步骤2：导入核心API
import numpy as np
import carb
import cv2
import sys
import os

# --------------------------------------------------------------------------------
# [CRITICAL] 强制注入 Conda 环境的 site-packages 路径
# 解决 Isaac Sim 找不到外部安装库 (如 einops) 的问题
# --------------------------------------------------------------------------------
CONDA_SITE_PACKAGES = "/home/fishyu/anaconda3/lib/python3.10/site-packages"
if CONDA_SITE_PACKAGES not in sys.path:
    print(f"[INFO] Injecting Conda site-packages: {CONDA_SITE_PACKAGES}")
    sys.path.append(CONDA_SITE_PACKAGES)

import omni.appwindow
import omni.kit.viewport.utility as viewport_utils
import omni.isaac.core.utils.nucleus as nucleus_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim
from omni.isaac.sensor import Camera

# 导入 V2CE 推理模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from v2ce_inference import V2CEPredictor

def main():
    # 3. 创建仿真世界
    # 关键设置：强制物理步长为 1/30 秒，与模型训练时的 FPS 一致
    target_fps = 30.0
    physics_dt = 1.0 / target_fps
    
    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=physics_dt)
    world.scene.add_default_ground_plane()
    
    # 添加光照
    prim_utils.create_prim(
        prim_path="/World/defaultLight",
        prim_type="DistantLight",
        attributes={"inputs:intensity": 3000, "inputs:angle": 1.0}
    )

    # 4. 获取并生成 Go2 资产
    assets_root_path = nucleus_utils.get_assets_root_path()
    go2_url = f"{assets_root_path}/Isaac/Robots/Unitree/Go2/go2.usd"
    go2_prim_path = "/World/Unitree_Go2"
    
    print(f"[INFO] 正在加载 Go2 并准备相机...")
    add_reference_to_stage(usd_path=go2_url, prim_path=go2_prim_path)
    
    # 设置初始位姿
    robot_prim = XFormPrim(prim_path=go2_prim_path)
    initial_pos = np.array([0.0, 0.0, 0.42])
    robot_prim.set_world_pose(position=initial_pos)

    # 5. 初始化 Articulation
    my_go2 = Articulation(prim_path=go2_prim_path, name="go2_dog")
    world.scene.add(my_go2)

    # 6. 初始化相机传感器
    # 注意：Go2 的相机通常位于 base 或 trunk 下的 front_cam
    camera_path = f"{go2_prim_path}/base/front_cam"
    
    # 初始化相机，分辨率建议与模型输入比例一致或更高
    my_camera = Camera(prim_path=camera_path, resolution=(640, 480))

    # 7. 初始化 V2CE 推理器
    print("[INFO] 初始化 V2CE 推理器...")
    
    # [FIX] 使用动态绝对路径加载模型，避免 FileNotFoundError
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, 'weights', 'v2ce_3d.pt')
    
    # [DEBUG] 打印路径以确认
    print("="*50)
    print(f"[DEBUG] Calculated Model Path: {model_path}")
    print("="*50)
    
    # 注意：这里的 height/width 应该与模型训练参数一致
    v2ce_predictor = V2CEPredictor(
        model_path=model_path, 
        device='cuda:5',
        fps=int(target_fps),
        height=260,
        width=346
    )

    # 8. 重置世界
    world.reset()
    my_camera.initialize()

    # 预热渲染器
    print("[INFO] 正在预热渲染器...")
    for _ in range(10):
        world.step(render=True)

    # 获取关节控制相关 (同 teleop 逻辑)
    dof_names = my_go2.dof_names
    if dof_names is None:
        my_go2.initialize()
        dof_names = my_go2.dof_names
    
    dof_dict = {name: i for i, name in enumerate(dof_names)}
    n_dof = my_go2.num_dof
    # 定义站立姿态
    stand_poses = {
        "FL_hip_joint": 0.1,  "RL_hip_joint": 0.1,
        "FR_hip_joint": -0.1, "RR_hip_joint": -0.1,
        "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0, 
        "FL_calf_joint": -1.5, "FR_calf_joint": -1.5,
        "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
    }
    default_joints = np.zeros(n_dof)
    for name, val in stand_poses.items():
        if name in dof_dict:
            default_joints[dof_dict[name]] = val
        elif name.replace("_joint", "") in dof_dict:
            default_joints[dof_dict[name.replace("_joint", "")]] = val
            
    kps = np.full(n_dof, 1000.0)
    kds = np.full(n_dof, 30.0)
    my_go2.get_articulation_controller().set_gains(kps=kps, kds=kds)

    # 键盘控制设置 (增强鲁棒性，适配 Headless)
    _input = carb.input.acquire_input_interface()
    app_window = omni.appwindow.get_default_app_window()
    
    if app_window:
        _keyboard = app_window.get_keyboard()
    else:
        _keyboard = None
        print("[WARN] 无窗口模式运行，键盘控制已禁用")

    def is_key_pressed(key):
        if _keyboard is None:
            return False
        return _input.get_keyboard_value(_keyboard, key) > 0

    move_speed = 0.5
    print("\n" + "="*50)
    print(f"同步模式已启动 (FPS={target_fps})")
    print("  每一步仿真时间严格等于 1/30 秒")
    print("  等待 V2CE 推理完成后才进行物理步进")
    print("  ↑/↓/←/→ : 平移控制")
    print("  ESC      : 退出")
    print("="*50 + "\n")

    # 9. 主循环
    step_count = 0
    while simulation_app.is_running():
        # A. 获取图像
        rgba_image = my_camera.get_rgba()
        
        if rgba_image is not None and rgba_image.size > 0:
            # 预处理图像
            if len(rgba_image.shape) == 1:
                w, h = 640, 480
                if rgba_image.size == h * w * 4:
                    rgba_image = rgba_image.reshape((h, w, 4))
            
            rgb_image = rgba_image[:, :, :3]
            if rgb_image.dtype == np.float32:
                rgb_image = (rgb_image * 255).astype(np.uint8)
            
            # 转换为 BGR 用于 OpenCV 和 V2CE
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            # B. 执行 V2CE 推理 (同步阻塞)
            # 无论这里推理耗时多少 (10ms 或 500ms)，仿真世界的时间都在这一刻暂停
            event_frame = v2ce_predictor.predict(bgr_image)
            
            # C. 显示结果 (可选)
            # 注意：在 Headless 模式下 cv2.imshow 会失败，需要 try-catch 或根据配置决定
            try:
                if event_frame is not None:
                    cv2.imshow("V2CE Event Frame", event_frame)
                    # 这里可以在 event_frame 上做进一步的避障决策逻辑
                    # ...
                
                cv2.imshow("RGB Input", bgr_image)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            except Exception:
                pass # Headless 模式忽略显示错误

        # D. 机器人控制逻辑 (简单的键盘控制示例)
        current_pos, current_rot = my_go2.get_world_pose()
        moved = False
        # 这里使用了固定的 dt (1/30)，确保移动速度在物理世界中是恒定的
        if is_key_pressed(carb.input.KeyboardInput.UP):
            current_pos[0] += move_speed * physics_dt
            moved = True
        if is_key_pressed(carb.input.KeyboardInput.DOWN):
            current_pos[0] -= move_speed * physics_dt
            moved = True
        if is_key_pressed(carb.input.KeyboardInput.LEFT):
            current_pos[1] += move_speed * physics_dt
            moved = True
        if is_key_pressed(carb.input.KeyboardInput.RIGHT):
            current_pos[1] -= move_speed * physics_dt
            moved = True
            
        if moved:
            my_go2.set_world_pose(position=current_pos, orientation=current_rot)
            my_go2.set_linear_velocity(np.zeros(3))
            my_go2.set_angular_velocity(np.zeros(3))

        # 维持站立
        my_go2.apply_action(ArticulationAction(joint_positions=default_joints))
        
        # E. 物理世界步进
        # 这一步会让仿真世界的时间向前推进 1/30 秒
        world.step(render=True)
        step_count += 1

    # 清理
    cv2.destroyAllWindows()
    simulation_app.close()

if __name__ == "__main__":
    main()
