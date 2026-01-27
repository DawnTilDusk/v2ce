from isaacsim import SimulationApp
import os
import sys

# 步骤1：初始化 SimulationApp
simulation_app = SimulationApp({"headless": True})

# 导入核心 API
import numpy as np
import carb
import cv2
import socket
import struct
import omni.appwindow
import omni.isaac.core.utils.nucleus as nucleus_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim
from omni.isaac.sensor import Camera

# 导入自定义的障碍物管理器
# 确保 integrate 目录在路径中
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
from obstacle_manager import DynamicObstacleManager

def main():
    # 步骤2：创建仿真世界
    world = World(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01)
    world.scene.add_default_ground_plane()
    
    # 添加光照
    prim_utils.create_prim(
        prim_path="/World/defaultLight",
        prim_type="DistantLight",
        attributes={"inputs:intensity": 3000, "inputs:angle": 1.0}
    )

    # 步骤3：初始化动态障碍物管理器 (已封装)
    # 生成 25 个障碍物，分布在 12x12 区域，避开中心 2.0m 范围
    obs_manager = DynamicObstacleManager(world, num_objects=25, area_size=12.0, excluded_center_radius=2.0)

    # 步骤4：加载 Unitree Go2 机器人
    assets_root_path = nucleus_utils.get_assets_root_path()
    go2_url = f"{assets_root_path}/Isaac/Robots/Unitree/Go2/go2.usd"
    go2_prim_path = "/World/Unitree_Go2"
    
    print(f"[INFO] 正在加载 Go2 机器人...")
    add_reference_to_stage(usd_path=go2_url, prim_path=go2_prim_path)
    
    robot_prim = XFormPrim(prim_path=go2_prim_path)
    robot_prim.set_world_pose(position=np.array([0.0, 0.0, 0.42]))

    my_go2 = Articulation(prim_path=go2_prim_path, name="go2_dog")
    world.scene.add(my_go2)

    # 步骤5：初始化相机
    camera_path = f"{go2_prim_path}/base/front_cam"
    my_camera = Camera(prim_path=camera_path, resolution=(640, 480))

    world.reset()
    my_camera.initialize()
    # 调整焦距为 10mm 以获得更宽的广角视野 (方案A)
    my_camera.set_focal_length(1.0)

    # 步骤6：设置 TCP 视频流服务器
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('127.0.0.1', 9999))
    server_socket.listen(1)
    server_socket.setblocking(False)
    
    print("\n" + "="*50)
    print("集成服务器 V2 (动态障碍物版) 已启动: 127.0.0.1:9999")
    print("等待客户端连接...")
    print("控制方式: 键盘方向键移动机器狗")
    print("="*50 + "\n")

    conn = None
    
    # 机器人控制增益与姿态初始化
    dof_names = my_go2.dof_names
    if dof_names is None:
        my_go2.initialize()
        dof_names = my_go2.dof_names
    dof_dict = {name: i for i, name in enumerate(dof_names)}
    
    stand_poses = {
        "FL_hip_joint": 0.1,  "RL_hip_joint": 0.1,
        "FR_hip_joint": -0.1, "RR_hip_joint": -0.1,
        "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0, 
        "FL_calf_joint": -1.5, "FR_calf_joint": -1.5,
        "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
    }
    default_joints = np.zeros(my_go2.num_dof)
    for name, val in stand_poses.items():
        if name in dof_dict: default_joints[dof_dict[name]] = val
        elif name.replace("_joint", "") in dof_dict: default_joints[dof_dict[name.replace("_joint", "")]] = val
    
    kps = np.full(my_go2.num_dof, 1000.0)
    kds = np.full(my_go2.num_dof, 30.0)
    my_go2.get_articulation_controller().set_gains(kps=kps, kds=kds)

    _input = carb.input.acquire_input_interface()
    _keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    move_speed = 0.8 # 稍微加快移动速度以适应动态环境
    dt = 0.01

    # 步骤7：仿真循环
    while simulation_app.is_running():
        # A. 更新动态障碍物
        obs_manager.update(dt)

        # B. 处理视频客户端连接
        if conn is None:
            try:
                conn, addr = server_socket.accept()
                conn.setblocking(True)
                print(f"[INFO] 客户端已连接: {addr}")
            except BlockingIOError:
                pass
        
        # C. 捕获并发送视频帧
        if conn:
            rgba_image = my_camera.get_rgba()
            if rgba_image is not None and rgba_image.size > 0:
                if len(rgba_image.shape) == 1:
                    rgba_image = rgba_image.reshape((480, 640, 4))
                rgb_image = rgba_image[:, :, :3]
                if rgb_image.dtype == np.float32:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                _, img_encoded = cv2.imencode('.jpg', bgr_image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                data = img_encoded.tobytes()
                
                try:
                    conn.sendall(struct.pack("L", len(data)) + data)
                except (ConnectionResetError, BrokenPipeError):
                    print("[WARN] 客户端连接已重置，等待新连接...")
                    conn.close()
                    conn = None

        # D. 键盘控制机器人
        current_pos, current_rot = my_go2.get_world_pose()
        moved = False
        if _input.get_keyboard_value(_keyboard, carb.input.KeyboardInput.UP) > 0:
            current_pos[0] += move_speed * dt
            moved = True
        if _input.get_keyboard_value(_keyboard, carb.input.KeyboardInput.DOWN) > 0:
            current_pos[0] -= move_speed * dt
            moved = True
        if _input.get_keyboard_value(_keyboard, carb.input.KeyboardInput.LEFT) > 0:
            current_pos[1] += move_speed * dt
            moved = True
        if _input.get_keyboard_value(_keyboard, carb.input.KeyboardInput.RIGHT) > 0:
            current_pos[1] -= move_speed * dt
            moved = True
            
        if moved:
            my_go2.set_world_pose(position=current_pos, orientation=current_rot)
            my_go2.set_linear_velocity(np.zeros(3))
            my_go2.set_angular_velocity(np.zeros(3))

        # E. 执行仿真
        my_go2.apply_action(ArticulationAction(joint_positions=default_joints))
        world.step(render=True)

    # 步骤8：清理退出
    if conn: conn.close()
    server_socket.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
