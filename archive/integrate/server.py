from isaacsim import SimulationApp

# 步骤1：初始化 SimulationApp (必须在所有其他 omni 导入之前)
simulation_app = SimulationApp({"headless": False})

import numpy as np
import random
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
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder

def main():
    # 步骤2：创建仿真世界
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    
    # 添加光照
    prim_utils.create_prim(
        prim_path="/World/defaultLight",
        prim_type="DistantLight",
        attributes={"inputs:intensity": 3000, "inputs:angle": 1.0}
    )

    # 步骤3：随机生成障碍物 (逻辑集成自 summon_obstacles.py)
    AREA_SIZE = 10.0      # 区域大小
    NUM_OBJECTS = 25      # 障碍物数量
    CUBOID_PROB = 0.6     # 长方体概率
    MIN_H, MAX_H = 0.5, 1.5
    MIN_W, MAX_W = 0.3, 0.6

    print(f"[INFO] 正在生成 {NUM_OBJECTS} 个随机障碍物...")
    for i in range(NUM_OBJECTS):
        x = random.uniform(-AREA_SIZE/2, AREA_SIZE/2)
        y = random.uniform(-AREA_SIZE/2, AREA_SIZE/2)
        
        # 避开中心区域 (机器人出生点)
        if abs(x) < 1.0 and abs(y) < 1.0:
            continue
            
        h = random.uniform(MIN_H, MAX_H)
        w = random.uniform(MIN_W, MAX_W)
        color = np.array([random.random(), random.random(), random.random()])
        
        prim_path = f"/World/Obstacles/obj_{i:02d}"
        obj_name = f"obstacle_{i:02d}"

        if random.random() < CUBOID_PROB:
            world.scene.add(
                DynamicCuboid(
                    prim_path=prim_path,
                    name=obj_name,
                    position=np.array([x, y, h/2]),
                    scale=np.array([w, w, h]),
                    color=color
                )
            )
        else:
            world.scene.add(
                DynamicCylinder(
                    prim_path=prim_path,
                    name=obj_name,
                    position=np.array([x, y, h/2]),
                    radius=w/2,
                    height=h,
                    color=color
                )
            )

    # 步骤4：加载 Unitree Go2 机器人 (逻辑集成自 go2_video_server.py)
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

    # 步骤6：设置 TCP 视频流服务器
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('127.0.0.1', 9999))
    server_socket.listen(1)
    server_socket.setblocking(False) # 非阻塞模式
    
    print("\n" + "="*50)
    print("集成视频服务器已启动: 127.0.0.1:9999")
    print("等待客户端连接...")
    print("使用键盘方向键控制机器人移动")
    print("="*50 + "\n")

    conn = None
    
    # 机器人控制参数初始化
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
        if name in dof_dict: 
            default_joints[dof_dict[name]] = val
        elif name.replace("_joint", "") in dof_dict: 
            default_joints[dof_dict[name.replace("_joint", "")]] = val
    
    kps = np.full(my_go2.num_dof, 1000.0)
    kds = np.full(my_go2.num_dof, 30.0)
    my_go2.get_articulation_controller().set_gains(kps=kps, kds=kds)

    _input = carb.input.acquire_input_interface()
    _keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    move_speed = 0.5
    dt = world.get_physics_dt()

    # 步骤7：仿真循环
    while simulation_app.is_running():
        # A. 检查连接
        if conn is None:
            try:
                conn, addr = server_socket.accept()
                conn.setblocking(True)
                print(f"[INFO] 客户端已连接: {addr}")
            except BlockingIOError:
                pass
        
        # B. 获取并发送图像
        if conn:
            rgba_image = my_camera.get_rgba()
            if rgba_image is not None and rgba_image.size > 0:
                if len(rgba_image.shape) == 1:
                    rgba_image = rgba_image.reshape((480, 640, 4))
                
                rgb_image = rgba_image[:, :, :3]
                if rgb_image.dtype == np.float32:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                _, img_encoded = cv2.imencode('.jpg', bgr_image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                data = img_encoded.tobytes()
                
                try:
                    conn.sendall(struct.pack("L", len(data)) + data)
                except (ConnectionResetError, BrokenPipeError):
                    print("[WARN] 客户端断开连接，重新等待中...")
                    conn.close()
                    conn = None

        # C. 键盘控制逻辑
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

        # D. 维持站立姿态并执行仿真步
        my_go2.apply_action(ArticulationAction(joint_positions=default_joints))
        world.step(render=True)

    # 步骤8：清理
    if conn: conn.close()
    server_socket.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
