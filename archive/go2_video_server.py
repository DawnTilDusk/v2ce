from isaacsim import SimulationApp

# 步骤1：初始化 SimulationApp
simulation_app = SimulationApp({"headless": False})

# 步骤2：导入核心API
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

def main():
    # 3. 创建仿真世界
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    
    prim_utils.create_prim(
        prim_path="/World/defaultLight",
        prim_type="DistantLight",
        attributes={"inputs:intensity": 3000, "inputs:angle": 1.0}
    )

    # 4. 加载 Go2
    assets_root_path = nucleus_utils.get_assets_root_path()
    go2_url = f"{assets_root_path}/Isaac/Robots/Unitree/Go2/go2.usd"
    go2_prim_path = "/World/Unitree_Go2"
    
    print(f"[INFO] 正在加载 Go2 并准备视频服务器...")
    add_reference_to_stage(usd_path=go2_url, prim_path=go2_prim_path)
    
    robot_prim = XFormPrim(prim_path=go2_prim_path)
    initial_pos = np.array([0.0, 0.0, 0.42])
    robot_prim.set_world_pose(position=initial_pos)

    my_go2 = Articulation(prim_path=go2_prim_path, name="go2_dog")
    world.scene.add(my_go2)

    # 5. 初始化相机
    camera_path = f"{go2_prim_path}/base/front_cam"
    my_camera = Camera(prim_path=camera_path, resolution=(640, 480))

    world.reset()
    my_camera.initialize()

    # 6. 设置 TCP 服务器
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('127.0.0.1', 9999))
    server_socket.listen(1)
    server_socket.setblocking(False) # 设置为非阻塞模式
    
    print("\n" + "="*50)
    print("视频服务器已启动: 127.0.0.1:9999")
    print("等待原生 Python 客户端连接...")
    print("="*50 + "\n")

    conn = None
    
    # 关节与控制逻辑复用
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
    move_speed = 0.5
    dt = world.get_physics_dt()

    # 仿真循环
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
                # 图像处理与压缩
                if len(rgba_image.shape) == 1:
                    rgba_image = rgba_image.reshape((480, 640, 4))
                
                rgb_image = rgba_image[:, :, :3]
                if rgb_image.dtype == np.float32:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                
                # 转换为 BGR 并压缩为 JPG
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                _, img_encoded = cv2.imencode('.jpg', bgr_image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                data = img_encoded.tobytes()
                
                try:
                    # 发送报头 (4字节长度) + 数据
                    conn.sendall(struct.pack("L", len(data)) + data)
                except (ConnectionResetError, BrokenPipeError):
                    print("[WARN] 客户端连接已断开，重新等待连接...")
                    conn.close()
                    conn = None

        # C. 键盘控制
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

        my_go2.apply_action(ArticulationAction(joint_positions=default_joints))
        world.step(render=True)

    if conn: conn.close()
    server_socket.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
