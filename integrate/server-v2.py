from isaacsim import SimulationApp
import os
import sys

# 步骤1：初始化 SimulationApp
simulation_app = SimulationApp({"headless": False})

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
from projectile_manager import ProjectileManager
from blocking_manager import BlockingObstacleManager
from trap_manager import TrapManager

def main():
    # 步骤2：创建仿真世界
    world = World(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01)
    
    # 获取资源路径
    assets_root_path = nucleus_utils.get_assets_root_path()
    if assets_root_path is None:
        print("[ERROR] 无法获取 Nucleus 资源路径，请确保 Nucleus 服务已运行。")
        return

    # --- 加载真实场景 (替代默认地面) ---
    # 选项1: 简易仓库 (空间较大，适合移动)
    scene_url = f"{assets_root_path}/Isaac/Environments/Simple_Warehouse/warehouse.usd"
    
    # 选项2: 露天/开放环境 (Grid)
    #scene_url = f"{assets_root_path}/Isaac/Environments/Grid/default_environment.usd"
    
    # 选项3: 办公室
    # scene_url = f"{assets_root_path}/Isaac/Environments/Office/office.usd"
    
    print(f"[INFO] 正在加载场景: {scene_url}")
    # 将场景加载到 /World/Environment
    add_reference_to_stage(usd_path=scene_url, prim_path="/World/Environment")
    
    # --- 调整场景大小 ---
    # 用户希望场景更大、天花板更高。我们通过 XFormPrim 对其进行整体缩放。
    env_prim = XFormPrim(prim_path="/World/Environment")
    # 缩放系数：1.0 表示保持原始比例 (仓库本身已经很大了)
    # 如果觉得还不够大，可以设为 1.5 或 2.0
    env_scale = 1.0 
    print(f"[INFO] 正在将场景放大 {env_scale} 倍...")
    env_prim.set_local_scale(np.array([env_scale, env_scale, env_scale]))
    
    # world.scene.add_default_ground_plane() # 注释掉默认地面，因为加载的场景通常已有地面
    
    # 添加光照 (如果场景自带光照可能需要调整，这里保留作为补光)
    prim_utils.create_prim(
        prim_path="/World/defaultLight",
        prim_type="DistantLight",
        attributes={"inputs:intensity": 3000, "inputs:angle": 1.0}
    )

    # 步骤3：初始化动态障碍物管理器 (已封装)
    # 生成 25 个障碍物，分布在 12x12 区域，避开中心 2.0m 范围
    obs_manager = DynamicObstacleManager(world, num_objects=25, area_size=12.0, excluded_center_radius=2.0)
    
    # 初始化抛射物管理器
    proj_manager = ProjectileManager(world, max_projectiles=10)
    
    # 初始化拦路障碍物管理器
    blocking_manager = BlockingObstacleManager(world, max_obstacles=5)
    
    # 初始化陷阱管理器
    trap_manager = TrapManager(world, max_traps=2)

    # 步骤4：加载 Unitree Go2 机器人
    # assets_root_path 已在前面获取
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
    move_speed = 8 # 稍微加快移动速度以适应动态环境
    dt = 0.01
    
    # 抛射物计时器
    projectile_timer = 0
    projectile_interval = 2.0 # 每 2 秒生成一个球
    
    # 拦路障碍物计时器
    blocking_timer = 0
    # 初始随机间隔
    blocking_interval = np.random.uniform(2.0, 4.0)
    
    # 围墙陷阱计时器
    trap_timer = 0
    trap_interval = np.random.uniform(5.0, 8.0) # 加快频率：每 5-8 秒生成一个

    # 步骤7：仿真循环
    while simulation_app.is_running():
        # A. 更新动态障碍物
        obs_manager.update(dt)
        
        # A2. 更新抛射物 (每隔一定时间在机器人前方生成)
        projectile_timer += dt
        if projectile_timer >= projectile_interval:
            projectile_timer = 0
            # 获取机器人当前位置
            current_pos, current_rot = my_go2.get_world_pose()
            # 在前方 3-5 米，高度 2-3 米生成，砸向机器人
            dist = np.random.uniform(3.0, 5.0)
            height = np.random.uniform(2.0, 3.0)
            proj_manager.spawn_projectile(current_pos, current_rot, forward_dist=dist, height=height)
            print(f"[INFO] 已生成抛射物! 距离: {dist:.2f}m")
            
        # A3. 更新拦路障碍物
        blocking_timer += dt
        if blocking_timer >= blocking_interval:
            blocking_timer = 0
            current_pos, current_rot = my_go2.get_world_pose()
            
            # FIX: 因为使用 set_world_pose 移动，物理引擎读到的速度为 0
            # 我们通过检测按键状态来手动构造"指令速度"
            cmd_vel = np.zeros(3)
            # 调试打印：检查按键输入
            # print(f"UP: {_input.get_keyboard_value(_keyboard, carb.input.KeyboardInput.UP)}")
            
            if _input.get_keyboard_value(_keyboard, carb.input.KeyboardInput.UP) > 0:
                cmd_vel[0] += move_speed
            if _input.get_keyboard_value(_keyboard, carb.input.KeyboardInput.DOWN) > 0:
                cmd_vel[0] -= move_speed
            if _input.get_keyboard_value(_keyboard, carb.input.KeyboardInput.LEFT) > 0:
                cmd_vel[1] += move_speed
            if _input.get_keyboard_value(_keyboard, carb.input.KeyboardInput.RIGHT) > 0:
                cmd_vel[1] -= move_speed
            
            # 只有当有移动指令时才生成
            if np.linalg.norm(cmd_vel[:2]) > 0.1:
                # 打印日志确认触发
                # print(f"[DEBUG] 触发路障生成! 速度指令: {cmd_vel}")
                blocking_manager.spawn_blocker(current_pos, cmd_vel)
                # 重置下一次的间隔为 2-4 秒之间的随机数
                blocking_interval = np.random.uniform(2.0, 4.0)
                # print(f"[INFO] 下一次路障将在 {blocking_interval:.2f} 秒后生成")
                
        # A4. 更新围墙陷阱 (每隔 10-15 秒)
        trap_timer += dt
        if trap_timer >= trap_interval:
            trap_timer = 0
            current_pos, current_rot = my_go2.get_world_pose()
            # 无论是否移动，强制生成围墙，把狗困住
            trap_manager.spawn_trap(current_pos, current_rot)
            # 重置下一次间隔
            trap_interval = np.random.uniform(5.0, 8.0)
            
        # A5. 更新陷阱状态 (检查是否需要移除)
        # 获取当前位置，确保变量已定义
        current_pos_check, _ = my_go2.get_world_pose()
        trap_manager.update(current_pos_check)

        # B. 处理视频客户端连接
        # D. 键盘控制机器人 (保持原有逻辑，确保移动本身是生效的)
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

        # D. 键盘控制机器人 (此段代码已移至上方检测逻辑中，避免重复检测)
        # current_pos, current_rot = my_go2.get_world_pose()
        # moved = False
        # ... (原代码被注释掉或删除)
        
        # 实际执行移动
        if moved:
            my_go2.set_world_pose(position=current_pos, orientation=current_rot)
            my_go2.set_linear_velocity(np.zeros(3))
            my_go2.set_angular_velocity(np.zeros(3))

        # E. 执行仿真
        my_go2.apply_action(ArticulationAction(joint_positions=default_joints))
        world.step(render=True)

    # 步骤8：清理退出
    proj_manager.cleanup() # 清理残留的球体
    blocking_manager.cleanup() # 清理路障
    trap_manager.cleanup() # 清理陷阱
    if conn: conn.close()
    server_socket.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
