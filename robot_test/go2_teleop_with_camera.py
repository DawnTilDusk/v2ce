from isaacsim import SimulationApp

# 步骤1：初始化 SimulationApp
simulation_app = SimulationApp({"headless": False})

# 步骤2：导入核心API
import numpy as np
import carb
import cv2
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

def main():
    # 3. 创建仿真世界
    world = World(stage_units_in_meters=1.0)
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
    
    # 如果路径不对，尝试在 stage 中查找相机
    my_camera = Camera(prim_path=camera_path, resolution=(640, 480))

    # 7. 重置世界 (关键：在初始化传感器后 reset)
    world.reset()
    my_camera.initialize()

    # 7.2 尝试将主视图切换到机器人相机 (作为 cv2.imshow 失败的备份)
    try:
        active_viewport = viewport_utils.get_active_viewport()
        if active_viewport:
            active_viewport.camera_path = camera_path
            print(f"[INFO] 已将 Isaac Sim 主视图切换至机器人相机: {camera_path}")
    except Exception as e:
        print(f"[WARN] 切换主视图失败: {e}")

    # 7.5 预热渲染器 (防止 get_rgba 返回空数据)
    print("[INFO] 正在预热渲染器...")
    for _ in range(10):
        world.step(render=True)

    # 6.1 获取关节映射与站立姿态 (同 teleop_direct)
    dof_names = my_go2.dof_names
    if dof_names is None:
        my_go2.initialize()
        dof_names = my_go2.dof_names
    
    dof_dict = {name: i for i, name in enumerate(dof_names)}
    n_dof = my_go2.num_dof
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

    # 8. 键盘控制设置
    _input = carb.input.acquire_input_interface()
    _keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    
    def is_key_pressed(key):
        return _input.get_keyboard_value(_keyboard, key) > 0

    move_speed = 0.5
    print("\n" + "="*50)
    print("控制与相机说明:")
    print("  ↑/↓/←/→ : 平移控制")
    print("  视频窗口 : 实时显示 front_cam RGB 流")
    print("  ESC      : 退出并关闭窗口")
    print("="*50 + "\n")

    # 9. 仿真循环
    dt = world.get_physics_dt()
    while simulation_app.is_running():
        # A. 获取图像流
        rgba_image = my_camera.get_rgba()
        if rgba_image is not None and rgba_image.size > 0:
            # 修复 IndexError: 如果返回的是 1D 数组，则手动 reshape
            if len(rgba_image.shape) == 1:
                w, h = 640, 480
                if rgba_image.size == h * w * 4:
                    rgba_image = rgba_image.reshape((h, w, 4))
                else:
                    continue # 数据大小不匹配，跳过此帧

            # 转换为 OpenCV 格式 (RGBA -> BGR)
            rgb_image = rgba_image[:, :, :3]
            
            # 如果是 float32 (0.0-1.0)，转换为 uint8 (0-255)
            if rgb_image.dtype == np.float32:
                rgb_image = (rgb_image * 255).astype(np.uint8)
                
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            # 在图像上添加文字提示
            cv2.putText(bgr_image, "Go2 Front Camera - Realtime Stream", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示窗口 (带错误捕获)
            try:
                cv2.imshow("Go2_Camera_View", bgr_image)
                if cv2.waitKey(1) & 0xFF == 27: # ESC 键
                    break
            except cv2.error:
                # 如果 cv2.imshow 报错，每隔 100 步提醒一次修复方法
                if world.current_time_step_index % 100 == 0:
                    print("\n[!] 提示: 您的 OpenCV 环境缺少 GUI 支持，无法弹出独立窗口。")
                    print("[!] 建议在 Isaac Sim 目录下运行: .\\python.bat -m pip install opencv-python --force-reinstall")
                    print("[!] 脚本已自动将 Isaac Sim 主视图切换为机器人视角，您可以直接在仿真主界面查看。")

        # B. 键盘控制逻辑
        current_pos, current_rot = my_go2.get_world_pose()
        moved = False
        if is_key_pressed(carb.input.KeyboardInput.UP):
            current_pos[0] += move_speed * dt
            moved = True
        if is_key_pressed(carb.input.KeyboardInput.DOWN):
            current_pos[0] -= move_speed * dt
            moved = True
        if is_key_pressed(carb.input.KeyboardInput.LEFT):
            current_pos[1] += move_speed * dt
            moved = True
        if is_key_pressed(carb.input.KeyboardInput.RIGHT):
            current_pos[1] -= move_speed * dt
            moved = True
            
        if moved:
            my_go2.set_world_pose(position=current_pos, orientation=current_rot)
            my_go2.set_linear_velocity(np.zeros(3))
            my_go2.set_angular_velocity(np.zeros(3))

        # C. 维持姿态与步进
        my_go2.apply_action(ArticulationAction(joint_positions=default_joints))
        world.step(render=True)

    # 10. 清理
    cv2.destroyAllWindows()
    simulation_app.close()

if __name__ == "__main__":
    main()
