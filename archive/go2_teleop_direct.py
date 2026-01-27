from isaacsim import SimulationApp

# 步骤1：初始化 SimulationApp
simulation_app = SimulationApp({"headless": False})

# 步骤2：导入核心API
import numpy as np
import carb
import omni.appwindow
import omni.isaac.core.utils.nucleus as nucleus_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim

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
    
    print(f"[INFO] 正在加载 Go2: {go2_url}")
    add_reference_to_stage(usd_path=go2_url, prim_path=go2_prim_path)
    
    # 设置初始位姿
    robot_prim = XFormPrim(prim_path=go2_prim_path)
    initial_pos = np.array([0.0, 0.0, 0.42])
    robot_prim.set_world_pose(position=initial_pos)

    # 5. 初始化 Articulation
    my_go2 = Articulation(prim_path=go2_prim_path, name="go2_dog")
    world.scene.add(my_go2)

    # 6. 重置世界
    world.reset()

    # 6.1 获取关节映射
    dof_names = my_go2.dof_names
    if dof_names is None:
        my_go2.initialize()
        dof_names = my_go2.dof_names
    
    dof_dict = {name: i for i, name in enumerate(dof_names)}
    n_dof = my_go2.num_dof

    # 标准站立姿态参数 (单位: 弧度)
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
    
    # 设置关节增益 (高刚度以保持站立)
    kps = np.full(n_dof, 1000.0)
    kds = np.full(n_dof, 30.0)
    for name in dof_names:
        if "calf" in name:
            idx = dof_dict[name]
            kps[idx] = 3000.0
            kds[idx] = 100.0
    my_go2.get_articulation_controller().set_gains(kps=kps, kds=kds)

    # 7. 键盘控制设置
    _input = carb.input.acquire_input_interface()
    _keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    
    def is_key_pressed(key):
        return _input.get_keyboard_value(_keyboard, key) > 0

    move_speed = 0.5  # 平移速度 (m/s)
    print("\n" + "="*50)
    print("控制说明:")
    print("  ↑ / W : 前进")
    print("  ↓ / S : 后退")
    print("  ← / A : 向左")
    print("  → / D : 向右")
    print("  R     : 重置位置")
    print("  ESC   : 退出")
    print("="*50 + "\n")

    # 8. 仿真循环
    dt = world.get_physics_dt()
    while simulation_app.is_running():
        # 获取当前位姿
        current_pos, current_rot = my_go2.get_world_pose()
        
        # 处理输入
        moved = False
        if is_key_pressed(carb.input.KeyboardInput.UP) or is_key_pressed(carb.input.KeyboardInput.W):
            current_pos[0] += move_speed * dt
            moved = True
        if is_key_pressed(carb.input.KeyboardInput.DOWN) or is_key_pressed(carb.input.KeyboardInput.S):
            current_pos[0] -= move_speed * dt
            moved = True
        if is_key_pressed(carb.input.KeyboardInput.LEFT) or is_key_pressed(carb.input.KeyboardInput.A):
            current_pos[1] += move_speed * dt
            moved = True
        if is_key_pressed(carb.input.KeyboardInput.RIGHT) or is_key_pressed(carb.input.KeyboardInput.D):
            current_pos[1] -= move_speed * dt
            moved = True
            
        if is_key_pressed(carb.input.KeyboardInput.R):
            current_pos = initial_pos.copy()
            moved = True
            
        if is_key_pressed(carb.input.KeyboardInput.ESCAPE):
            break

        # 如果有移动，更新世界位姿
        if moved:
            my_go2.set_world_pose(position=current_pos, orientation=current_rot)
            # 重置速度，防止惯性抖动
            my_go2.set_linear_velocity(np.zeros(3))
            my_go2.set_angular_velocity(np.zeros(3))

        # 持续应用站立姿态
        my_go2.apply_action(ArticulationAction(joint_positions=default_joints))
        
        # 物理步进
        world.step(render=True)

    simulation_app.close()

if __name__ == "__main__":
    main()
