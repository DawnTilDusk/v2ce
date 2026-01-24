from isaacsim import SimulationApp

# 步骤1：初始化 SimulationApp
simulation_app = SimulationApp({"headless": False})

# 步骤2：导入核心API
import numpy as np
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
    
    # Go2 的官方路径通常在 Robots/Unitree/Go2 下
    go2_url = f"{assets_root_path}/Isaac/Robots/Unitree/Go2/go2.usd"
    go2_prim_path = "/World/Unitree_Go2"
    
    print(f"[INFO] 正在尝试从以下路径加载 Go2: {go2_url}")
    add_reference_to_stage(usd_path=go2_url, prim_path=go2_prim_path)
    
    # 设置初始位姿 (稍微抬高一点，0.42m 接近 Go1/Go2 的站立高度)
    robot_prim = XFormPrim(prim_path=go2_prim_path)
    robot_prim.set_world_pose(position=np.array([0.0, 0.0, 0.42]))

    # 5. 初始化 Articulation
    my_go2 = Articulation(prim_path=go2_prim_path, name="go2_dog")
    world.scene.add(my_go2)

    # 6. 重置世界 (关键：必须先 reset，Articulation 才会填充 dof_names)
    world.reset()

    # 6.1 定义站立姿态并动态映射索引 (参考 Zhefan-Xu 仓库逻辑)
    # 此时 dof_names 已经由 world.reset() 填充
    dof_names = my_go2.dof_names
    if dof_names is None:
        print("[WARN] dof_names 仍为 None，尝试手动初始化...")
        my_go2.initialize()
        dof_names = my_go2.dof_names

    print(f"[INFO] 机器人关节名称: {dof_names}")
    dof_dict = {name: i for i, name in enumerate(dof_names)}
    
    # 标准站立姿态参数 (单位: 弧度)
    # 参考官方配置: 后腿 Thigh 为 1.0, 其他为 0.8
    stand_poses = {
        "FL_hip_joint": 0.1,  "RL_hip_joint": 0.1,
        "FR_hip_joint": -0.1, "RR_hip_joint": -0.1,
        "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0, 
        "FL_calf_joint": -1.5, "FR_calf_joint": -1.5,
        "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
    }
    
    # 按照实际的 dof_names 顺序填充数组
    n_dof = my_go2.num_dof
    default_joints = np.zeros(n_dof)
    for name, val in stand_poses.items():
        if name in dof_dict:
            default_joints[dof_dict[name]] = val
        else:
            short_name = name.replace("_joint", "")
            if short_name in dof_dict:
                default_joints[dof_dict[short_name]] = val
    
    # 重置后立即强制设置一次初始关节角度和位姿，防止其倒下
    my_go2.set_joint_positions(default_joints)
    
    # 6.5 设置关节增益
    # 暴力提升刚度 (kps) 和 阻尼 (kds)
    kps = np.full(n_dof, 1000.0)
    kds = np.full(n_dof, 30.0)
    
    # 特别强化膝盖 (calf) 支撑力
    for name in dof_names:
        if "calf" in name:
            idx = dof_dict[name]
            kps[idx] = 3000.0
            kds[idx] = 100.0
    
    my_go2.get_articulation_controller().set_gains(kps=kps, kds=kds)

    print(f"[INFO] 已通过名称映射应用增益与姿态")

    print("Go2 机器人已准备就绪，正在保持站立姿态...")
    
    step_count = 0
    while simulation_app.is_running():
        # 持续应用站立姿态指令
        my_go2.apply_action(ArticulationAction(joint_positions=default_joints))
        
        # 物理步进
        world.step(render=True)
        
        if step_count % 100 == 0:
            pos, _ = my_go2.get_world_pose()
            print(f"Step {step_count:04d} | Go2 高度: {pos[2]:.3f}m | 状态: 稳定站立")
            
        step_count += 1

    simulation_app.close()

if __name__ == "__main__":
    main()
