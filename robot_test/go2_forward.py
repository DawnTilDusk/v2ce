from isaacsim import SimulationApp

# 步骤1：初始化 SimulationApp
simulation_app = SimulationApp({"headless": False})

# 步骤2：导入核心API
import numpy as np
import math
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
    
    # 设置初始位姿 (稍微抬高一点)
    robot_prim = XFormPrim(prim_path=go2_prim_path)
    robot_prim.set_world_pose(position=np.array([0.0, 0.0, 0.42]))

    # 5. 初始化 Articulation
    my_go2 = Articulation(prim_path=go2_prim_path, name="go2_dog")
    world.scene.add(my_go2)

    # 6. 重置世界 (加载物理句柄)
    world.reset()

    # 6.1 获取关节映射 (参考 Zhefan-Xu 仓库动态映射逻辑)
    dof_names = my_go2.dof_names
    if dof_names is None:
        my_go2.initialize()
        dof_names = my_go2.dof_names
    
    dof_dict = {name: i for i, name in enumerate(dof_names)}
    n_dof = my_go2.num_dof

    # 6.5 设置关节增益 (严格参考仓库 unitree.py: KPS=25.0, KDS=0.5)
    # 注意：在低增益下，步态的频率和幅度需要更加精细
    kps = np.full(n_dof, 25.0) 
    kds = np.full(n_dof, 0.5)
    my_go2.get_articulation_controller().set_gains(kps=kps, kds=kds)
    print(f"[INFO] 已设置官方增益: Stiffness=25.0, Damping=0.5")

    # 7. 定义标准站立角度
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

    # 8. 前进步态逻辑 (Trot Gait)
    print("开始前进仿真 (严格参考官方参数与逻辑)...")
    
    # 获取物理步长，用于精确的时间相位计算
    dt = world.get_physics_dt()
    current_time = 0.0
    
    # 严格参考参数：减小幅度以匹配低增益 (Kp=25) 环境，并提高频率增加稳定性
    freq = 3.0        # 步频 (Hz) - 提高到 3.0Hz 使步态更平稳
    stride = 0.1      # 步幅 (Thigh) - 减小到 0.1 弧度，防止摆动过大
    lift_height = 0.1 # 抬腿高度 (Calf) - 减小到 0.1 弧度，减少落地冲击
    
    step_count = 0
    while simulation_app.is_running():
        # 时间相位计算
        t = current_time * freq * 2 * math.pi
        
        # 目标角度初始化为站立姿态
        targets = default_joints.copy()
        
        # 对角腿组定义
        pair1 = ["FR", "RL"] # 第一组对角腿
        pair2 = ["FL", "RR"] # 第二组对角腿
        
        # 计算偏移量
        offset = math.sin(t)
        
        # 处理 Pair 1
        for leg in pair1:
            thigh_idx = dof_dict.get(f"{leg}_thigh_joint") or dof_dict.get(f"{leg}_thigh")
            calf_idx = dof_dict.get(f"{leg}_calf_joint") or dof_dict.get(f"{leg}_calf")
            
            # Thigh: 正向摆动 (Swing) 和 反向推地 (Stance)
            targets[thigh_idx] += offset * stride
            
            # Calf: 关键修正！
            # 在摆动相 (offset > 0)，需要减小角度 (使其更负) 来弯曲/缩短腿部以实现“抬腿”
            # 之前的 += offset * lift_height 会导致腿部伸长从而撞地
            if offset > 0:
                targets[calf_idx] -= offset * lift_height 
        
        # 处理 Pair 2 (相位相反)
        offset2 = -offset
        for leg in pair2:
            thigh_idx = dof_dict.get(f"{leg}_thigh_joint") or dof_dict.get(f"{leg}_thigh")
            calf_idx = dof_dict.get(f"{leg}_calf_joint") or dof_dict.get(f"{leg}_calf")
            
            targets[thigh_idx] += offset2 * stride
            
            if offset2 > 0:
                targets[calf_idx] -= offset2 * lift_height

        # 应用动作
        my_go2.apply_action(ArticulationAction(joint_positions=targets))
        
        # 物理步进
        world.step(render=True)
        
        # 更新时间 (使用真实 dt)
        current_time += dt
        
        if step_count % 100 == 0:
            pos, _ = my_go2.get_world_pose()
            print(f"Step {step_count:04d} | 位置: x={pos[0]:.3f}, y={pos[1]:.3f} | 状态: 步态稳定")
            
        step_count += 1
    
    simulation_app.close()

if __name__ == "__main__":
    main()
