from isaacsim import SimulationApp

# 步骤1：启动 Isaac Sim (headless=False 会显示 UI 界面)
simulation_app = SimulationApp({"headless": False})

# 步骤2：导入核心API (必须在 SimulationApp 初始化之后)
import numpy as np
import omni.isaac.core.utils.nucleus as nucleus_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage

def main():
    # 3. 创建仿真世界
    print("[INFO] 正在初始化仿真世界...")
    world = World(stage_units_in_meters=1.0)
    
    # 4. 添加基础场景：地面和灯光
    world.scene.add_default_ground_plane()
    prim_utils.create_prim(
        prim_path="/World/defaultLight",
        prim_type="DistantLight",
        attributes={"inputs:intensity": 3000, "inputs:angle": 1.0}
    )

    # 5. 获取官方资产根路径 (需要连接到 Nucleus)
    assets_root_path = nucleus_utils.get_assets_root_path()
    if assets_root_path is None:
        print("[ERROR] 无法连接到 Nucleus 资产服务器。请确保 Isaac Sim 的 Nucleus 服务已启动。")
        simulation_app.close()
        return

    # 6. 定义 Unitree Go1 的官方 USD 路径
    # 这是 Isaac Sim 官方提供的标准机器人资产路径
    go1_url = f"{assets_root_path}/Isaac/Robots/Unitree/Go1/go1.usd"
    go1_prim_path = "/World/Unitree_Go1"

    print(f"[INFO] 正在从以下路径加载 Go1 资产: {go1_url}")

    # 7. 在场景中生成机器人
    add_reference_to_stage(usd_path=go1_url, prim_path=go1_prim_path)
    
    # 设置机器人的初始位置 (x, y, z)
    # 建议 z 设置为 0.45 左右，防止与地面穿模
    robot_prim = XFormPrim(prim_path=go1_prim_path)
    robot_prim.set_world_pose(position=np.array([0.0, 0.0, 0.45]))
    
    print(f"[SUCCESS] 已在 {go1_prim_path} 生成 Go1 机器人")

    # 8. 重置并开始仿真
    world.reset()
    print("[INFO] 仿真已启动。请在 GUI 窗口中查看机器狗。")
    print("[INFO] 按 Ctrl+C 或关闭窗口以退出。")
    
    while simulation_app.is_running():
        # 执行仿真步并渲染
        world.step(render=True)

    # 9. 清理并关闭
    simulation_app.close()

if __name__ == "__main__":
    main()
