from isaacsim import SimulationApp

# 步骤1：初始化 SimulationApp
simulation_app = SimulationApp({"headless": False})

# 步骤2：导入核心API (必须在 SimulationApp 初始化之后)
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim

# 导入分离出的障碍物管理器
from obstacle_manager import DynamicObstacleManager

def main():
    # 3.1 创建仿真世界
    world = World(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01)

    # 3.2 基础场景
    world.scene.add_default_ground_plane()
    create_prim(
        prim_path="/World/defaultLight",
        prim_type="DistantLight",
        attributes={"inputs:intensity": 3000, "inputs:angle": 1.0}
    )

    # 3.3 初始化动态障碍物管理器
    obs_manager = DynamicObstacleManager(world, num_objects=30, area_size=15.0)

    # 3.4 重置世界
    world.reset()

    print("开始动态障碍物仿真...")
    print("障碍物已设置为 Kinematic 模式（无重力，手动控制移动）")
    
    # 3.5 仿真循环
    dt = 0.01
    while simulation_app.is_running():
        # 更新障碍物位置
        obs_manager.update(dt)
        
        # 执行仿真步
        world.step(render=True)

    simulation_app.close()

if __name__ == "__main__":
    main()
