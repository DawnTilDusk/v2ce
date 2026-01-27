# 步骤1：初始化 SimulationApp (必须在所有其他 omni 导入之前)
from isaacsim import SimulationApp

# 配置无头模式 (headless=True) 或 GUI 模式 (headless=False)
simulation_app = SimulationApp({"headless": False})

# 步骤2：导入核心API (必须在 SimulationApp 初始化之后)
import numpy as np
import random
import omni.isaac.core
from omni.isaac.core import World
import pxr.UsdGeom
import pxr.UsdPhysics
import pxr.Usd
from isaacsim.core.utils.prims import create_prim
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder

# 步骤3：定义仿真主函数
def main():
    # ==================== 调试参数区 ====================
    AREA_SIZE = 10.0      # 正方形区域边长 (米)
    NUM_OBJECTS = 30      # 生成障碍物的总数量
    CUBOID_PROB = 0.6     # 生成长方体的概率 (剩下的是圆柱体)
    
    # 尺寸范围
    MIN_H, MAX_H = 0.5, 2.0   # 高度范围
    MIN_W, MAX_W = 0.2, 0.8   # 宽度/半径范围
    # ====================================================

    # 3.1 创建仿真世界 (在构造函数中设置 dt)
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=0.01,
        rendering_dt=0.01
    )

    # 3.2 场景基础搭建
    world.scene.add_default_ground_plane()
    create_prim(
        prim_path="/World/defaultLight",
        prim_type="DistantLight",
        attributes={"inputs:intensity": 1000, "inputs:angle": 1.0}
    )

    # 3.3 随机生成几何体
    print(f"正在生成 {NUM_OBJECTS} 个随机障碍物...")
    
    for i in range(NUM_OBJECTS):
        # 随机位置 (在正方形区域内)
        x = random.uniform(-AREA_SIZE/2, AREA_SIZE/2)
        y = random.uniform(-AREA_SIZE/2, AREA_SIZE/2)
        
        # 随机高度和粗细
        h = random.uniform(MIN_H, MAX_H)
        w = random.uniform(MIN_W, MAX_W)
        
        # 随机颜色
        color = np.array([random.random(), random.random(), random.random()])
        
        prim_path = f"/World/Obstacles/obj_{i:02d}"
        obj_name = f"obstacle_{i:02d}"

        if random.random() < CUBOID_PROB:
            # 生成长方体
            world.scene.add(
                DynamicCuboid(
                    prim_path=prim_path,
                    name=obj_name,
                    position=np.array([x, y, h/2]), # z设为高度的一半，使其刚好在地面上
                    scale=np.array([w, w, h]),      # 宽度x, 宽度y, 高度z
                    color=color
                )
            )
        else:
            # 生成圆柱体
            world.scene.add(
                DynamicCylinder(
                    prim_path=prim_path,
                    name=obj_name,
                    position=np.array([x, y, h/2]),
                    radius=w/2,                    # w作为直径，所以半径除以2
                    height=h,
                    color=color
                )
            )

    # 步骤4：仿真初始化 (Isaac Sim 4.5.0 不再需要 world.initialize() 和 set_dt 方法)
    # 在开始循环前重置世界
    world.reset()

    # 步骤5：仿真循环
    print("开始仿真...")
    for step in range(500):
        if simulation_app.is_running():
            # 执行单步仿真
            world.step(render=True)
            
            # 获取仿真数据
            if world.is_playing() and step % 100 == 0:
                print(f"仿真正在运行中... 当前步数: {step}")
        else:
            break

    print("仿真完成。请手动关闭窗口以退出...")
    while simulation_app.is_running():
        world.step(render=True)

# 步骤6：运行主函数并清理
if __name__ == "__main__":
    main()
    simulation_app.close()