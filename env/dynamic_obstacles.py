from isaacsim import SimulationApp

# 步骤1：初始化 SimulationApp
simulation_app = SimulationApp({"headless": False})

# 步骤2：导入核心API (必须在 SimulationApp 初始化之后)
import numpy as np
import random
import torch
from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils.prims import create_prim, get_prim_at_path
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder
from pxr import UsdPhysics

class DynamicObstacleManager:
    """管理动态障碍物的类"""
    def __init__(self, world, num_objects=20, area_size=10.0):
        self.world = world
        self.num_objects = num_objects
        self.area_size = area_size
        self.obstacles = []
        self.velocities = []
        
        # 障碍物参数
        self.MIN_H, self.MAX_H = 0.5, 2.0
        self.MIN_W, self.MAX_W = 0.4, 0.8
        self.SPEED_RANGE = (0.5, 2.0) # 米/秒
        
        self._setup_obstacles()

    def _setup_obstacles(self):
        print(f"正在生成 {self.num_objects} 个动态障碍物 (Kinematic)...")
        for i in range(self.num_objects):
            # 随机位置
            x = random.uniform(-self.area_size/2, self.area_size/2)
            y = random.uniform(-self.area_size/2, self.area_size/2)
            
            # 随机尺寸
            h = random.uniform(self.MIN_H, self.MAX_H)
            w = random.uniform(self.MIN_W, self.MAX_W)
            
            # 随机颜色
            color = np.array([random.random(), random.random(), random.random()])
            
            prim_path = f"/World/Obstacles/obj_{i:02d}"
            name = f"obstacle_{i:02d}"

            # 创建物体
            if random.random() < 0.5:
                obj = DynamicCuboid(
                    prim_path=prim_path,
                    name=name,
                    position=np.array([x, y, h/2]),
                    scale=np.array([w, w, h]),
                    color=color
                )
            else:
                obj = DynamicCylinder(
                    prim_path=prim_path,
                    name=name,
                    position=np.array([x, y, h/2]),
                    radius=w/2,
                    height=h,
                    color=color
                )
            
            # 关键步骤：设置为运动学刚体 (Kinematic)
            # 1. 获取 Prim
            prim = get_prim_at_path(prim_path)
            # 2. 应用或获取 RigidBodyAPI 并设置 kinematicEnabled
            rb_api = UsdPhysics.RigidBodyAPI.Get(self.world.stage, prim_path)
            if not rb_api:
                rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
            rb_api.CreateKinematicEnabledAttr(True)
            
            # 3. 包装为 RigidPrim 方便后续操作
            rigid_obj = RigidPrim(prim_path=prim_path, name=f"{name}_rigid")
            self.obstacles.append(rigid_obj)
            
            # 随机初始速度方向
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(*self.SPEED_RANGE)
            vx = np.cos(angle) * speed
            vy = np.sin(angle) * speed
            self.velocities.append(np.array([vx, vy, 0.0]))

    def update(self, dt):
        """每步仿真更新障碍物位置"""
        for i, obj in enumerate(self.obstacles):
            # 获取当前位置
            pos, rot = obj.get_world_pose()
            
            # 计算新位置
            new_pos = pos + self.velocities[i] * dt
            
            # 边界检查：如果超出范围则反弹
            if abs(new_pos[0]) > self.area_size / 2:
                self.velocities[i][0] *= -1
                new_pos[0] = np.clip(new_pos[0], -self.area_size/2, self.area_size/2)
            if abs(new_pos[1]) > self.area_size / 2:
                self.velocities[i][1] *= -1
                new_pos[1] = np.clip(new_pos[1], -self.area_size/2, self.area_size/2)
            
            # 应用新位置 (由于是 Kinematic，我们需要手动设置位姿)
            obj.set_world_pose(position=new_pos)

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
