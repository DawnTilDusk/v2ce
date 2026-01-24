import numpy as np
import random
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils.prims import get_prim_at_path
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder
from pxr import UsdPhysics

class DynamicObstacleManager:
    """管理动态障碍物的类，支持 Kinematic 运动模式和边界反弹"""
    def __init__(self, world, num_objects=20, area_size=10.0, excluded_center_radius=1.5):
        self.world = world
        self.num_objects = num_objects
        self.area_size = area_size
        self.excluded_radius = excluded_center_radius
        self.obstacles = []
        self.velocities = []
        
        # 障碍物尺寸与速度参数
        self.MIN_H, self.MAX_H = 0.5, 1.5
        self.MIN_W, self.MAX_W = 0.3, 0.6
        self.SPEED_RANGE = (0.5, 1.5) # 米/秒
        
        self._setup_obstacles()

    def _setup_obstacles(self):
        print(f"[ObstacleManager] 正在生成 {self.num_objects} 个动态障碍物...")
        for i in range(self.num_objects):
            # 随机位置，直到不在排除半径内
            while True:
                x = random.uniform(-self.area_size/2, self.area_size/2)
                y = random.uniform(-self.area_size/2, self.area_size/2)
                if np.sqrt(x**2 + y**2) > self.excluded_radius:
                    break
            
            h = random.uniform(self.MIN_H, self.MAX_H)
            w = random.uniform(self.MIN_W, self.MAX_W)
            color = np.array([random.random(), random.random(), random.random()])
            
            prim_path = f"/World/Obstacles/obj_{i:02d}"
            name = f"obstacle_{i:02d}"

            # 创建几何体
            if random.random() < 0.6:
                self.world.scene.add(
                    DynamicCuboid(
                        prim_path=prim_path,
                        name=name,
                        position=np.array([x, y, h/2]),
                        scale=np.array([w, w, h]),
                        color=color
                    )
                )
            else:
                self.world.scene.add(
                    DynamicCylinder(
                        prim_path=prim_path,
                        name=name,
                        position=np.array([x, y, h/2]),
                        radius=w/2,
                        height=h,
                        color=color
                    )
                )
            
            # 设置为 Kinematic 模式（手动控制位置，不受重力影响，但有碰撞体）
            rb_api = UsdPhysics.RigidBodyAPI.Get(self.world.stage, prim_path)
            if not rb_api:
                prim = get_prim_at_path(prim_path)
                rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
            rb_api.CreateKinematicEnabledAttr(True)
            
            # 包装为 RigidPrim
            rigid_obj = RigidPrim(prim_path=prim_path, name=f"{name}_rigid")
            self.obstacles.append(rigid_obj)
            
            # 随机初始速度方向
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(*self.SPEED_RANGE)
            vx = np.cos(angle) * speed
            vy = np.sin(angle) * speed
            self.velocities.append(np.array([vx, vy, 0.0]))

    def update(self, dt):
        """每步仿真更新障碍物位置，处理边界反弹"""
        for i, obj in enumerate(self.obstacles):
            pos, rot = obj.get_world_pose()
            
            # 计算预测位置
            new_pos = pos + self.velocities[i] * dt
            
            # 边界检查与反弹逻辑
            limit = self.area_size / 2
            if abs(new_pos[0]) > limit:
                self.velocities[i][0] *= -1
                new_pos[0] = np.clip(new_pos[0], -limit, limit)
            if abs(new_pos[1]) > limit:
                self.velocities[i][1] *= -1
                new_pos[1] = np.clip(new_pos[1], -limit, limit)
            
            # 更新位姿
            obj.set_world_pose(position=new_pos)
