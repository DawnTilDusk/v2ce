import numpy as np
import random
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils.prims import get_prim_at_path
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
            
            print(f"[ENV INFO] 正在创建 第 {i+1}/{self.num_objects} 个障碍物")
            prim_path = f"/World/Obstacles/obj_{i:02d}"
            name = f"obstacle_{i:02d}"

            # 创建物体
            print(f"[ENV INFO] 正在创建 {name} (位置: {[x, y, h/2]}, 尺寸: {[w, w, h]})")
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
            #print(f"[ENV INFO] 正在获取 {prim_path} 的 Prim")
            prim = get_prim_at_path(prim_path)
            # 2. 应用或获取 RigidBodyAPI 并设置 kinematicEnabled
            #print(f"[ENV INFO] 正在应用 RigidBodyAPI 到 {prim_path}")
            rb_api = UsdPhysics.RigidBodyAPI.Get(self.world.stage, prim_path)
            if not rb_api:
                rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
            rb_api.CreateKinematicEnabledAttr(True)
            
            # 3. 包装为 RigidPrim 方便后续操作
            #print(f"[ENV INFO] 正在包装 {prim_path} 为 RigidPrim")
            rigid_obj = RigidPrim(prim_path=prim_path, name=f"{name}_rigid")
            self.obstacles.append(rigid_obj)
            
            # 随机初始速度方向
            #print(f"[ENV INFO] 正在设置 {name} 的初始速度 (方向: {angle:.2f}, 速度: {speed:.2f})")
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
