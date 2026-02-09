import numpy as np
import random
import os
from omni.isaac.core.prims import RigidPrim, GeometryPrim
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder
from pxr import UsdPhysics, UsdGeom, Gf

class DynamicObstacleManager:
    """管理动态障碍物的类，支持 Kinematic 运动模式和边界反弹"""
    def __init__(self, world, num_objects=20, area_size=10.0, excluded_center_radius=1.5):
        self.world = world
        self.num_objects = num_objects
        self.area_size = area_size
        self.excluded_radius = excluded_center_radius
        self.obstacles = []
        self.velocities = []
        self.z_offsets = [] # 存储每个障碍物的高度偏移
        self._positions = [] # 存储每个障碍物的当前位置 (替代 get_world_pose)
        
        # 障碍物尺寸与速度参数
        self.MIN_H, self.MAX_H = 0.3, 0.8  # 调低高度，防止遮挡
        self.MIN_W, self.MAX_W = 0.3, 0.5  # 稍微调小宽度
        self.SPEED_RANGE = (0.5, 1.5) # 米/秒
        
        # 资产路径列表 - 使用相对路径
        # 获取当前脚本所在目录 (integrate/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取项目根目录 (假设 assets 在 integrate 的上一级)
        project_root = os.path.dirname(current_dir)
        assets_dir = os.path.join(project_root, "assets")
        
        self.ASSET_LIST = [
            os.path.join(assets_dir, "Containers", "Cardboard", "Cardbox_A1.usd"),
            os.path.join(assets_dir, "Containers", "Cardboard", "Cardbox_B1.usd"),
            os.path.join(assets_dir, "Containers", "Cardboard", "Cardbox_C1.usd"),
            os.path.join(assets_dir, "Containers", "Wooden", "WoodenCrate_A1.usd"),
            os.path.join(assets_dir, "Containers", "Wooden", "WoodenCrate_B1.usd"),
            os.path.join(assets_dir, "Pallets", "Pallet_A1.usd")
        ]
        
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
            
            # 预先定义 z_offset，稍后赋值
            z_offset = 0.0

            h = random.uniform(self.MIN_H, self.MAX_H)
            w = random.uniform(self.MIN_W, self.MAX_W)
            color = np.array([random.random(), random.random(), random.random()])
            
            prim_path = f"/World/Obstacles/obj_{i:02d}"
            name = f"obstacle_{i:02d}"

            # 创建几何体 - 混合使用本地 USD 资产和几何体
            # 50% 概率使用真实的本地资产
            use_asset = False
            selected_asset_path = None
            
            if random.random() < 0.5:
                # 随机选择一个资产
                candidate_path = random.choice(self.ASSET_LIST)
                # 检查文件是否存在
                import os
                if os.path.exists(candidate_path):
                     use_asset = True
                     selected_asset_path = candidate_path

            if use_asset:
                # 使用本地 USD 资产
                add_reference_to_stage(usd_path=selected_asset_path, prim_path=prim_path)
                
                # 包装为 GeometryPrim 以便设置碰撞
                # 工业资产包通常单位是厘米(cm)，而 Isaac Sim 环境单位是米(m)
                # 所以通常需要缩小 100 倍 (0.01)
                base_scale = 0.01 
                
                # 针对不同资产的微调
                scale_val = base_scale * 0.8 # 默认稍微小一点
                if "Pallet" in selected_asset_path:
                    scale_val = base_scale * 1.5
                elif "WoodenCrate" in selected_asset_path:
                    scale_val = base_scale * 0.5 
                
                rigid_obj = RigidPrim(prim_path=prim_path, name=name, scale=np.array([scale_val, scale_val, scale_val]))
                
                self.world.scene.add(rigid_obj)
                
                # 设置初始位置
                # USD 道具原点通常在底部，设为 0.0 即可贴地
                z_offset = 0.0
                rigid_obj.set_world_pose(position=np.array([x, y, z_offset]))
                self.z_offsets.append(z_offset)
                self._positions.append(np.array([x, y, z_offset]))

            elif random.random() < 0.6:
                # 使用立方体
                z_offset = h / 2
                rigid_obj = DynamicCuboid(
                    prim_path=prim_path,
                    name=name,
                    position=np.array([x, y, z_offset]),
                    scale=np.array([w, w, h]),
                    color=color
                )
                self.world.scene.add(rigid_obj)
                self.z_offsets.append(z_offset)
                self._positions.append(np.array([x, y, z_offset]))

            else:
                # 使用圆柱体
                z_offset = h / 2
                rigid_obj = DynamicCylinder(
                    prim_path=prim_path,
                    name=name,
                    position=np.array([x, y, z_offset]),
                    radius=w/2,
                    height=h,
                    color=color
                )
                self.world.scene.add(rigid_obj)
                self.z_offsets.append(z_offset)
                self._positions.append(np.array([x, y, z_offset]))
            
            # 设置为 Kinematic 模式（手动控制位置，不受重力影响，但有碰撞体）
            rb_api = UsdPhysics.RigidBodyAPI.Get(self.world.stage, prim_path)
            if not rb_api:
                prim = get_prim_at_path(prim_path)
                rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
            
            rb_api.CreateKinematicEnabledAttr(True)
            
            # 包装为 RigidPrim
            # 注意：不要在这里再次初始化 RigidPrim，因为上面可能已经创建过了
            # 我们可以直接使用上面创建的 rigid_obj 对象
            if 'rigid_obj' in locals():
                self.obstacles.append(rigid_obj)
            else:
                 # 如果上面没有创建（不太可能），再创建一次
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
        # 调试：每 100 步打印一次第一个障碍物的位置，确认是否在移动
        # static 变量模拟
        if not hasattr(self, "_debug_counter"): self._debug_counter = 0
        self._debug_counter += 1
        
        for i, obj in enumerate(self.obstacles):
            # 获取当前位置：使用我们自己维护的状态
            # 注意：_positions 已经在 _setup_obstacles 中初始化
            current_pos = self._positions[i]
            
            # 计算预测位置
            new_pos = current_pos + self.velocities[i] * dt
            
            # 边界检查与反弹逻辑
            limit = self.area_size / 2
            if abs(new_pos[0]) > limit:
                self.velocities[i][0] *= -1
                new_pos[0] = np.clip(new_pos[0], -limit, limit)
            if abs(new_pos[1]) > limit:
                self.velocities[i][1] *= -1
                new_pos[1] = np.clip(new_pos[1], -limit, limit)
            
            # 更新内部状态
            self._positions[i] = new_pos

            # --- 终极移动方案：底层 USD 操作 ---
            try:
                # 获取 Prim
                prim = get_prim_at_path(obj.prim_path)
                if prim:
                    # 使用 Xformable API
                    xform = UsdGeom.Xformable(prim)
                    
                    # 查找或创建 translate 操作
                    # 通常 Isaac Sim 的对象有一个 "xformOp:translate" 属性
                    translate_op = None
                    for op in xform.GetOrderedXformOps():
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            translate_op = op
                            break
                    
                    if not translate_op:
                        translate_op = xform.AddTranslateOp()
                    
                    # 设置新位置 (Gf.Vec3d)
                    # 关键修复：Z 轴应该使用初始化的 z_offset，而不是计算出的 new_pos[2]
                    target_z = self.z_offsets[i]
                    translate_op.Set(Gf.Vec3d(float(new_pos[0]), float(new_pos[1]), float(target_z)))
            except Exception as e:
                if self._debug_counter % 200 == 0:
                    print(f"[Error] 更新障碍物 {obj.name} 失败: {e}")

            if self._debug_counter % 200 == 0 and i == 0:
                 print(f"[DEBUG] Obstacle[0] target: {new_pos}")