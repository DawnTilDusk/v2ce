import numpy as np
from omni.isaac.core.utils.prims import create_prim, is_prim_path_valid, delete_prim
from omni.isaac.core.prims import RigidPrim

class BlockingObstacleManager:
    """
    管理"拦路"障碍物：检测机器狗移动方向，并在前方生成阻挡物
    """
    def __init__(self, world, max_obstacles=3):
        self.world = world
        self.max_obstacles = max_obstacles
        self.obstacle_pool = []
        self.current_index = 0
        self.prim_base_path = "/World/BlockingObstacles"
        
        # 确保父 Prim 存在
        if not is_prim_path_valid(self.prim_base_path):
            create_prim(self.prim_base_path, "Xform")
            
        self._initialize_pool()

    def _initialize_pool(self):
        """预先创建障碍物池 (使用立方体作为路障)"""
        for i in range(self.max_obstacles):
            prim_path = f"{self.prim_base_path}/block_{i}"
            initial_pos = np.array([0.0, 0.0, -100.0])
            
            if not is_prim_path_valid(prim_path):
                # 创建一个较大的立方体路障 (宽1m, 高0.5m, 厚0.2m)
                create_prim(
                    prim_path=prim_path,
                    prim_type="Cube",
                    position=initial_pos,
                    scale=np.array([0.2, 1.0, 0.5]), # 默认Cube边长2? 需要确认缩放
                    # Isaac Sim Cube 默认边长可能是 1.0 或 2.0，通常 scale 1 = 1m
                    # 假设我们需要一个扁平的板子
                )
            
            rigid_prim = RigidPrim(prim_path=prim_path, name=f"block_{i}")
            rigid_prim.enable_rigid_body_physics()
            # 增加质量，让它不容易被撞飞
            rigid_prim.set_mass(100.0) 
            
            self.obstacle_pool.append(rigid_prim)

    def spawn_blocker(self, robot_pos, robot_vel, min_dist=1.0, max_dist=2.5):
        """
        根据机器狗的速度向量，在前方生成阻挡物
        """
        # 1. 检测移动方向
        speed = np.linalg.norm(robot_vel[:2])
        if speed < 0.1:
            # 如果静止不动，就不生成，或者随机生成在四周
            # print("[Blocking] 机器狗静止，不生成路障")
            return
            
        # 归一化速度向量 (仅水平方向)
        move_dir = robot_vel[:2] / speed
        move_dir_3d = np.array([move_dir[0], move_dir[1], 0.0])
        
        # 2. 计算生成位置
        spawn_dist = np.random.uniform(min_dist, max_dist)
        spawn_pos = robot_pos + move_dir_3d * spawn_dist
        
        # 优化1：限制生成范围在障碍物区域内 (假设区域大小为 +/- 6.0m)
        # 这样路障就不会生成在仓库外面
        spawn_pos[0] = np.clip(spawn_pos[0], -6.0, 6.0)
        spawn_pos[1] = np.clip(spawn_pos[1], -6.0, 6.0)
        
        # 放在地面上 (假设路障高度 0.5m，中心在 0.25m)
        spawn_pos[2] = 0.25  
        
        # 3. 计算旋转：让路障"横"在路上
        # 移动方向是 move_dir (x, y)
        # 路障应该垂直于移动方向
        # 计算 Yaw 角：atan2(y, x)
        yaw = np.arctan2(move_dir[1], move_dir[0])
        # 转换为四元数 (绕Z轴旋转)
        # Isaac Sim quaternion is [w, x, y, z]
        # 半角公式: w = cos(theta/2), z = sin(theta/2)
        theta = yaw
        # 如果不做任何旋转，Cube是轴对齐的。我们需要它垂直于运动方向。
        # 假设 Cube 缩放是 [0.2, 1.0, 0.5] (X轴薄，Y轴宽)
        # 那么它的"宽面"法线是 X 轴。
        # 如果我们直接用 yaw 旋转，它的 X 轴就会指向运动方向 -> 变成"纵向"拦路 (像隔离带)
        # 我们希望它是"横向"拦路 (像墙)。
        # 所以它的法线(X轴)应该指向运动方向。
        # 结论：直接用 yaw 旋转即可，因为 Cube 的 X 轴(薄边) 会指向运动方向，Y 轴(宽边) 会垂直于运动方向。
        
        q_w = np.cos(theta / 2)
        q_z = np.sin(theta / 2)
        orientation = np.array([q_w, 0.0, 0.0, q_z])
        
        # 4. 从池中取出并放置
        blocker = self.obstacle_pool[self.current_index]
        self.current_index = (self.current_index + 1) % self.max_obstacles
        
        blocker.set_world_pose(position=spawn_pos, orientation=orientation)
        blocker.set_linear_velocity(np.zeros(3))
        blocker.set_angular_velocity(np.zeros(3))
        
        print(f"[Blocking] 在前方 {spawn_dist:.2f}m 处生成路障!")

    def cleanup(self):
        if is_prim_path_valid(self.prim_base_path):
            delete_prim(self.prim_base_path)
