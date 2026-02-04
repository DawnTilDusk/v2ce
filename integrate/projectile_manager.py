import numpy as np
import random
from omni.isaac.core.utils.prims import create_prim, is_prim_path_valid, delete_prim
from omni.isaac.core.prims import RigidPrim
from pxr import UsdPhysics, Gf

class ProjectileManager:
    """
    管理抛物线球体障碍物的生成与销毁
    """
    def __init__(self, world, max_projectiles=5):
        self.world = world
        self.max_projectiles = max_projectiles
        self.projectile_pool = []  # 存储所有的球体 Prim 对象池
        self.current_index = 0
        self.prim_base_path = "/World/Projectiles"
        
        # 确保父 Prim 存在
        if not is_prim_path_valid(self.prim_base_path):
            create_prim(self.prim_base_path, "Xform")
            
        # 预先创建所有球体
        self._initialize_pool()

    def _initialize_pool(self):
        """预先创建对象池，避免运行时创建/删除导致的物理引擎崩溃"""
        for i in range(self.max_projectiles):
            prim_path = f"{self.prim_base_path}/ball_{i}"
            
            # 初始位置放在很远的地方
            initial_pos = np.array([0.0, 0.0, -100.0])
            
            if not is_prim_path_valid(prim_path):
                create_prim(
                    prim_path=prim_path,
                    prim_type="Sphere",
                    position=initial_pos,
                    scale=np.array([0.2, 0.2, 0.2]), # 半径 0.2m
                )
            
            rigid_prim = RigidPrim(prim_path=prim_path, name=f"ball_{i}")
            rigid_prim.enable_rigid_body_physics()
            
            # 存入对象池
            self.projectile_pool.append(rigid_prim)

    def spawn_projectile(self, robot_pos, robot_rot, forward_dist=3.0, height=2.0):
        """
        在机器人前方生成一个球体，并给予初始速度使其做抛物线运动
        """
        # 1. 计算生成位置：机器人前方 forward_dist 处，高度 height
        # 将四元数转换为旋转矩阵 (这里简化处理，假设机器人主要在水平面旋转)
        q_w, q_x, q_y, q_z = robot_rot
        yaw = np.arctan2(2.0*(q_w*q_z + q_x*q_y), 1.0 - 2.0*(q_y*q_y + q_z*q_z))
        forward_dir = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        
        spawn_pos = robot_pos + forward_dir * forward_dist
        spawn_pos[2] = height # 强制设定高度
        
        # 2. 从对象池获取球体 (循环使用)
        rigid_prim = self.projectile_pool[self.current_index]
        self.current_index = (self.current_index + 1) % self.max_projectiles
        
        # 瞬移到生成位置 (先设为 Kinematic 防止物理干扰，移动后再改回 Dynamic)
        # 或者直接 set_world_pose + set_linear_velocity
        # 注意：在 Isaac Sim 中，直接瞬移刚体最好同时重置速度
        
        rigid_prim.set_world_pose(position=spawn_pos)
        rigid_prim.set_angular_velocity(np.zeros(3))
        
        # 3. 施加初始速度 (抛物线)
        # 物理公式: P_target = P_start + V * t + 0.5 * g * t^2
        # 我们想让球在时间 t 后击中 robot_pos (通常 t 可以设为 0.5s - 1.0s)
        # V = (P_target - P_start - 0.5 * g * t^2) / t
        
        g = np.array([0.0, 0.0, -9.81]) # 重力加速度
        
        # 增加随机性：
        # 50% 概率直接瞄准机器狗
        # 50% 概率稍微偏一点，落在机器狗附近
        target_pos = robot_pos.copy()
        if random.random() < 0.5:
            offset = np.random.uniform(-0.5, 0.5, size=2) # 水平偏移
            target_pos[0] += offset[0]
            target_pos[1] += offset[1]
            # 目标高度设为地面附近 (0.2m)
            target_pos[2] = 0.2
        else:
            # 瞄准机器狗身体中心 (假设 0.3m 高)
            target_pos[2] = 0.3
            
        # 飞行时间 t：距离越远，飞行时间越长
        # 简单估算 t = distance / horizontal_speed
        # 设水平速度大概 5m/s
        horizontal_dist = np.linalg.norm(target_pos[:2] - spawn_pos[:2])
        flight_time = horizontal_dist / 5.0
        flight_time = np.clip(flight_time, 0.5, 1.5) # 限制飞行时间
        
        # 计算所需初速度
        # V0 = (P_target - P_start - 0.5 * g * t^2) / t
        delta_p = target_pos - spawn_pos
        velocity = (delta_p - 0.5 * g * (flight_time**2)) / flight_time
        
        # 为了增加冲击力，稍微加大向下的分量 (模拟用力扔)
        # 这里的计算已经是物理精确的"抛物线"能击中目标。
        # 如果想让球"砸"得更狠，可以减少飞行时间 t，这样初速度会变大，轨迹更平直有力。
        # 这里我们给 velocity 额外加一点向下的速度，让它不是完美的抛物线，而是"扣杀"
        velocity[2] -= 2.0 
        
        rigid_prim.set_linear_velocity(velocity)
        
        # 无需再进行 append 或 delete 操作

    def cleanup(self):
        """清理所有生成的球体"""
        # 注意：这里我们不再删除，只是建议在仿真结束时调用
        # 如果需要彻底清除，可以删除父节点
        if is_prim_path_valid(self.prim_base_path):
            delete_prim(self.prim_base_path)

    def cleanup(self):
        """清理所有生成的球体"""
        if is_prim_path_valid(self.prim_base_path):
            delete_prim(self.prim_base_path)
