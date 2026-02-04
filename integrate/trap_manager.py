import numpy as np
from omni.isaac.core.utils.prims import create_prim, is_prim_path_valid, delete_prim
from omni.isaac.core.prims import RigidPrim
from pxr import UsdPhysics

class TrapManager:
    """
    管理"围墙陷阱"：生成三面墙（前、左、右）包围机器狗
    """
    def __init__(self, world, max_traps=2):
        self.world = world
        self.max_traps = max_traps
        # 对象池：每个 trap 包含 3 个 wall (front, left, right)
        # 结构: [ {'front': prim, 'left': prim, 'right': prim}, ... ]
        self.trap_pool = [] 
        self.current_index = 0
        self.prim_base_path = "/World/Traps"
        
        # 确保父 Prim 存在
        if not is_prim_path_valid(self.prim_base_path):
            create_prim(self.prim_base_path, "Xform")
            
        self._initialize_pool()

    def _initialize_pool(self):
        """预先创建围墙池"""
        for i in range(self.max_traps):
            trap_group = {}
            parts = ['front', 'left', 'right']
            
            for part in parts:
                prim_path = f"{self.prim_base_path}/trap_{i}_{part}"
                initial_pos = np.array([0.0, 0.0, -100.0]) # 藏在地下
                
                if not is_prim_path_valid(prim_path):
                    # 墙体尺寸：长 2m, 高 1m, 厚 0.2m
                    # 用户希望把陷阱调大一点 (在之前的缩小版基础上回调)
                    # 缩小版: [0.2, 1.2, 0.5]
                    # 现在的版本: [0.2, 1.6, 0.8] (长 1.6m, 高 0.8m)
                    scale = np.array([0.2, 1.6, 0.8]) 
                    
                    create_prim(
                        prim_path=prim_path,
                        prim_type="Cube",
                        position=initial_pos,
                        scale=scale,
                    )
                
                rigid_prim = RigidPrim(prim_path=prim_path, name=f"trap_{i}_{part}")
                rigid_prim.enable_rigid_body_physics()
                rigid_prim.set_mass(500.0) # 很重，推不动
                
                # 修复：防止陷阱穿过地面掉下去
                # 将刚体设置为 Kinematic (运动学)
                # 注意：RigidPrim (omni.isaac.core) 没有 set_kinematic_enabled 方法
                # 需要使用 UsdPhysics API 直接操作 USD Prim
                
                stage = self.world.stage
                prim = stage.GetPrimAtPath(prim_path)
                # 使用 UsdPhysics.RigidBodyAPI
                rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
                rb_api.CreateKinematicEnabledAttr(True)
                
                trap_group[part] = rigid_prim
            
            self.trap_pool.append(trap_group)
            
        # 记录当前活跃的 trap 信息，用于后续检测距离
        # 结构: {'index': int, 'center_pos': np.array, 'active': bool}
        self.active_trap_info = {'active': False}

    def spawn_trap(self, robot_pos, robot_rot):
        """
        在机器狗周围生成三面墙（U型包围）
        """
        # 1. 获取当前的 trap 组
        trap_index = self.current_index
        trap = self.trap_pool[trap_index]
        self.current_index = (self.current_index + 1) % self.max_traps
        
        # ... (中间计算代码保持不变) ...
        
        # 2. 计算机器狗的朝向
        q_w, q_x, q_y, q_z = robot_rot
        yaw = np.arctan2(2.0*(q_w*q_z + q_x*q_y), 1.0 - 2.0*(q_y*q_y + q_z*q_z))
        
        # 基础方向向量
        forward_dir = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        left_dir = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        right_dir = -left_dir
        
        # 3. 计算三面墙的位置和旋转
        # 距离机器狗的距离 (调大包围圈)
        dist_front = 1.2 # 之前是 0.8
        dist_side = 0.9  # 之前是 0.6
        
        # 墙高 0.8m，中心高度应为 0.4m，确保贴地
        wall_height_center = 0.4
        
        # 前墙位置
        pos_front = robot_pos + forward_dir * dist_front
        pos_front[2] = wall_height_center
        
        # 左墙位置
        pos_left = robot_pos + left_dir * dist_side
        pos_left[2] = wall_height_center
        
        # 右墙位置
        pos_right = robot_pos + right_dir * dist_side
        pos_right[2] = wall_height_center
        
        # 4. 计算旋转四元数
        theta_front = yaw
        q_front = np.array([np.cos(theta_front/2), 0, 0, np.sin(theta_front/2)])
        
        theta_side = yaw + np.pi/2
        q_side = np.array([np.cos(theta_side/2), 0, 0, np.sin(theta_side/2)])
        
        # 5. 设置位姿
        trap['front'].set_world_pose(position=pos_front, orientation=q_front)
        trap['left'].set_world_pose(position=pos_left, orientation=q_side)
        trap['right'].set_world_pose(position=pos_right, orientation=q_side)
        
        # 确保速度为0 (虽然是Kinematic，但为了保险)
        # 移除 set_linear_velocity 调用，因为对 Kinematic 物体调用会产生警告
        # trap['front'].set_linear_velocity(np.zeros(3))
        # trap['front'].set_angular_velocity(np.zeros(3))
        
        print(f"[Trap] 警告！已生成U型围墙陷阱！")
        
        # 记录活跃状态
        self.active_trap_info = {
            'index': trap_index,
            'center_pos': robot_pos.copy(), # 记录生成时的中心点 (即陷阱中心)
            'active': True
        }

    def update(self, robot_pos):
        """
        每帧调用：检查机器狗是否远离了当前的陷阱
        """
        if not self.active_trap_info['active']:
            return

        # 计算机器狗与陷阱中心的距离
        trap_center = self.active_trap_info['center_pos']
        # 只计算水平距离
        dist = np.linalg.norm(robot_pos[:2] - trap_center[:2])
        
        # 如果距离超过 3.0 米，认为已经逃脱
        if dist > 3.0:
            print(f"[Trap] 机器狗已逃脱 ({dist:.2f}m)，移除陷阱。")
            self.hide_active_trap()

    def hide_active_trap(self):
        """将当前活跃的陷阱移到地下"""
        if not self.active_trap_info['active']:
            return
            
        trap_index = self.active_trap_info['index']
        trap = self.trap_pool[trap_index]
        
        hidden_pos = np.array([0.0, 0.0, -100.0])
        
        trap['front'].set_world_pose(position=hidden_pos)
        trap['left'].set_world_pose(position=hidden_pos)
        trap['right'].set_world_pose(position=hidden_pos)
        
        self.active_trap_info['active'] = False

    def cleanup(self):
        if is_prim_path_valid(self.prim_base_path):
            delete_prim(self.prim_base_path)
