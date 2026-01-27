import os
import sys
import cv2
import torch
import logging
import numpy as np
import os.path as op
from collections import deque
from pathlib import Path
from torchvision import transforms
from functools import partial

# 假设该脚本位于项目根目录或被正确添加到了 PYTHONPATH
sys.path.append(op.abspath(op.dirname(__file__)))
from scripts.v2ce_3d import V2ce3d
from scripts.LDATI import sample_voxel_statistical

# 配置 Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('V2CE_Inference')

class V2CEPredictor:
    def __init__(self, model_path='weights/v2ce_3d.pt', device='cuda', 
                 height=260, width=346, seq_len=16, fps=30, 
                 ceil=10, upper_bound_percentile=98):
        """
        初始化 V2CE 推理器
        """
        self.device = device
        self.height = height
        self.width = width
        self.seq_len = seq_len
        self.fps = fps
        self.ceil = ceil
        self.upper_bound_percentile = upper_bound_percentile
        
        # 加载模型
        logger.info(f"Loading V2CE model from {model_path} to {device}...")
        self.model = V2ce3d()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.eval().to(device)
        
        # 初始化 LDATI 采样策略
        self.ldati = partial(sample_voxel_statistical, fps=fps, bidirectional=False, additional_events_strategy='slope')
        
        # 帧归一化变换
        self.frame_normalize = transforms.Compose([transforms.Normalize([0.153, 0.153], [0.165, 0.165])])
        
        # 帧缓冲区
        self.frame_buffer = deque(maxlen=seq_len + 1)
        
        logger.info("V2CE Predictor initialized.")

    def preprocess_sequence(self, frames):
        """预处理帧序列"""
        resized = [cv2.resize(f.astype(np.float32) / 255.0, (int(f.shape[1] / f.shape[0] * self.height), self.height)) for f in frames]
        images = np.stack(resized, axis=0)
        # 构造图像对: [T, 2, H, W]
        image_units = torch.tensor(np.stack([images[:-1], images[1:]], axis=1)).unsqueeze(0)
        image_units = self.frame_normalize(image_units)
        return image_units

    @torch.no_grad()
    def infer_center_image_unit(self, image_units):
        """中心裁剪推理"""
        # Center crop
        image_units = image_units[..., image_units.shape[-1]//2 - self.width//2 : image_units.shape[-1]//2 + self.width//2]
        inputs = image_units.float().to(self.device)
        
        # [DEBUG] 打印输入统计
        # print(f"[V2CE DEBUG] Input Tensor: Min={inputs.min():.4f}, Max={inputs.max():.4f}, Mean={inputs.mean():.4f}")
        
        outputs = self.model(inputs)
        
        # [DEBUG] 打印输出统计
        if outputs.max().item() == 0:
             print(f"[V2CE DEBUG] Model Output is ALL ZEROS! Something is wrong.")
        #else:
        #     print(f"[V2CE DEBUG] Model Output Raw: Min={outputs.min():.6f}, Max={outputs.max():.6f}, Mean={outputs.mean():.6f}")

        return outputs

    def make_event_frame(self, voxel_grid, keep_polarity=True):
        """生成可视化事件帧"""
        #[DEBUG] 打印输入形状
        #print(f"[V2CE DEBUG] make_event_frame Input Shape: {voxel_grid.shape}")
        
        if voxel_grid.ndim == 5:
            B, P, L, H, W = voxel_grid.shape
            if keep_polarity:
                efs = np.sum(voxel_grid, axis=2)
                if P >= 2:
                    pos = efs[:, 0]
                    neg = efs[:, 1]
                    efs = np.stack([pos, neg, np.zeros_like(pos)], axis=1)
                    efs_flatten = efs.flatten()
                    efs_flatten = efs_flatten[efs_flatten > 0]
                    if efs_flatten.size == 0:
                        print("[V2CE DEBUG] efs_flatten is empty! Returning black frame.")
                        return np.zeros((H, W, 3), dtype=np.uint8)
                    efs_upper_bound = min(np.percentile(efs_flatten, self.upper_bound_percentile), self.ceil)
                    
                    # [DEBUG] 打印边界值
                    # print(f"[V2CE DEBUG] Upper Bound: {efs_upper_bound}, Max Val: {efs_flatten.max()}")
                    
                    if efs_upper_bound <= 0:
                         # 防止除以零
                         efs_upper_bound = 1.0
                         
                    efs = np.clip(efs, 0, efs_upper_bound) / efs_upper_bound
                else:
                    # 单极性处理逻辑简化...
                    return np.zeros((H, W, 3), dtype=np.uint8) 
            else:
                # 不保留极性逻辑简化...
                return np.zeros((H, W, 3), dtype=np.uint8)
        else:
             print(f"[V2CE DEBUG] Unexpected ndim: {voxel_grid.ndim}")
             raise ValueError('voxel_grid must be 5D')
             
        efs = np.moveaxis(efs, 1, -1)
        rgb = (efs[0] * 255).astype(np.uint8)
        # RGB -> BGR for OpenCV
        if rgb.shape[-1] == 3:
            bgr = rgb[..., [2, 1, 0]]
        else:
            bgr = np.zeros((H, W, 3), dtype=np.uint8)
        return bgr

    def predict(self, frame_bgr):
        """
        主推理函数
        Args:
            frame_bgr: OpenCV 读取的 BGR 图像 (H, W, 3)
        Returns:
            event_frame_bgr: 可视化的事件帧 (H, W, 3)，如果缓冲区不足则返回 None
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.frame_buffer.append(gray)
        
        # 缓冲区不足时不推理
        if len(self.frame_buffer) < 2:
            return None
            
        # 预处理
        image_units = self.preprocess_sequence(list(self.frame_buffer))
        
        # 模型推理
        pred_voxel = self.infer_center_image_unit(image_units)
        
        # 取最后一个时间步的结果
        pred_voxel = pred_voxel[:, -1:, ...]
        
        # 转换为可视化帧 (Stage 2 LDATI 可选，这里主要为了可视化)
        # 如果需要稀疏事件列表，可以在这里调用 self.ldati(stage2_input)
        
        L, P, C, H, W = pred_voxel.shape
        stage2_input = pred_voxel.reshape(L, 2, 10, H, W).to(self.device)
        ef_frame = self.make_event_frame(stage2_input.cpu().numpy(), keep_polarity=True)
        
        return ef_frame
