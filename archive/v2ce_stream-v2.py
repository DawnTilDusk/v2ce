import os
import sys
import socket
import struct
import cv2
import torch
import logging
import argparse
import numpy as np
import os.path as op
import time
from collections import deque
from pathlib2 import Path
from torchvision import transforms
from functools import partial

sys.path.append(op.abspath('../..'))
from scripts.v2ce_3d import V2ce3d
from scripts.LDATI import sample_voxel_statistical

class SocketFrameReader:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.payload_size = struct.calcsize("L")
        self.data = b""
        self.opened = True

    def read_frame(self):
        while len(self.data) < self.payload_size:
            packet = self.sock.recv(4096)
            if not packet:
                self.opened = False
                return None
            self.data += packet
        packed_size = self.data[:self.payload_size]
        self.data = self.data[self.payload_size:]
        msg_size = struct.unpack("L", packed_size)[0]
        while len(self.data) < msg_size:
            packet = self.sock.recv(4096)
            if not packet:
                self.opened = False
                return None
            self.data += packet
        frame_bytes = self.data[:msg_size]
        self.data = self.data[msg_size:]
        buf = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        return frame

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass
        self.opened = False

def SBool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_trained_mode(model_path='./weights/v2ce_3d.pt', device='cpu'):
    model = V2ce3d()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.eval()
    model = model.to(device)
    return model

def preprocess_pair(prev_frame, cur_frame, height=260):
    frame_normalize = transforms.Compose([transforms.Normalize([0.153, 0.153], [0.165, 0.165])])
    prev = prev_frame.astype(np.float32) / 255.0
    cur = cur_frame.astype(np.float32) / 255.0
    prev = cv2.resize(prev, (int(prev.shape[1] / prev.shape[0] * height), height))
    cur = cv2.resize(cur, (int(cur.shape[1] / cur.shape[0] * height), height))
    images = np.stack([prev, cur], axis=0)
    image_units = torch.tensor(images).unsqueeze(0)
    image_units = frame_normalize(image_units)
    image_units = image_units.unsqueeze(1)
    return image_units

def preprocess_sequence(frames, height=260):
    frame_normalize = transforms.Compose([transforms.Normalize([0.153, 0.153], [0.165, 0.165])])
    resized = [cv2.resize(f.astype(np.float32) / 255.0, (int(f.shape[1] / f.shape[0] * height), height)) for f in frames]
    images = np.stack(resized, axis=0)
    image_units = torch.tensor(np.stack([images[:-1], images[1:]], axis=1)).unsqueeze(0)
    image_units = frame_normalize(image_units)
    return image_units

@torch.no_grad()
def infer_center_image_unit(model, image_units, width=346):
    image_units = image_units[..., image_units.shape[-1]//2-width//2:image_units.shape[-1]//2+width//2]
    device = next(model.parameters()).device
    inputs = image_units.float().to(device)
    outputs = model(inputs)
    pred_voxel = outputs.cpu()
    return pred_voxel

def make_event_frame(voxel_grid, fps, ceil=10, upper_bound_percentile=98, keep_polarity=True):
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
                    frame = np.zeros((H, W, 3), dtype=np.uint8)
                    return frame
                efs_upper_bound = min(np.percentile(efs_flatten, upper_bound_percentile), ceil)
                efs = np.clip(efs, 0, efs_upper_bound) / (efs_upper_bound if efs_upper_bound > 0 else 1.0)
            else:
                total = np.sum(efs, axis=1) if efs.ndim == 4 else efs[:, 0]
                total_flat = total.flatten()
                total_flat = total_flat[total_flat > 0]
                if total_flat.size == 0:
                    frame = np.zeros((H, W, 3), dtype=np.uint8)
                    return frame
                ub = min(np.percentile(total_flat, upper_bound_percentile), ceil)
                total = np.clip(total, 0, ub) / (ub if ub > 0 else 1.0)
                efs = np.stack([total, np.zeros_like(total), np.zeros_like(total)], axis=1)
        else:
            efs = np.sum(voxel_grid, axis=(1, 2))[:, np.newaxis, ...]
            efs = np.repeat(efs, 3, axis=1)
    elif voxel_grid.ndim == 4:
        B, L, H, W = voxel_grid.shape
        efs_total = np.sum(voxel_grid, axis=1)
        efs = np.stack([efs_total, np.zeros_like(efs_total), np.zeros_like(efs_total)], axis=1)
    else:
        raise ValueError('voxel_grid must be 4D or 5D')
    efs = np.moveaxis(efs, 1, -1)
    rgb = (efs[0] * 255).astype(np.uint8)
    if rgb.shape[-1] == 3:
        bgr = rgb[..., [2, 1, 0]]
    elif rgb.shape[-1] == 2:
        pos = rgb[..., 0]
        neg = rgb[..., 1]
        bgr = np.zeros((H, W, 3), dtype=np.uint8)
        bgr[..., 1] = neg
        bgr[..., 2] = pos
    else:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    return bgr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--socket_host', type=str, default='127.0.0.1')
    parser.add_argument('--socket_port', type=int, default=9999)
    parser.add_argument('--model_path', type=str, default='./weights/v2ce_3d.pt')
    parser.add_argument('--height', type=int, default=260)
    parser.add_argument('--width', type=int, default=346)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--ceil', type=int, default=10)
    parser.add_argument('--upper_bound_percentile', type=int, default=98)
    parser.add_argument('--interval', type=int, default=150)
    parser.add_argument('--seq_len', type=int, default=16)
    parser.add_argument('--vis_keep_polarity', type=SBool, default=True, nargs='?', const=True)
    parser.add_argument('-l', '--log_level', type=str, default='info')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger = logging.getLogger('V2CE-STREAM')

    try:
        reader = SocketFrameReader(args.socket_host, args.socket_port)
    except Exception as e:
        print('Failed to connect socket:', e)
        sys.exit(1)

    model = get_trained_mode(model_path=args.model_path, device=args.device)
    device = next(model.parameters()).device
    ldati = partial(sample_voxel_statistical, fps=args.fps, bidirectional=False, additional_events_strategy='slope')

    last_ts = 0
    events_list = None
    frame_buffer = deque(maxlen=args.seq_len + 1)

    try:
        while True:
            frame = reader.read_frame()
            if frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            now_ms = int(time.time() * 1000)
            frame_buffer.append(gray)
            if len(frame_buffer) >= 2 and (now_ms - last_ts >= args.interval):
                image_units = preprocess_sequence(list(frame_buffer), height=args.height)
                pred_voxel = infer_center_image_unit(model, image_units, width=args.width)
                pred_voxel = pred_voxel[:, -1:, ...]
                L, P, C, H, W = pred_voxel.shape
                stage2_input = pred_voxel.reshape(L, 2, 10, H, W).to(device)
                if stage2_input.max().item() == 0:
                    events_list = []
                else:
                    events_list = ldati(stage2_input)
                ef_frame = make_event_frame(stage2_input.cpu().numpy(), args.fps, args.ceil, args.upper_bound_percentile, args.vis_keep_polarity)
                cv2.imshow('EventFrame', ef_frame)
                cv2.imshow('RGBFrame', frame)
                last_ts = now_ms
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        reader.close()
        cv2.destroyAllWindows()
