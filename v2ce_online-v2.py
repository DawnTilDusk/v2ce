import os
import sys
import cv2
import torch
import logging
import argparse
import numpy as np
import os.path as op
import time
from pathlib2 import Path
from torchvision import transforms
from functools import partial

sys.path.append(op.abspath('../..'))
from scripts.v2ce_3d import V2ce3d
from scripts.LDATI import sample_voxel_statistical

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

@torch.no_grad()
def video_to_voxels(model, image_paths=None, vidcap=None, infer_type='center', 
                              seq_len=16, width=346, height=260, batch_size=1):
    """ Infer the voxel from the video or image sequence
    Args:
        model: the trained model
        image_paths: the paths to the images
        vidcap: the video reader
        infer_type: the type of inference, can be center or pano
        seq_len: the sequence length
        width: the width of the image
        height: the height of the image
        batch_size: batch size for inference
    Returns:
        all_pred_voxel: the predicted voxel
    """
    assert image_paths is not None or vidcap is not None
    infer_video = True if vidcap is not None else False
    frame_count = vidcap.frame_count if infer_video else len(image_paths)
    sequence_num = np.ceil((frame_count-1)/seq_len).astype(int)
    mode = (frame_count-1) % seq_len
    starting_indexes = np.arange(sequence_num) * seq_len
    if mode != 0:
        starting_indexes[-1] -= (seq_len-mode)

    logger.debug(f'Found {frame_count} images, divided into {sequence_num} sequences')
    logger.debug(f'Starting indexes: {starting_indexes}')
    logger.debug(f'Mode: {mode}')
    
    all_pred_voxel = []
    batch_idx = 0
    input_image_batches = []
    for seq_idx in tqdm(range(len(starting_indexes))):
        starting_idx = starting_indexes[seq_idx]
        ending_idx = starting_idx + seq_len + 1 # +1 for geting the last frame of the last image unit
        logger.debug(f'Using images {starting_idx} to {ending_idx-1}')
            
        if infer_video:
            # Load rgb images as grayscale
            images = vidcap.read_frames_at_indices(range(starting_idx, ending_idx))
        else:
            image_paths_seq = image_paths[starting_idx:ending_idx]
            # Load rgb images as grayscale
            images = np.stack([cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in image_paths_seq], axis=0)
        
        image_units = image_pre_processing(images, height=height)
        resized_width = image_units.shape[-1]
        
        input_image_batches.append(image_units[np.newaxis, ...])
        batch_idx += 1
        if batch_idx == batch_size or seq_idx == len(starting_indexes)-1:
            # Concatenate the input image batches
            if len(input_image_batches) > 1:
                input_image_batches = torch.cat(input_image_batches, dim=0)
            elif len(input_image_batches) == 1:
                input_image_batches = input_image_batches[0]
            else:
                raise ValueError('No input image batches')
            
            logger.debug(f'Input_image_batches shape: {input_image_batches.shape}')
            
            # Infer the voxel
            if infer_type == 'center':
                out_width = width
                pred_voxel = infer_center_image_unit(model, input_image_batches, width)
            elif infer_type == 'pano':
                out_width = resized_width
                pred_voxel = infer_pano_image_unit(model, input_image_batches, width)        
            else:
                raise ValueError(f'Invalid infer_type {infer_type}')
            batch_idx = 0
            input_image_batches = []
            
            all_pred_voxel.append(pred_voxel.cpu().detach().numpy())
        
    all_pred_voxel = merge_voxels(all_pred_voxel, height=height, width=out_width, mode=mode)
    
    logger.debug(f"predicted voxels shape: {all_pred_voxel.shape}")
    return all_pred_voxel
        
def merge_voxels(voxel_list, height=260, width=346, mode=0):
    """
    Merge the voxel list into a single voxel
    Args:
        voxel_list: the list of voxels
    """
    if len(voxel_list) > 1:
        pred_voxel = np.concatenate(voxel_list[:-1], axis=0).reshape(-1, 2, 10, height, width)
    else:
        pred_voxel = None
    
    if voxel_list[-1].shape[0] > 1:
        temp = voxel_list[-1][:-1].reshape(-1, 2, 10, height, width)
        if pred_voxel is None:
            pred_voxel = temp
        else:
            pred_voxel = np.concatenate([pred_voxel, temp], axis=0)
    
    if mode != 0:
        temp = voxel_list[-1][-1][-mode:].reshape(-1, 2, 10, height, width)
    else:
        temp = voxel_list[-1][-1].reshape(-1, 2, 10, height, width)

    if pred_voxel is None:
        pred_voxel = temp
    else:
        pred_voxel = np.concatenate([pred_voxel, temp], axis=0)

    return pred_voxel
    
def write_event_frame_video(voxel_grid, ef_video_path, fps, ceil, upper_bound_percentile=98, keep_polarity=True):
    """
    Write the event frame video.
    Args:
        voxel_grid: the voxel grid to generate the event frames
        ef_video_path: the path to write the video
        fps: the FPS of the video
        ceil: the ceiling of the ef value
    """
    logger.info("Writing event frame video...")
    # Write all_pred_ef into a mp4 video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    B, P, L, H, W = voxel_grid.shape
    if keep_polarity:
        efs = np.sum(voxel_grid, axis=2) # [B, P(2), L, H, W]
        # Concatenate a zero tensor to the blue channel to make the channel number 3
        efs = np.concatenate([efs, np.zeros((B, 1, H, W))], axis=1)
    else:
        efs = np.sum(voxel_grid, axis=(1,2))[:,np.newaxis,...] # [B, P(2), L(10), H, W]
        efs = np.repeat(efs, 3, axis=1) # [B, P(10), H, W]
    # get the <u>% percentile of the ef value to set the upper bound
    efs_flatten = efs.flatten()
    efs_flatten = efs_flatten[efs_flatten > 0]
    efs_upper_bound = min(np.percentile(efs_flatten, upper_bound_percentile), ceil)
    logger.info(f'Upper bound of the event frame value during video writing: {efs_upper_bound}')
    # Clip the ef value to the upper bound
    efs = np.clip(efs, 0, efs_upper_bound) / efs_upper_bound
    # Move the Channel dimension to the last dimension
    efs = np.moveaxis(efs, 1, -1)
    print(efs.shape)
    
    video_size = (W, H)
    video = cv2.VideoWriter(ef_video_path, fourcc, fps, video_size)
    for i in range(efs.shape[0]):
        frame = efs[i]#/efs.max() 
        frame = (frame*255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)
    video.release()
    logger.info(f'Event frame video written to {ef_video_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--camera_index', type=int, default=0)
    parser.add_argument('--input_video_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='./weights/v2ce_3d.pt')
    parser.add_argument('--height', type=int, default=260)
    parser.add_argument('--width', type=int, default=346)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--ceil', type=int, default=10)
    parser.add_argument('--upper_bound_percentile', type=int, default=98)
    parser.add_argument('--interval', type=int, default=150)
    parser.add_argument('--vis_keep_polarity', type=SBool, default=True, nargs='?', const=True)
    parser.add_argument('-l', '--log_level', type=str, default='info')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger = logging.getLogger('V2CE')

    model = get_trained_mode(model_path=args.model_path, device=args.device)
    device = next(model.parameters()).device
    ldati = partial(sample_voxel_statistical, fps=args.fps, bidirectional=False, additional_events_strategy='slope')

    cap = cv2.VideoCapture(args.input_video_path if args.input_video_path else args.camera_index)
    if not cap.isOpened():
        print('Failed to open video source')
        sys.exit(1)

    prev_gray = None
    last_ts = 0
    events_list = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        now_ms = int(time.time() * 1000)
        if prev_gray is not None and (now_ms - last_ts >= args.interval):
            image_units = preprocess_pair(prev_gray, gray, height=args.height)
            pred_voxel = infer_center_image_unit(model, image_units, width=args.width)
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
        prev_gray = gray
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
