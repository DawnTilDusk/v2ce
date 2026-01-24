import socket
import struct
import cv2
import numpy as np
import torch
import argparse
import logging
import time
import os.path as op
import sys
from functools import partial
from v2ce_online import preprocess_pair, infer_center_image_unit, make_event_frame, get_trained_mode, SBool
import v2ce_online as v2

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--socket_host", type=str, default="127.0.0.1")
    parser.add_argument("--socket_port", type=int, default=9999)
    parser.add_argument("--model_path", type=str, default="./weights/v2ce_3d.pt")
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--width", type=int, default=346)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--ceil", type=int, default=10)
    parser.add_argument("--upper_bound_percentile", type=int, default=98)
    parser.add_argument("--interval", type=int, default=150)
    parser.add_argument("--vis_keep_polarity", type=SBool, default=True, nargs="?", const=True)
    parser.add_argument("-l", "--log_level", type=str, default="info")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger = logging.getLogger("V2CE-STREAM")

    try:
        reader = SocketFrameReader(args.socket_host, args.socket_port)
    except Exception as e:
        print("Failed to connect socket:", e)
        sys.exit(1)

    model = get_trained_mode(model_path=args.model_path, device=args.device)
    device = next(model.parameters()).device
    ldati = partial(v2.sample_voxel_statistical, fps=args.fps, bidirectional=False, additional_events_strategy="slope")

    prev_gray = None
    last_ts = 0
    events_list = None

    try:
        while True:
            frame = reader.read_frame()
            if frame is None:
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
                cv2.imshow("EventFrame", ef_frame)
                cv2.imshow("RGBFrame", frame)
                last_ts = now_ms
            prev_gray = gray
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        reader.close()
        cv2.destroyAllWindows()
