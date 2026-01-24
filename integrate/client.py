import socket
import cv2
import numpy as np
import struct

def main():
    # 1. 创建 Socket 客户端
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('127.0.0.1', 9999)
    
    print(f"[INFO] 正在尝试连接集成服务器 {server_address}...")
    try:
        client_socket.connect(server_address)
        print("[INFO] 已连接到 Isaac Sim 视频服务器")
    except ConnectionRefusedError:
        print("[ERROR] 无法连接到服务器。请确保 integrate/server.py 正在运行！")
        return

    payload_size = struct.calcsize("L") # 4 字节长整型
    data = b""

    print("\n" + "="*50)
    print("集成 Python 客户端已启动")
    print("正在显示视频流窗口...")
    print("按 'q' 键退出显示")
    print("="*50 + "\n")

    try:
        while True:
            # A. 接收报头 (图像数据长度)
            while len(data) < payload_size:
                packet = client_socket.recv(4096)
                if not packet: break
                data += packet
            
            if not data: break
            
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]
            
            # B. 接收完整的图像数据
            while len(data) < msg_size:
                packet = client_socket.recv(4096)
                if not packet: break
                data += packet
            
            frame_data = data[:msg_size]
            data = data[msg_size:]
            
            # C. 解码 JPG 字节流
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if frame is not None:
                # 在窗口显示
                cv2.imshow("Isaac Sim - Go2 Integrated Vision", frame)
                
                # 按 q 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("[WARN] 帧解码失败")

    except Exception as e:
        print(f"[ERROR] 发生错误: {e}")
    finally:
        print("[INFO] 正在关闭连接...")
        client_socket.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
