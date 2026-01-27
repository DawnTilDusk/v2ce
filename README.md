## 快速开始(这个分支服务于ubuntu)
```shell
# ssh：
ssh fishyu@162.105.195.38
# 启动vnc
x11vnc -display :0 -forever -shared -rfbport 5900
# 克隆仓库
clone https://github.com/DawnTilDusk/v2ce.git
# 进入虚拟环境
conda create -n v2ce python=3.10
conda activate v2ce
# 安装依赖
pip install -r requirements.txt
```

### gpu的启用
**测试是否能正常使用gpu**
```shell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```
返回类似以下即成功：
```shell
2.0.0+cu118
True
11.8
```

**或者使用：**
```shell
nvidia-smi
```
### 集成于一个主脚本的事件相机模拟测试
```shell
export DISPLAY=:0 && $ISAACSIM_PYTHON_EXE robot_test/go2_v2ce_sync.py
```

#### 架构说明：
1. 同步机制 ：
   
   - 架构中最关键的是 推理(L) 与 物理步进(Q) 的串行关系。
   - 代码显式等待 V2CE 模型推理完成生成 Event Frame 后，才调用 world.step() 推进物理时间。这确保了无论推理耗时多久，仿真世界的时间流逝始终是精确的 1/30 秒，避免了“慢动作”或“跳帧”现象。
2. 模块分工 ：
   
   - World/Robot : 负责物理仿真环境和机器人本体。
   - V2CEPredictor : 负责视觉算法，将普通 RGB 图像转换为事件相机风格的特征图。
   - DynamicObstacleManager : 独立管理环境中的动态物体，与机器人解耦。
   - Visualization : 使用 Matplotlib 实时展示算法效果（因为 Isaac Sim 的 OpenCV GUI 受限）。

```Mermaid
graph TD
    %% 初始化阶段
    subgraph Init [初始化阶段]
        A[Start] --> B[初始化 SimulationApp]
        B --> C[构建仿真世界 World]
        C --> D[加载 Unitree Go2 机器人]
        C --> E[初始化相机传感器 Camera]
        D --> F[初始化动态障碍物管理器 DynamicObstacleManager]
        E --> G[初始化 V2CE 推理器 V2CEPredictor]
        G --> H[初始化 Matplotlib 可视化窗口]
    end

    %% 主循环阶段
    subgraph Loop [同步仿真主循环]
        H --> I{仿真运行中?}
        
        %% 1. 感知
        I -- Yes --> J[获取相机图像 Get RGB]
        J --> K[图像预处理 (RGB -> BGR)]
        
        %% 2. 推理 (核心同步点)
        K --> L[V2CE 模型推理 Predict]
        L -- 生成事件帧 --> M[Event Frame]
        
        %% 3. 可视化
        M --> N[更新 Matplotlib 显示]
        
        %% 4. 控制与环境
        N --> O[读取键盘输入 & 更新机器人位姿]
        O --> P[更新动态障碍物状态 Update Obstacles]
        
        %% 5. 物理步进
        P --> Q[物理世界步进 World Step]
        Q -->|时间推进 1/30s| I
    end

    I -- No --> R[释放资源 & 退出]

    %% 样式定义
    style L fill:#f9f,stroke:#333,stroke-width:2px
    style Q fill:#bbf,stroke:#333,stroke-width:2px
```






---

## archive里面的一些文件的启动方式(有一些仍然是只兼容windows)
- 摄像头转事件相机：
```powershell
export DISPLAY=:0 && python v2ce_online.py --device cuda --camera_index 0 --interval 1 --fps 30 --ceil 10 --upper_bound_percentile 98
```

- 视频转事件相机：
```powershell
export DISPLAY=:0 && python v2ce_online.py --device cuda --input_video_path test.mp4 --interval 1 --fps 30 --ceil 10 --upper_bound_percentile 98

export DISPLAY=:0 && python v2ce_online-v3.py --device cuda --input_video_path test.mp4 --interval 3 --fps 30
```
- 服务端与客户端解耦运行可视化模拟事件相机：
```powershell
cd C:\isaac-sim\
.\python.bat C:\Users\JerryY1\Desktop\v2ce\integrate\server-v2.py

cd C:\Users\JerryY1\Desktop\v2ce
python integrate/client.py 
```

#### 参数说明
- --device [cpu|cuda]，默认 cpu：推理设备，需与已安装的 PyTorch 版本匹配。参考 [v2ce_online.py:L263](file:///c:/Users/Notebook/Desktop/v2ce/v2ce_online.py#L263) 与 [get_trained_mode](file:///c:/Users/Notebook/Desktop/v2ce/v2ce_online.py#L29-L34)。
- --camera_index [int]，默认 0：摄像头索引；当不提供 --input_video_path 时使用该索引打开视频源。参考 [v2ce_online.py:L264-L285](file:///c:/Users/Notebook/Desktop/v2ce/v2ce_online.py#L264-L285)。
- --input_video_path [path]，默认空：输入视频路径；与 --camera_index 二选一。参考 [v2ce_online.py:L265-L285](file:///c:/Users/Notebook/Desktop/v2ce/v2ce_online.py#L265-L285)。
- --model_path [path]，默认 ./weights/v2ce_3d.pt：模型权重路径。参考 [v2ce_online.py:L266,L280](file:///c:/Users/Notebook/Desktop/v2ce/v2ce_online.py#L266)。
- --height [int]，默认 260：预处理统一高度（按高缩放保持宽高比），影响体素尺寸。参考 [preprocess_pair](file:///c:/Users/Notebook/Desktop/v2ce/v2ce_online.py#L36-L46)、[v2ce_online.py:L300-L311](file:///c:/Users/Notebook/Desktop/v2ce/v2ce_online.py#L300-L311)。
- --width [int]，默认 346：中心裁剪推理宽度，仅在 center 推理路径使用。参考 [infer_center_image_unit](file:///c:/Users/Notebook/Desktop/v2ce/v2ce_online.py#L49-L55)、[v2ce_online.py:L301](file:///c:/Users/Notebook/Desktop/v2ce/v2ce_online.py#L301)。
- --fps [int]，默认 30：事件时间步相关参数，影响事件帧生成与采样。参考 [v2ce_online.py:L269,L282,L308-L309](file:///c:/Users/Notebook/Desktop/v2ce/v2ce_online.py#L269)。
- --ceil [int]，默认 10：事件帧像素值裁剪上限，控制可视化对比度。参考 [make_event_frame](file:///c:/Users/Notebook/Desktop/v2ce/v2ce_online.py#L57-L108)、[write_event_frame_video](file:///c:/Users/Notebook/Desktop/v2ce/v2ce_online.py#L220-L260)。
- --upper_bound_percentile [int]，默认 98：可视化归一化的百分位上界，抑制异常高值。参考 [v2ce_online.py:L241-L247,L308-L309](file:///c:/Users/Notebook/Desktop/v2ce/v2ce_online.py#L241-L247)。
- --interval [ms]，默认 150：两帧配对处理的时间间隔，控制在线推理频率。参考 [v2ce_online.py:L298-L311](file:///c:/Users/Notebook/Desktop/v2ce/v2ce_online.py#L298-L311)。
- --vis_keep_polarity [bool]，默认 True：事件帧可视化时是否保留极性分离（正/负）。参考 [v2ce_online.py:L57-L108,L273,L308-L309](file:///c:/Users/Notebook/Desktop/v2ce/v2ce_online.py#L57-L108)。
- -l/--log_level [debug|info|warning|error]，默认 info：日志等级。参考 [v2ce_online.py:L274-L279](file:///c:/Users/Notebook/Desktop/v2ce/v2ce_online.py#L274-L279)。

#### 进阶用法
- CPU 摄像头（更清晰显示负/正事件分离）：
  python v2ce_online.py --device cpu --camera_index 0 --interval 150 --fps 30 --ceil 10 --upper_bound_percentile 98 --vis_keep_polarity true -l debug
- CPU 视频（提高可视化亮度并关闭极性分离）：
  python v2ce_online.py --device cpu --input_video_path test.mp4 --interval 100 --fps 30 --ceil 15 --upper_bound_percentile 95 --vis_keep_polarity false
- GPU 摄像头（分辨率与中心裁剪调整）：
  python v2ce_online.py --device cuda --camera_index 0 --height 300 --width 320 --interval 120 --fps 60 --ceil 12 --upper_bound_percentile 97
- GPU 视频（更高帧率的事件生成）：
  python v2ce_online.py --device cuda --input_video_path test.mp4 --fps 60 --interval 100 --ceil 12 --upper_bound_percentile 97 --vis_keep_polarity true
- 日志等级：在调试时使用 -l debug 以查看细节；部署时建议 info 或 warning。