## 快速开始
```powershell
clone https://github.com/DawnTilDusk/v2ce.git
进入虚拟环境
pip install -r requirements.txt
```

### 然后可以直接用cpu跑
- 摄像头：
```powershell
python v2ce_online.py --device cpu --camera_index 0 --interval 150 --fps 30 --ceil 10 --upper_bound_percentile 98
```

- 视频：
```powershell
python v2ce_online.py --device cpu --input_video_path test.mp4 --interval 150 --fps 30 --ceil 10 --upper_bound_percentile 98
```

### 如果要用gpu
**则先替换pytorch为gpu版**
```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**测试是否能正常使用gpu**
```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```
返回类似以下即成功：
```plaintext
2.0.0+cu118
True
11.8
```

- 摄像头：
```powershell
python v2ce_online.py --device cuda --camera_index 0 --interval 150 --fps 30 --ceil 10 --upper_bound_percentile 98
```

- 视频：
```powershell
python v2ce_online.py --device cuda --input_video_path test.mp4 --interval 150 --fps 30 --ceil 10 --upper_bound_percentile 98
```

### 可能的问题：
1. 如果pip安装失败，可能是网络问题
- 如果使用代理请添加--proxy参数或者直接关掉
- 也可以改用清华源
```powershell
pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 可能会有warning请忽略
```

2. gpu出现与pytorch版本兼容的warning：
忽略即可，但是第一次加载可能需要较长时间，耐心等待

## 参数说明
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

## 进阶用法
- CPU 摄像头（更清晰显示负/正事件分离）：
  python v2ce_online.py --device cpu --camera_index 0 --interval 150 --fps 30 --ceil 10 --upper_bound_percentile 98 --vis_keep_polarity true -l debug
- CPU 视频（提高可视化亮度并关闭极性分离）：
  python v2ce_online.py --device cpu --input_video_path test.mp4 --interval 100 --fps 30 --ceil 15 --upper_bound_percentile 95 --vis_keep_polarity false
- GPU 摄像头（分辨率与中心裁剪调整）：
  python v2ce_online.py --device cuda --camera_index 0 --height 300 --width 320 --interval 120 --fps 60 --ceil 12 --upper_bound_percentile 97
- GPU 视频（更高帧率的事件生成）：
  python v2ce_online.py --device cuda --input_video_path test.mp4 --fps 60 --interval 100 --ceil 12 --upper_bound_percentile 97 --vis_keep_polarity true
- 日志等级：在调试时使用 -l debug 以查看细节；部署时建议 info 或 warning。

## 常见问题与建议
- 摄像头索引：Windows 常见索引为 0/1；若无法打开请检查权限或驱动，确认其他应用未占用摄像头。参考 [v2ce_online.py:L284-L288](file:///c:/Users/Notebook/Desktop/v2ce/v2ce_online.py#L284-L288)。
- 视频路径：推荐使用绝对路径或与 README 同级的相对路径；路径中如含空格请使用引号包裹。
- 首次加载耗时：GPU 首次运行可能需额外时间进行 CUDA 初始化与权重加载，耐心等待。
- 可视化不明显：降低 --upper_bound_percentile（如 95）或提高 --ceil（如 15），并尝试关闭 --vis_keep_polarity 以合并极性提升亮度。
- 帧率与间隔：提高 --fps 会使事件时间步更细；缩短 --interval 可提升更新频率但增大计算负载。
- 日志调试：设置 -l debug 可查看预处理与推理形状、序列分段信息（参见 video_to_voxels 等函数的 debug 输出）。
- 网络与镜像：若 pip 安装失败，可切换至清华镜像或关闭代理（已在主干说明中提供命令）。
