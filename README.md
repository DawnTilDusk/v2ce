clone https://github.com/DawnTilDusk/v2ce.git
进入虚拟环境
pip install -r requirements.txt

然后可以直接用cpu跑
摄像头：
python v2ce_online.py --device cpu --camera_index 0 --interval 150 --fps 30 --ceil 10 --upper_bound_percentile 98

视频：
python v2ce_online.py --device cpu --input_video_path test.mp4 --interval 150 --fps 30 --ceil 10 --upper_bound_percentile 98

如果要用gpu
则先替换pytorch为gpu版
pip uninstall -y torch torchvision torchaudio
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118

测试是否能正常使用gpu
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
返回类似：
2.0.0+cu118
True
11.8

摄像头：
python v2ce_online.py --device cuda --camera_index 0 --interval 150 --fps 30 --ceil 10 --upper_bound_percentile 98

视频：
python v2ce_online.py --device cuda --input_video_path test.mp4 --interval 150 --fps 30 --ceil 10 --upper_bound_percentile 98

可能的问题：
如果pip安装失败，可能是网络问题
如果使用代理请添加--proxy参数或者直接关掉
也可以改用清华源
pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
可能会有warning请忽略

gpu出现与pytorch版本兼容的warning：
忽略即可，但是第一次加载可能需要较长时间，耐心等待