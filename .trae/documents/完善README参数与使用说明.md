**整体目标**
- 扩充 README，在“快速开始”之后新增“参数说明”“进阶用法”“常见问题与建议”三部分，覆盖所有 CLI 参数的含义、默认值、推荐设置及示例命令。

**改动内容**
- 参数说明：逐项列出并解释以下参数（含类型/默认值/影响范围/代码引用）：
  - --device、--camera_index、--input_video_path、--model_path、--height、--width、--fps、--ceil、--upper_bound_percentile、--interval、--vis_keep_polarity、-l/--log_level。
- 进阶用法：
  - 摄像头/视频在 CPU 与 GPU 的完整示例；
  - 调整分辨率（height/width）与帧率（fps）的效果与建议；
  - 可视化调参（ceil/upper_bound_percentile/vis_keep_polarity）的示例对比；
  - 日志等级（log_level）在调试/性能监控中的使用。
- 常见问题与建议：
  - pip 安装失败的网络/镜像建议；
  - GPU 版本核对与首次加载较慢的提示；
  - 摄像头索引/权限问题、视频路径格式注意事项；
  - 大模型权重文件的体积与下载耗时提示。

**呈现方式**
- 使用清晰的标题与要点式说明，示例命令使用代码块展示；
- 对关键参数附加代码位置引用，便于读者进一步查阅实现细节；
- 保持中文术语统一与简洁表达，避免冗长表格。