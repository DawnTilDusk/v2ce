#!/bin/bash

# 1. 定义 Isaac Sim 安装路径
# 注意：这里我们使用绝对路径，避免相对路径带来的问题
ISAAC_SIM_PATH="/media/fishyu/6955024a-ed66-4a86-b94a-687c51c28306/fishyu/luoac/isaacsim"

# 检查 Isaac Sim 路径是否存在
if [ ! -d "$ISAAC_SIM_PATH" ]; then
    echo "Error: Isaac Sim directory not found at $ISAAC_SIM_PATH"
    exit 1
fi

# 2. 加载官方提供的环境配置脚本
# 这个脚本会设置 EXP_PATH, CARB_APP_PATH 并正确设置 PYTHONPATH
# 同时它会保留你当前 Conda 环境的优先级
echo "[INFO] Sourcing setup_conda_env.sh from Isaac Sim..."
source "$ISAAC_SIM_PATH/setup_conda_env.sh"

# 3. 运行你的代码
# "$@" 允许你向脚本传递参数
echo "[INFO] Running simple_go2_teleop.py..."
python simple_go2_teleop.py "$@"
