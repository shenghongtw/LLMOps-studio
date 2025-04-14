#!/bin/sh
# 启动 Ollama 服务
ollama serve &
# 等待服务启动
sleep 10
# 切换到模型目录
cd /models/shared
# 创建模型
ollama create qwen2.5-0.5b-instruct -f Modelfile
# 显示完成信息
echo '模型已创建'
# 保持容器运行
tail -f /dev/null