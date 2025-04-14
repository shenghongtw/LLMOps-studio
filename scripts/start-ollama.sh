#!/bin/sh
# 启动 Ollama 服务
ollama serve &
# 等待服务启动
sleep 10
# 拉取qwen2.5:0.5b模型
ollama pull qwen2.5:0.5b
# 显示完成信息
echo '模型已拉取完成'
# 保持容器运行
tail -f /dev/null