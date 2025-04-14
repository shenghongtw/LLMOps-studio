#!/bin/sh

# 等待 MinIO 服务启动
echo "等待 MinIO 服务启动..."
sleep 10

# 配置 MinIO 客户端
mc config host add myminio http://minio:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD

# 创建 bucket（如果不存在）
mc mb myminio/mlflow-artifacts --ignore-existing

echo "MinIO 初始化完成，已创建 mlflow-artifacts bucket" 