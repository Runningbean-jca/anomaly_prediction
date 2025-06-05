#!/bin/bash

# 设置使用的 GPU 设备（可根据需要修改）
export CUDA_VISIBLE_DEVICES=0

# === 训练阶段 ===
python main.py \
  --anormly_ratio 1 \
  --num_epochs 3 \
  --batch_size 256 \
  --mode train \
  --dataset TUBE \
  --data_path data \
  --input_c 55 \
  --output_c 55

# === 测试阶段 ===
python main.py \
  --anormly_ratio 1 \
  --num_epochs 10 \
  --batch_size 256 \
  --mode test \
  --dataset TUBE \
  --data_path data \
  --input_c 55 \
  --output_c 55 \
  --pretrained_model 3
