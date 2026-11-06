#!/usr/bin/env bash
# 本机示例：把 QAT .pt 一键转成 INT8 engine
# 使用前请改 WEIGHTS 路径；输出目录可按需修改。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$ROOT/scripts"

# ===== 改这里 =====
WEIGHTS="/home/jovan/weigth/phone_cig/phone_cig_11_det_n_640_1_qat_20261027.pt"
ONNX="/home/jovan/weigth/onnx/phone_cig_11_det_n_640_1_qat_20261027_dsyolo_b1.onnx"
ENGINE="/home/jovan/weigth/engine/phone_cig_11_det_n_640_1_qat_20261027_dsyolo_b1_int8.engine"
IMGSZ=640
BATCH=1
# ==================

echo "检查环境..."
cd /tmp
python3 "$SCRIPTS/check_env.py"

echo "开始转换..."
bash "$SCRIPTS/run_qat_to_engine.sh" \
  -w "$WEIGHTS" \
  -o "$ONNX" \
  -e "$ENGINE" \
  -s "$IMGSZ" \
  --batch "$BATCH" \
  --workspace 2
