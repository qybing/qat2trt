#!/usr/bin/env bash
# YOLO11 QAT .pt → Q/DQ ONNX → TensorRT INT8 engine（无需校准图）
#
# 用法:
#   ./run_qat_to_engine.sh \
#     -w /path/to/qat_best.pt \
#     -o /home/jovan/weigth/onnx/foo_qat.onnx \
#     [-e /home/jovan/weigth/engine/foo_qat_int8.engine] \
#     [-s 640] [--batch 1] [--workspace 2]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS=""
ONNX=""
ENGINE=""
IMGSZ=640
BATCH=1
WORKSPACE=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    -w|--weights) WEIGHTS="$2"; shift 2 ;;
    -o|--onnx) ONNX="$2"; shift 2 ;;
    -e|--engine) ENGINE="$2"; shift 2 ;;
    -s|--imgsz) IMGSZ="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --workspace) WORKSPACE="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

if [[ -z "$WEIGHTS" || -z "$ONNX" ]]; then
  echo "必须提供 -w/--weights 与 -o/--onnx"
  exit 1
fi

if [[ ! -f "$WEIGHTS" ]]; then
  echo "找不到权重文件: $WEIGHTS"
  exit 1
fi

if [[ -z "$ENGINE" ]]; then
  base="$(basename "$ONNX" .onnx)"
  ENGINE="/home/jovan/weigth/engine/${base}_int8.engine"
fi

mkdir -p "$(dirname "$ONNX")" "$(dirname "$ENGINE")"

echo "[0/2] 建议先检查环境: cd /tmp && python3 $SCRIPT_DIR/check_env.py"
echo "[1/2] 导出 Q/DQ ONNX ..."
# 在 /tmp 跑，避免 /home/jovan/code 下目录名干扰 import ultralytics
cd /tmp
python3 "$SCRIPT_DIR/export_yolo11_qat_onnx.py" \
  -w "$WEIGHTS" \
  -o "$ONNX" \
  -s "$IMGSZ" \
  --batch "$BATCH"

echo "[2/2] 编译 INT8 engine（不校准）..."
python3 "$SCRIPT_DIR/onnx_qdq_to_engine.py" \
  -i "$ONNX" \
  -o "$ENGINE" \
  --workspace "$WORKSPACE"

echo ""
echo "完成（无需校准图）:"
echo "  输入 PT:  $WEIGHTS"
echo "  输出 ONNX:   $ONNX"
echo "  输出 ENGINE: $ENGINE"
echo "  标签文件:    ${ONNX%.onnx}_labels.txt"
