#!/usr/bin/env python3
"""把 YOLO11 QAT 权重量化范围导出成 DeepStream-Yolo ONNX。

加载 .pt 时会恢复 ModelOpt 假量化器。导出图与 export_yolo11.py 相同：
输出为 (batch, 8400, 6)，依次是 x1, y1, x2, y2, score, class_id，
供 libnvdsinfer_custom_impl_Yolo.so 的 NvDsInferParseYolo 解析。
假量化器会写成 QuantizeLinear / DequantizeLinear，scale 来自 QAT，不再校准。

示例:
    python3 export_yolo11_qat_onnx.py \\
        -w runs/detect/person_stage3_qat_20261018_v11n-7/weights/best.pt \\
        -o /home/jovan/weigth/onnx/phone_cig_qat_v11n_640_dsyolo_b1.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# 保证从任意工作目录运行时都能找到同目录的 export_yolo11.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_yolo11 import DeepStreamOutput, yolo11_export  # noqa: E402
from ultralytics.utils.export.engine import torch2onnx  # noqa: E402
from ultralytics.utils.torch_utils import is_qat  # noqa: E402


def export_qat_onnx(
    weights: Path,
    onnx_path: Path,
    imgsz: int,
    batch: int,
    opset: int,
) -> None:
    """导出带 Q/DQ 的 DeepStream-Yolo ONNX。

    Args:
        weights: QAT 训练得到的 .pt 路径。
        onnx_path: 输出 ONNX 路径。同目录会写一份 labels.txt。
        imgsz: 网络输入边长。
        batch: 静态 batch，与后续 TensorRT 引擎一致。
        opset: ONNX opset 版本。

    Raises:
        FileNotFoundError: 权重不存在。
        RuntimeError: 权重里没有 QAT 量化器，或导出后没有 QuantizeLinear。
    """
    if not weights.is_file():
        raise FileNotFoundError(f"找不到权重: {weights}")

    # YOLO() 会按 checkpoint 里的 modelopt 状态插回假量化器
    model = yolo11_export(str(weights), torch.device("cpu"), fuse=True)
    if not is_qat(model):
        raise RuntimeError(
            "该权重没有 QAT 假量化器。普通 FP32 权重要走 DeepStream-Yolo 的校准导出，不要用本脚本。"
        )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path = onnx_path.with_name(onnx_path.stem + "_labels.txt")
    names = list(model.names.values())
    labels_path.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")

    wrapped = nn.Sequential(model, DeepStreamOutput()).eval()
    dummy = torch.zeros(batch, 3, imgsz, imgsz)
    torch2onnx(
        wrapped,
        dummy,
        str(onnx_path),
        opset=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic=None,
    )

    import onnx

    graph = onnx.load(str(onnx_path))
    quantize_count = sum(1 for node in graph.graph.node if node.op_type == "QuantizeLinear")
    if quantize_count == 0:
        raise RuntimeError("ONNX 里没有 QuantizeLinear，QAT 范围没有写进图。")

    print(f"类别: {names}")
    print(f"QuantizeLinear: {quantize_count}")
    print(f"输入: input {[batch, 3, imgsz, imgsz]}")
    print(f"输出: output (batch, 8400, 6)，供 NvDsInferParseYolo")
    print(f"ONNX: {onnx_path}")
    print(f"标签: {labels_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="YOLO11 QAT .pt 导出为 DeepStream-Yolo Q/DQ ONNX")
    parser.add_argument("-w", "--weights", required=True, type=Path, help="QAT best.pt")
    parser.add_argument("-o", "--onnx", required=True, type=Path, help="输出 .onnx 路径")
    parser.add_argument("-s", "--imgsz", type=int, default=640, help="输入边长，默认 640")
    parser.add_argument("--batch", type=int, default=1, help="静态 batch，默认 1")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset，默认 17")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    if args.batch < 1 or args.imgsz < 32:
        raise SystemExit("需要 batch>=1 且 imgsz>=32")
    export_qat_onnx(args.weights, args.onnx, args.imgsz, args.batch, args.opset)


if __name__ == "__main__":
    main()
