#!/usr/bin/env python3
"""把带 Q/DQ 的 ONNX 编成 TensorRT INT8 引擎。

量化范围已经在 QuantizeLinear 节点里，本脚本不读校准图片。
生成的 engine 不带 Ultralytics metadata 头，DeepStream 可以直接加载。
输出布局须已是 DeepStream-Yolo 的 (batch, anchors, 6)。

示例:
    python3 onnx_qdq_to_engine.py \\
        -i /home/jovan/weigth/onnx/phone_cig_qat_v11n_640_dsyolo_b1.onnx \\
        -o /home/jovan/weigth/engine/phone_cig_qat_v11n_640_dsyolo_b1_int8.engine
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def prepare_cuda_env() -> None:
    """把 CUDA 库路径加进当前进程，便于随后加载 TensorRT。"""
    cuda_lib = "/usr/local/cuda/lib64"
    cuda_bin = "/usr/local/cuda/bin"
    if Path(cuda_lib).is_dir():
        os.environ["LD_LIBRARY_PATH"] = cuda_lib + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    if Path(cuda_bin).is_dir():
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")


def count_quantize_nodes(onnx_path: Path) -> tuple[int, tuple[int, ...]]:
    """统计 QuantizeLinear，并读出输入 shape。

    Args:
        onnx_path: 第 1 步导出的 ONNX。

    Returns:
        (QuantizeLinear 数量, 输入 shape)。静态维是正整数，动态维记为 -1。

    Raises:
        RuntimeError: 图中没有 QuantizeLinear，或没有输入。
    """
    import onnx

    graph = onnx.load(str(onnx_path), load_external_data=False)
    quantize_count = sum(1 for node in graph.graph.node if node.op_type == "QuantizeLinear")
    if quantize_count == 0:
        raise RuntimeError("ONNX 没有 QuantizeLinear。这个脚本只编已经带 QAT 范围的图。")
    if not graph.graph.input:
        raise RuntimeError("ONNX 没有输入")
    dims = []
    for dim in graph.graph.input[0].type.tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.dim_value > 0 else -1)
    return quantize_count, tuple(dims)


def build_engine(onnx_path: Path, engine_path: Path, workspace_gb: float) -> None:
    """用图内 Q/DQ 范围构建 INT8 engine。

    Args:
        onnx_path: 带 QuantizeLinear 的 ONNX。
        engine_path: 输出 .engine 路径。
        workspace_gb: TensorRT workspace，单位 GiB。

    Raises:
        FileNotFoundError: ONNX 不存在。
        RuntimeError: 缺少量化节点、输入不是静态 shape，或引擎无法反序列化。
    """
    if not onnx_path.is_file():
        raise FileNotFoundError(f"找不到 ONNX: {onnx_path}")

    quantize_count, shape = count_quantize_nodes(onnx_path)
    if len(shape) != 4 or any(dim <= 0 for dim in shape):
        raise RuntimeError(f"需要静态 NCHW 输入，当前 shape={shape}")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    # metadata=None：不写 Ultralytics 的 4 字节头，否则 DeepStream 无法反序列化
    # dataset=None：范围已在 Q/DQ 里，不再校准
    from ultralytics.utils.export.engine import onnx2engine

    print(f"QuantizeLinear: {quantize_count}")
    print(f"输入 shape: {shape}")
    onnx2engine(
        str(onnx_path),
        str(engine_path),
        workspace=workspace_gb,
        quantize=8,
        dynamic=False,
        shape=shape,
        dataset=None,
        metadata=None,
    )

    import tensorrt as trt

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"引擎无法反序列化: {engine_path}")
    print(f"引擎: {engine_path} ({engine_path.stat().st_size / 1024 / 1024:.2f} MB)")
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        print(f"  {name} {engine.get_tensor_mode(name)} {tuple(engine.get_tensor_shape(name))}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Q/DQ ONNX 转 TensorRT INT8 engine（不校准）")
    parser.add_argument("-i", "--onnx", required=True, type=Path, help="第 1 步导出的 .onnx")
    parser.add_argument("-o", "--engine", required=True, type=Path, help="输出 .engine 路径")
    parser.add_argument("--workspace", type=float, default=2.0, help="TensorRT workspace（GiB），默认 2")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    prepare_cuda_env()
    build_engine(args.onnx, args.engine, args.workspace)


if __name__ == "__main__":
    main()
