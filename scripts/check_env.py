#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 QAT → INT8 转换所需环境是否齐全。

在 /tmp 下运行，避免本地目录干扰 ultralytics 导入：
  cd /tmp && python3 /home/jovan/code/qat2trt/scripts/check_env.py
"""

from __future__ import annotations


def main() -> int:
    """逐项检查依赖并打印结果。返回 0 表示全部通过。"""
    ok = True
    print("=== QAT2TRT 环境检查 ===\n")

    checks = [
        ("torch", "PyTorch（Jetson 版）"),
        ("modelopt", "nvidia-modelopt（恢复 QAT 假量化器）"),
        ("ultralytics", "ultralytics"),
        ("onnx", "onnx"),
        ("tensorrt", "TensorRT Python 绑定"),
    ]
    for mod, desc in checks:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"[OK] {desc}: {mod}=={ver}")
        except Exception as e:
            ok = False
            print(f"[缺] {desc}: 无法 import {mod} ({e.__class__.__name__})")

    try:
        import torch

        print(f"\nCUDA 可用: {torch.cuda.is_available()}")
        print(f"torch.version.cuda: {torch.version.cuda}")
    except Exception:
        pass

    print("\n说明:")
    print("- 本流程是 QAT，不需要校准图片")
    print("- 建 engine 必须在目标 Jetson 上执行（与 GPU/TRT 版本绑定）")
    print("- 建议: cd /tmp 后再跑本项目的导出脚本")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
