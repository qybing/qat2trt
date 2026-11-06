# qat2trt — YOLO11 QAT 权重转 DeepStream INT8

把**已经量化训练（QAT）好的** YOLO11 `.pt`，转成 DeepStream 能用的 **INT8 TensorRT 引擎**。

```text
QAT .pt  →  Q/DQ ONNX  →  INT8 .engine
         （无需校准图片）
```

适合：烟火 / 手机烟 / 人员等业务，只要训练时用了 `quantize=8`。

---

## 小白从哪看起？

1. 先看 **[快速开始.md](./快速开始.md)**（复制命令就能跑）  
2. 想搞懂每一步 → **[docs/流程说明.md](./docs/流程说明.md)**  
3. 懒得改命令 → 改 **[examples/命令示例.sh](./examples/命令示例.sh)** 里的路径再执行  

---

## 项目结构

```text
qat2trt/
├── README.md                 # 本文件：项目说明 + 环境
├── 快速开始.md               # 最短操作步骤
├── docs/
│   └── 流程说明.md           # 流程图、每个脚本的输入/输出
├── examples/
│   └── 命令示例.sh           # 改路径即可跑的示例
└── scripts/
    ├── check_env.py                 # 检查环境
    ├── run_qat_to_engine.sh         # 一键：.pt → onnx → engine
    ├── export_yolo11_qat_onnx.py    # 第1步：.pt → Q/DQ ONNX
    ├── onnx_qdq_to_engine.py        # 第2步：ONNX → INT8 engine
    └── export_yolo11.py             # 依赖（DeepStream 输出格式）
```

---

## 你需要准备什么？

### 1）硬件 / 系统

| 项目 | 要求 | 说明 |
|------|------|------|
| 设备 | NVIDIA Jetson | 建 `.engine` 必须在目标板上 |
| JetPack | 与线上一致 | 本仓库验证环境：JP 6.2 / L4T R36.4 |
| CUDA / TensorRT | 随 JetPack | 验证：CUDA 12.6 + TensorRT 10.7 |
| Python | 3.10（JP6） | |

> `.engine` **不能**随便拷到另一块不同 GPU / 不同 TRT 版本的板子上用。

### 2）Python 包（转换用）

| 包 | 干什么 |
|----|--------|
| PyTorch（**Jetson 版**） | 加载 `.pt` |
| nvidia-modelopt | 恢复 QAT 假量化器 |
| ultralytics | YOLO11 + QAT 支持 |
| onnx | 读写带 QuantizeLinear 的图 |
| tensorrt（Python） | 编 engine |

检查是否装好：

```bash
cd /tmp
python3 /home/jovan/code/qat2trt/scripts/check_env.py
```

### 3）输入文件

- 一份 **QAT** 训练得到的 `.pt`（训练参数里有 `quantize=8`）  
- **不要**拿普通 FP32 的 `best.pt` 硬跑本项目（会报没有假量化器）

### 4）输出目录（可自己指定）

常用默认：

| 类型 | 建议目录 |
|------|----------|
| ONNX | `/home/jovan/weigth/onnx/` |
| ENGINE | `/home/jovan/weigth/engine/` |

用 `-o` / `-e` 可以改成任意路径。

---

## 一键转换（最常用）

```bash
bash /home/jovan/code/qat2trt/scripts/run_qat_to_engine.sh \
  -w /你的路径/xxx_qat.pt \
  -o /home/jovan/weigth/onnx/xxx_qat_v11n_640_dsyolo_b1.onnx \
  -e /home/jovan/weigth/engine/xxx_qat_v11n_640_dsyolo_b1_int8.engine \
  -s 640 \
  --batch 1
```

### 输入 → 输出

| | 路径 / 内容 |
|--|-------------|
| **输入** | `-w`：QAT `.pt` |
| **中间输出** | `-o`：带 `QuantizeLinear` 的 ONNX；旁路 `*_labels.txt` |
| **最终输出** | `-e`：INT8 `.engine`（给 DeepStream） |

成功时终端会出现 `QuantizeLinear: N`（N 应大于 0）。

---

## 分步转换（想看中间文件时）

```bash
cd /tmp
SCRIPTS=/home/jovan/code/qat2trt/scripts

# ① .pt → ONNX
python3 "$SCRIPTS/export_yolo11_qat_onnx.py" \
  -w /你的路径/xxx_qat.pt \
  -o /home/jovan/weigth/onnx/xxx_qat_dsyolo_b1.onnx \
  -s 640 --batch 1

# ② ONNX → engine（不校准）
python3 "$SCRIPTS/onnx_qdq_to_engine.py" \
  -i /home/jovan/weigth/onnx/xxx_qat_dsyolo_b1.onnx \
  -o /home/jovan/weigth/engine/xxx_qat_dsyolo_b1_int8.engine \
  --workspace 2
```

各脚本详细「输入/输出」见 [docs/流程说明.md](./docs/流程说明.md)。

---

## 重要提醒

1. **始终建议 `cd /tmp` 再跑 Python**，避免在 `/home/jovan/code` 下 import 出错。一键脚本已自动 `cd /tmp`。  
2. `-s`（如 640）必须和训练/推理尺寸一致。  
3. `--batch` 必须和 DeepStream 配置里的 `batch-size` 一致。  
4. 本项目 **只要 QAT**；普通权重做 INT8（需要校准图）请走别的工具（如 `pt2trt`），不要混用。  
5. DeepStream 还需要 YOLO 自定义解析库（`NvDsInferParseYolo`），配置示例见流程说明。

---

## 本机已跑通过的例子（参考）

| 业务 | 输入 `.pt` 示例 | 输出 engine 示例 |
|------|-----------------|------------------|
| 烟火 | `fire_smoke_aqt_...qat....pt` | `.../fire_smoke_qat_v11n_640_dsyolo_b1_int8.engine` |
| 手机/烟 | `phone_cig_..._qat_20261027.pt` | `.../phone_cig_11_det_n_640_1_qat_20261027_dsyolo_b1_int8.engine` |

---

## 和 Cursor Skill 的关系

Cursor 技能 `yolo11-qat-deepstream` 与本项目脚本同源。  
**给人看、给人跑：用本仓库 `qat2trt`。**  
对话里也可以说「按 qat2trt 流程转换」。
