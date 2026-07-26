# OvisOCR2-ROCm

**OvisOCR2 跑在 AMD Radeon 上 —— 首个登顶 OmniDocBench v1.6 的端到端模型，现已原生运行于 ROCm。**

[OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2)（ATH-MaaS / 阿里，Apache-2.0，0.8B 参数）是一个紧凑的端到端页面解析模型：输入一页文档图像，它一次性输出结构化 Markdown——文本、表格、公式、阅读顺序。它在 [OmniDocBench v1.6](https://arxiv.org/abs/2607.13639) 上取得 **综合得分 96.58** 的新纪录，是首个超越此前由流水线方法主导的排行榜的端到端模型。本仓库将其运行于 **AMD ROCm 上的 vLLM**（gfx1100 / Radeon PRO W7900），全量复现综合得分 **95.88** ——**专区内排名第一**（4 项指标中 3 项基本完美；公式 CDM 与论文 96.58 的小差距是已查明的**模型固有**分段差异，与 vLLM 版本无关，详见 `docs/known-gaps.md`）。

- **模型：** `ovisocr2` v1.0 —— Apache-2.0，无商用限制
- **后端：** vLLM 0.22.1（ROCm），进程内加载 —— 上游模型卡指定的版本
- **平台：** `linux-rocm`（community）· `windows-hip`（community-wanted）
- **所属专区：** [OmniDocBench-ROCm](https://github.com/AIwork4me/OmniDocBench-ROCm)

> **架构说明。** 虽名为 "Ovis"，但 `config.json` 声明 `model_type: qwen3_5` / `Qwen3_5ForConditionalGeneration`——即 Qwen3-VL 视觉编码器 + **Qwen3-Next GDN（门控增量网络）混合**文本骨干。vLLM 路由到其原生 `qwen3_5` 实现；GDN 线性注意力层在 ROCm 上经 Triton/FLA 预填充内核运行。

## 对比 —— OmniDocBench v1.6（linux-rocm）

| 模型 | 参数量 | 后端 | 综合得分 | 徽章 |
|---|---|---|---|---|
| **OvisOCR2（本仓库）** | **0.8B** | **vLLM/ROCm** | **95.88** | community |
| PaddleOCR-VL-1.6 | 0.9B | llama.cpp/HIP | 95.77 | community |
| MinerU2.5 | 1.2B | vLLM/ROCm | 95.56 | community |
| HunyuanOCR | 1B | vLLM/ROCm | 93.64 | community |

OvisOCR2 是本专区**参数量最小**、**得分最高**的模型，也是首个领跑榜单的*端到端*解析器。论文参考：综合 96.58，文本编辑距离 0.025，公式 CDM 97.5，表格 TEDS 94.8，阅读顺序 0.111（arXiv 2607.13639，表 2）。已提交的实测值见 [`model_card.json`](model_card.json)。

## 安装（Install）

需要一个支持 qwen3_5 的 ROCm vLLM。参考构建由
[`rocm-vllm-installer`](https://github.com/AIwork4me/rocm-vllm-installer) 产出
（构建 vLLM v0.22.1 + ROCm 补丁，为 gfx110X-all 构建，torch 2.10+rocm7.12）。

```bash
# 1. ROCm vLLM 0.22.1 虚拟环境（一次性，构建约 1-2 小时）。请本地克隆安装器
#    以获得其 patches/ 目录（不要用 curl|bash）：
git clone --branch v1.0.0 https://github.com/AIwork4me/rocm-vllm-installer.git
cd rocm-vllm-installer && VENV=/root/venvs/vllm-0221b VLLM_VERSION=v0.22.1 bash install.sh
# 验证 qwen3_5 已注册：
/root/venvs/vllm-0221b/bin/python -c "from vllm.model_executor.models.registry import ModelRegistry as m; \
  print('Qwen3_5ForConditionalGeneration' in m.get_supported_archs())"   # -> True

# 2. 引擎（omnidocbench-rocm，GitHub 同源项目，不在 PyPI；装入上述 venv）+ 本仓库：
/root/venvs/vllm-0221b/bin/pip install "omnidocbench-rocm @ git+https://github.com/AIwork4me/omnidocbench-rocm.git@v0.3.2"
pip install -e ".[dev]"        # 本仓库（用于测试 / conformance）

# 3. 权重（HF 或 ModelScope，二者一致；国内推荐 ModelScope）：
python -c "from modelscope import snapshot_download; \
  snapshot_download('ATH-MaaS/OvisOCR2', local_dir='/root/models/OvisOCR2')"
export OVISOCR2_WEIGHTS=/root/models/OvisOCR2
```

GPU：任一显存 ≥ 16 GB 的 gfx1100（0.8B 模型在 32k 上下文下峰值约 6 GB）。验证：`rocminfo | grep gfx1100`。

## 演示（Demo）

`smoke` 后端无需 GPU，会为每张图片写出占位 `.md`，便于在 CI 中端到端验证契约：

```bash
bash examples/run_demo.sh
```

用真实模型解析一页：

```bash
export HIP_VISIBLE_DEVICES=0
mkdir -p /tmp/in /tmp/out && cp examples/demo.png /tmp/in/
python adapter/run_adapter.py --img-dir /tmp/in --out-dir /tmp/out \
  --platform linux-rocm --backend vllm
cat /tmp/out/*.md
```

## 评测（Evaluation）

全量 OmniDocBench v1.6（1651 页），Edit_dist + TEDS + CDM：

```bash
export DATASET=/root/datasets/OmniDocBench_data
export OMNIDOCBENCH_CHECKOUT=/path/to/OmniDocBench   # 锁定于 2b161d0
make eval-linux        # = omnidocbench-rocm run --stage all ...（推理 + 打分 + 发布）
```

也可分步执行（推理 → 打分 → 发布），见 [`reproduce.md`](reproduce.md)。评测配置：[`eval/configs/omnidocbench_v16.yaml`](eval/configs/omnidocbench_v16.yaml)。

## 可复现性（Reproducibility）

- **硬件：** AMD gfx1100（Radeon PRO W7900，48 GB）× 4；单卡即可运行。
- **ROCm 驱动：** 7.2（torch 2.10.0+rocm7.12）。
- **后端：** vLLM 0.22.1 ROCm（`vllm-0221b` 虚拟环境），进程内加载，`gdn_prefill_backend='triton'`。
- **配方：** 官方 OvisOCR2 卡 —— 贪心解码（temp=0）、`max_tokens=16384`、像素 448²–2880²、`_clean_truncated_repeats`、过滤视觉区域标签。
- **权重：** `ATH-MaaS/OvisOCR2` —— 版本与 sha256 见 [`REPRO.yaml`](REPRO.yaml)。
- 结果与溯源：[`results/omnidocbench/v16/linux-rocm/`](results/omnidocbench/v16/linux-rocm/)。详见 [`docs/reproducibility.md`](docs/reproducibility.md)。

## 已知限制（Known Gaps）

- **`windows-hip`：** `community-wanted`。OvisOCR2 的 Qwen3-Next GDN 架构暂无 GGUF/HIP-SDK 服务路径，Windows 暂缓。
- **公式 CDM 差距（模型固有）：** 综合 95.88 vs 论文 96.58；0.70 分差距全在公式 CDM（95.41 vs 97.53）。系统化调试验证这是**模型固有**的公式分段差异（约 22/2352 个公式，中位 CDM 1.0）——**与版本无关**（本仓库运行模型卡指定的 vLLM 0.22.1；全量 1651 页上 0.22.1 与 0.19.0 的 CDM 完全一致：均为 95.41；0.19.0 综合 95.87、0.22.1 综合 95.88）。无法通过 recipe 或版本修复；详见 [`docs/known-gaps.md`](docs/known-gaps.md)。
- **吞吐：** eager 模式 + ROCm 分页注意力回退，吞吐中等；单卡全量**实测**约 1 小时（双卡分片约 30 分钟）——人工观测值，非 CI 测得；已发布产物中不记录逐页延迟。非 eager/cudagraph 调优为后续工作。
- **`verified` 等级：** 暂无 —— 需维护者 Docker 复现（开发环境无 Docker）。本条目为 `community`（自证、CI 校验）；见 [`docs/known-gaps.md`](docs/known-gaps.md)。

## 许可证

Apache-2.0（权重与代码）。见 [`LICENSE`](LICENSE)。
