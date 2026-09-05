# Novel LLM：小说写作专用模型

面向 RTX 3060 6GB、Windows、Conda Python 3.12 的中文小说写作助手项目。

## 推荐路线

```text
Qwen2.5-1.5B-Instruct + QLoRA 4-bit/FP16 + RAG 长期记忆
```

LoRA 学习题材、文风和输出格式；RAG 与结构化记忆负责人物、时间线、伏笔和长篇一致性。不做从零预训练或 7B 全参数微调。

## 结构

```text
novel_llm/
├── configs/qlora.yaml
├── data/
├── docs/IMPLEMENTATION_PLAN.md
├── novel_memory/
├── scripts/
├── app/
├── outputs/
├── requirements.txt
└── main.py
```

## 快速开始

## 开发环境（固定）

本项目统一使用 Conda 的 Python 3.12 环境，不使用 base 环境：

```text
Conda 本体：D:\miniconda3
环境名称：python312
环境路径：D:\conda_envs\python312
Python：3.12.x
```

在 IDEA 中请选择解释器 `D:\conda_envs\python312\python.exe`。每次安装依赖或运行脚本前，先执行 `conda activate "D:\conda_envs\python312"`，并用 `python -m pip` 安装。

```powershell
conda activate "D:\conda_envs\python312"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

按 `docs/IMPLEMENTATION_PLAN.md` 阶段执行，先用 `data/train.example.jsonl` 验证格式，再扩充数据。

## 已实现命令

```powershell
python main.py check
python main.py memory --init novel_memory/my_novel
Copy-Item data/train.example.jsonl data/train.jsonl
python main.py generate "设计一个东方奇幻小说开篇，主角在雨夜收到一封血书"
python main.py train --train-file data/train.jsonl
python main.py ui
```

AI 演进与本项目的对应案例见 [docs/AI_EVOLUTION_NOVEL_LLM_CASE.md](docs/AI_EVOLUTION_NOVEL_LLM_CASE.md)。该案例依据“希萌”知识库中的 AI 演进资料整理，覆盖 Qwen、LoRA、Agent、长期记忆、Web UI 和 MCP 的完整链路。

首次生成会从 Hugging Face 下载模型；训练前必须准备 `data/train.jsonl`。`python main.py ui` 会启动 Streamlit 本地写作界面。

训练完成且 `outputs/qwen25-1.5b-novel-lora/adapter_model.safetensors` 存在时，生成命令和网页界面会自动加载该 LoRA，无需修改模型路径。对比未经微调的基础模型时使用 `python main.py generate "提示词" --base-only`。

模型和网络默认值已写入 `scripts/settings.py`：自动使用 `https://hf-mirror.com`，优先使用 Hugging Face 本地缓存；后续缓存默认放在项目的 `.cache/huggingface`。因此通常无需再设置 `$env:HF_ENDPOINT` 或在命令行传 `--model`。

### Hugging Face 下载失败（SSL/MaxRetryError）

如果出现 `SSLEOFError`、`MaxRetryError` 或 `HTTPSConnectionPool(host='huggingface.co')`，说明模型尚未下载成功，通常是网络/代理链路问题，不是显卡或训练代码问题。可在当前 PowerShell 会话使用镜像：

```powershell
conda activate "D:\conda_envs\python312"
$env:HF_ENDPOINT = "https://hf-mirror.com"
python main.py generate "写一个东方奇幻小说开篇"
```

也可以永久设置（新开终端后生效）：

```powershell
[Environment]::SetEnvironmentVariable("HF_ENDPOINT", "https://hf-mirror.com", "User")
```

下载完成后可离线运行：

```powershell
$env:HF_HUB_OFFLINE = "1"
python main.py generate "续写这一章" --model "D:\models\Qwen2.5-1.5B-Instruct"
```

更稳妥的方式是用浏览器或另一台可联网电脑下载完整模型目录，再将目录复制到本机；`--model` 支持本地目录。不要把关闭 SSL 校验作为常规方案，这会掩盖证书/代理问题并降低安全性。

只使用自有、已授权或明确允许训练的数据。

## TXT 小说数据处理

项目包含 `data/志怪书.txt` 时，可执行：

```powershell
python main.py prepare
```

该命令会清洗文本、截取前 100,000 字，按约 600 字正文加 350 字前文切分为“前文 → 续写”对话样本，生成 `data/train.jsonl`（90%）和 `data/valid.jsonl`（10%）。这个长度适配当前 1024-token 训练配置，避免截断答案。原始 TXT 不会修改。可用 `python main.py prepare --chars 100000 --chunk-chars 600 --context-chars 350` 调整参数。
