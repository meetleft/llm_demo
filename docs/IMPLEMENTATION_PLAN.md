# 小说写作 LLM 实施计划

## 目标

先在 RTX 3060 6GB 本地训练 Qwen2.5-1.5B-Instruct，再按需迁移到云端 7B。LoRA 负责文风和格式，RAG/结构化记忆负责长篇一致性。

## 固定开发环境

所有开发、安装、训练和推理均使用 Conda `python312` 环境：

```text
Conda：D:\miniconda3\Scripts\conda.exe
环境：D:\conda_envs\python312
解释器：D:\conda_envs\python312\python.exe
Python：3.12.x
```

PowerShell 中先运行：

```powershell
conda activate "D:\conda_envs\python312"
python -c "import sys; print(sys.executable); print(sys.version)"
```

输出路径不是 `D:\conda_envs\python312\python.exe` 时，不要安装依赖或启动训练。

## 阶段 0：环境与基线（半天）

- [ ] 激活 `D:\conda_envs\python312`，安装 `requirements.txt`。
- [ ] 验证 CUDA、显存和 Transformers。
- [ ] 下载模型并完成一次基线生成。

验收：`torch.cuda.is_available()` 为 `True`，能生成中文文本，并保存基线样例。

## 阶段 1：数据集（2～5 天）

- [ ] 确定题材和目标文风。
- [ ] 按 `data/train.example.jsonl` 制作 JSONL。
- [ ] 覆盖大纲、章纲、人物卡、续写、对话、描写、改写、摘要、伏笔提取、一致性修复。
- [ ] 去重、去乱码、去过短样本，按 90/10 切分训练/验证集。

规模：500～1,000 条验证流程；5,000～20,000 条第一版；20,000～100,000 条稳定版。质量优先。

验收：每条都有合法 `messages`，完成长度统计和人工抽样。

## 阶段 2：QLoRA 微调（1～3 天）

底座 `Qwen/Qwen2.5-1.5B-Instruct`，4-bit NF4、FP16、LoRA。配置见 `configs/qlora.yaml`：batch=1、gradient accumulation=16、上下文=1024、rank=16、alpha=32、学习率=1e-4、epoch=2～3、开启 gradient checkpointing、`fp16=true`、`bf16=false`。

- [ ] 先用 500～1,000 条短实验。
- [ ] 保存 loss、显存、耗时和固定提示样例。
- [ ] 每次只调整一个变量。

显存不足时依次降低上下文长度、累积步数和 LoRA rank。

验收：无 OOM，验证 loss 下降，文风和格式优于基线且无明显复读。

## 阶段 3：长篇记忆/RAG（3～5 天）

每部小说维护 `world.md`、`characters.json`、`timeline.json`、`foreshadowing.json`、`outline.md`、`chapter_summaries.json`。

- [ ] 章节切片与摘要。
- [ ] FAISS/Chroma 向量索引。
- [ ] 生成前检索章纲、人物、最近章节、规则和未回收伏笔。
- [ ] 生成后更新摘要、人物状态和伏笔状态。

验收：跨 10 章测试无明显人物、时间线和设定冲突。

## 阶段 4：写作应用（2～4 天）

- [ ] Streamlit 页面：创建项目，生成大纲、章纲和章节。
- [ ] 支持续写、重写、扩写、压缩、换文风、一致性检查。
- [ ] 保存 Markdown 并同步记忆文件。
- [ ] 暴露 temperature、top_p、max_new_tokens、重复惩罚。

验收：新建项目到保存一章不超过 5 个操作，重启后数据可恢复。

## 阶段 5：评估迭代（持续）

固定记录结构、连贯、文风、可读性四类指标；每次只改一个主要因素，保留 adapter、配置和结果。1.5B 稳定后再用同一数据在 24GB 云 GPU 训练 7B QLoRA。

里程碑：`M1 环境可生成 → M2 500 条可训练 → M3 5k 条风格稳定 → M4 跨章一致 → M5 应用可用 → M6 决定是否上 7B`。
