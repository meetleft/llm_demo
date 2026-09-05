# 训练数据

每行一个 JSONL 样本，采用 OpenAI chat `messages` 格式。复制 `train.example.jsonl` 为 `train.jsonl` 后扩充；验证集放在 `valid.jsonl`。

建议覆盖大纲、章纲、人物卡、续写、对话、环境描写、文风改写、章节摘要、伏笔提取和一致性修复。

对于 TXT 小说，可在项目根目录执行 `python main.py prepare`。默认读取 `data/志怪书.txt` 的前 100,000 字，按约 600 字正文加 350 字前文切分为续写样本。
