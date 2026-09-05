# LangChain 双模式智能体

本项目使用 LangChain LCEL 封装本地 `Qwen2.5-1.5B-Instruct` 和已经训练好的小说 LoRA。

## 工作方式

```text
用户输入
  ├─ 简单问答 → 基础 Qwen（临时禁用小说 LoRA）
  └─ 小说创作 → 读取小说长期记忆
                    ↓
             基础 Qwen 制定创作计划
                    ↓
             小说 LoRA 执行计划并起草
                    ↓
             基础 Qwen 核验约束并定稿
                    ↓
       显式保存时更新章节和结构化记忆
```

程序只加载一份基础模型和一份 LoRA。问答时通过 PEFT 的 adapter 上下文临时禁用 LoRA，避免在 6GB 显存中同时加载两份 Qwen。

## 安装

```powershell
conda activate "D:\conda_envs\python312"
python -m pip install -r requirements.txt
```

## 命令行

自动识别问答或小说任务：

```powershell
python main.py agent "中国的首都是哪里？"
python main.py agent "设计一部东方奇幻小说的大纲"
```

显式选择模式：

```powershell
python main.py agent "解释什么是重力" --mode qa
python main.py agent "续写下一章节，约800字" --mode novel
```

保存章节并更新长期记忆：

```powershell
python main.py agent "续写下一章节，约800字" --mode novel --save --memory novel_memory/my_novel
```

`--save` 只对章节、续写、正文、开篇和场景类任务生效。大纲、人物设定与普通问答不会被误存为章节。

## 网页界面

```powershell
python main.py ui
```

侧边栏可以选择：

- 自动识别、简单问答或专业小说模式；
- 小说记忆目录；
- 是否保存章节并更新记忆；
- 最大生成长度和创作温度；
- 基础模型与 LoRA 路径。

模型权重在第一次提问时加载，之后由 Streamlit 缓存并复用。

## 小说长期记忆

默认目录是 `novel_memory/my_novel`：

- `world.md`：确定的世界规则；
- `outline.md`：故事总纲；
- `characters.json`：人物状态和变化；
- `timeline.json`：关键事件时间线；
- `foreshadowing.json`：伏笔及回收状态；
- `chapter_summaries.json`：最近章节摘要；
- `chapters/chapter_XXXX.md`：保存的章节正文。

生成小说前，智能体会把世界观、大纲、人物、近期时间线、未回收伏笔、最近六章摘要和上一章末尾注入提示词。基础模型先把用户要求整理为包含必写元素、场景推进、人物动机和连续性禁区的创作计划，小说 LoRA 再执行计划生成草稿。随后基础模型核验硬约束、去除重复并输出终稿。显式保存章节后，基础模型会额外执行一次结构化信息提取，并更新上述 JSON 文件。

建议先手工填写 `world.md`、`outline.md` 和主要人物。它们为空时，智能体仍能创作，但长篇约束较弱。

## 自动路由与手动模式

自动模式根据“小说、章节、续写、大纲、人物设定、世界观、剧情”等关键词选择小说链，其他请求进入问答链。遇到含义模糊的输入时，直接用 `--mode qa` 或 `--mode novel`，网页中也可以手动选择。

## 验证

不加载真实模型的快速测试：

```powershell
python -m unittest discover -s tests -v
```

真实模型短问答测试：

```powershell
python main.py agent "1+1等于多少？只回答数字。" --mode qa --max-new-tokens 16
```

Qwen2.5-1.5B 适合简单问答和受控创作。复杂事实核验、多步智能体规划和高可靠工具调用仍建议升级到更强的指令模型，或继续由 Python 固定流程控制。
