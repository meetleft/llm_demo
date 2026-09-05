"""Convert the first N Chinese characters of a TXT novel into chat JSONL samples."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

SYSTEM = "你是一名中文长篇小说写作助手，遵守人物和世界观设定，保持叙事连贯，推动剧情，避免复读。"

def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try: return raw.decode(encoding)
        except UnicodeDecodeError: pass
    return raw.decode("utf-8", errors="replace")

def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def chunk_text(text: str, target: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        while len(para) > target:
            if current: chunks.append(current); current = ""
            chunks.append(para[:target]); para = para[target:]
        if not para: continue
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > target and current: chunks.append(current); current = para
        else: current = candidate
    if current: chunks.append(current)
    return chunks

def make_sample(target: str, context: str, index: int) -> dict:
    user = "请根据前文自然续写下一段，保持人物、语气和情节连贯。\n\n前文：\n" + (context or "（故事开头）")
    return {"messages": [{"role":"system","content":SYSTEM},{"role":"user","content":user},{"role":"assistant","content":target}], "source":"志怪书.txt", "chunk_id":index}

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/志怪书.txt"); p.add_argument("--chars", type=int, default=100_000)
    p.add_argument("--chunk-chars", type=int, default=600); p.add_argument("--context-chars", type=int, default=350)
    p.add_argument("--train-output", default="data/train.jsonl"); p.add_argument("--valid-output", default="data/valid.jsonl")
    a = p.parse_args(); source = Path(a.input)
    if not source.exists(): raise SystemExit(f"找不到输入文件：{source}")
    text = clean(read_text(source))[:a.chars]; chunks = chunk_text(text, a.chunk_chars)
    rows = []
    for i, target in enumerate(chunks):
        context = chunks[i-1][-a.context_chars:] if i else ""
        rows.append(make_sample(target, context, i))
    if len(rows) < 2: raise SystemExit("文本过短，至少需要生成 2 条样本")
    split = max(1, min(len(rows)-1, int(len(rows)*0.9)))
    for path, part in ((a.train_output, rows[:split]), (a.valid_output, rows[split:])):
        with Path(path).open("w", encoding="utf-8", newline="\n") as f:
            for row in part: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"原文：{source}，清洗后截取：{len(text)} 字；样本：{len(rows)} 条；训练：{split}；验证：{len(rows)-split}")
    print(f"已写入：{a.train_output}、{a.valid_output}"); return 0

if __name__ == "__main__": raise SystemExit(main())
