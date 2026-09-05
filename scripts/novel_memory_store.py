"""File-backed long-term memory for a novel project."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


MEMORY_FILES = {
    "characters.json": [],
    "timeline.json": [],
    "foreshadowing.json": [],
    "chapter_summaries.json": [],
}


class NovelMemoryStore:
    """Read and update the small, structured memory used by the writing chain."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "chapters").mkdir(exist_ok=True)
        for name, default in MEMORY_FILES.items():
            target = self.root / name
            if not target.exists():
                self._write_json(target, default)
        for name, title in (("world.md", "世界观"), ("outline.md", "故事大纲")):
            target = self.root / name
            if not target.exists():
                target.write_text(f"# {title}\n", encoding="utf-8")

    def _read_json(self, name: str) -> list[dict[str, Any]]:
        target = self.root / name
        if not target.exists():
            return []
        try:
            value = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return value if isinstance(value, list) else []

    @staticmethod
    def _write_json(target: Path, value: Any) -> None:
        target.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _clip(text: str, limit: int, *, keep_tail: bool = False) -> str:
        if len(text) <= limit:
            return text
        marker = "\n……（内容已截断）\n"
        if keep_tail:
            return marker + text[-(limit - len(marker)) :]
        return text[: limit - len(marker)] + marker

    def _read_markdown(self, name: str, limit: int) -> str:
        target = self.root / name
        if not target.exists():
            return "（未设置）"
        text = target.read_text(encoding="utf-8").strip()
        return self._clip(text, limit) if text else "（未设置）"

    @staticmethod
    def _json_text(items: list[dict[str, Any]]) -> str:
        return json.dumps(items, ensure_ascii=False, indent=2) if items else "[]"

    def build_context(self, query: str = "", max_chars: int = 7000) -> str:
        """Build a bounded prompt context from authoritative and recent memory."""
        self.initialize()
        characters = self._read_json("characters.json")
        if query:
            # Put characters named in the request first without dropping the rest.
            characters.sort(
                key=lambda item: 0
                if (
                    str(item.get("name", item.get("姓名", "")))
                    and str(item.get("name", item.get("姓名", ""))) in query
                )
                else 1
            )
        timeline = self._read_json("timeline.json")[-12:]
        foreshadowing = self._read_json("foreshadowing.json")
        unresolved = [
            item
            for item in foreshadowing
            if str(item.get("status", item.get("状态", "未回收"))).lower()
            not in {"resolved", "closed", "已回收", "完成"}
        ][-12:]
        summaries = self._read_json("chapter_summaries.json")[-6:]

        recent_chapter = "（暂无）"
        chapters = sorted((self.root / "chapters").glob("chapter_*.md"))
        if chapters:
            recent_chapter = self._clip(
                chapters[-1].read_text(encoding="utf-8"), 1800, keep_tail=True
            )

        sections = [
            ("世界观（最高优先级）", self._read_markdown("world.md", 1400)),
            ("故事大纲", self._read_markdown("outline.md", 1400)),
            ("人物档案", self._clip(self._json_text(characters), 1400)),
            ("近期时间线", self._clip(self._json_text(timeline), 900)),
            ("未回收伏笔", self._clip(self._json_text(unresolved), 900)),
            ("最近章节摘要", self._clip(self._json_text(summaries), 1200)),
            ("最近章节末尾", recent_chapter),
        ]
        result: list[str] = []
        remaining = max_chars
        for title, content in sections:
            block = f"## {title}\n{content.strip()}\n"
            if remaining <= 0:
                break
            block = self._clip(block, remaining)
            result.append(block)
            remaining -= len(block)
        return "\n".join(result)

    def next_chapter_number(self) -> int:
        self.initialize()
        numbers: list[int] = []
        for target in (self.root / "chapters").glob("chapter_*.md"):
            match = re.fullmatch(r"chapter_(\d+)\.md", target.name)
            if match:
                numbers.append(int(match.group(1)))
        return max(numbers, default=0) + 1

    def save_chapter(self, content: str, chapter_number: int | None = None) -> Path:
        self.initialize()
        number = chapter_number or self.next_chapter_number()
        target = self.root / "chapters" / f"chapter_{number:04d}.md"
        if target.exists():
            raise FileExistsError(f"章节文件已经存在：{target}")
        target.write_text(content.strip() + "\n", encoding="utf-8")
        return target

    def apply_chapter_update(
        self,
        chapter_number: int,
        chapter_path: Path,
        update: dict[str, Any],
        fallback_summary: str,
    ) -> None:
        """Merge model-extracted facts after a chapter has been explicitly saved."""
        self.initialize()
        summary = str(update.get("summary", "")).strip() or fallback_summary[:300]
        summaries = self._read_json("chapter_summaries.json")
        summaries.append(
            {
                "chapter": chapter_number,
                "file": chapter_path.name,
                "summary": summary,
            }
        )
        self._write_json(self.root / "chapter_summaries.json", summaries)

        mappings = (
            ("characters", "characters.json"),
            ("timeline", "timeline.json"),
            ("foreshadowing", "foreshadowing.json"),
        )
        for response_key, filename in mappings:
            additions = update.get(response_key, [])
            if not isinstance(additions, list):
                continue
            current = self._read_json(filename)
            for item in additions:
                if isinstance(item, dict):
                    item = {**item, "chapter": item.get("chapter", chapter_number)}
                    current.append(item)
            self._write_json(self.root / filename, current)
