from __future__ import annotations

import json
import sys
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from langchain_agent import NovelLangChainAgent  # noqa: E402
from novel_memory_store import NovelMemoryStore  # noqa: E402


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, messages, **kwargs) -> str:
        text = "\n".join(str(message.content) for message in messages)
        self.calls.append({"text": text, **kwargs})
        if "提取长期记忆" in text:
            return json.dumps(
                {
                    "summary": "主角进入旧城并发现线索。",
                    "characters": [{"name": "林默", "change": "获得线索"}],
                    "timeline": [{"time": "雨夜", "event": "进入旧城"}],
                    "foreshadowing": [{"content": "黑色戒指", "status": "未回收"}],
                },
                ensure_ascii=False,
            )
        return "生成结果"


class AgentTests(unittest.TestCase):
    @staticmethod
    def memory_directory() -> Path:
        target = PROJECT_ROOT / "tests" / "_runtime" / uuid.uuid4().hex
        target.mkdir(parents=True)
        return target

    def test_router_separates_qa_and_novel(self) -> None:
        self.assertEqual(NovelLangChainAgent.route("中国首都是哪里？"), "qa")
        self.assertEqual(NovelLangChainAgent.route("续写下一章节"), "novel")
        self.assertEqual(NovelLangChainAgent.route("续写", mode="qa"), "qa")

    def test_qa_disables_adapter_and_novel_enables_it(self) -> None:
        runtime = FakeRuntime()
        agent = NovelLangChainAgent(runtime=runtime, memory_path=self.memory_directory())
        qa = agent.invoke("1+1等于多少？")
        novel = agent.invoke("设计一个小说人物")
        self.assertEqual(qa.mode, "qa")
        self.assertEqual(novel.mode, "novel")
        self.assertFalse(runtime.calls[0]["use_adapter"])
        self.assertFalse(runtime.calls[1]["use_adapter"])
        self.assertTrue(runtime.calls[2]["use_adapter"])
        self.assertFalse(runtime.calls[3]["use_adapter"])

    def test_saved_chapter_updates_long_term_memory(self) -> None:
        runtime = FakeRuntime()
        root = self.memory_directory()
        agent = NovelLangChainAgent(runtime=runtime, memory_path=root)
        result = agent.invoke("续写下一章节", mode="novel", save=True)
        summaries = json.loads(
            (root / "chapter_summaries.json").read_text(encoding="utf-8")
        )
        characters = json.loads(
            (root / "characters.json").read_text(encoding="utf-8")
        )
        self.assertTrue(result.memory_updated)
        self.assertTrue(result.saved_path and result.saved_path.exists())
        self.assertEqual(summaries[0]["summary"], "主角进入旧城并发现线索。")
        self.assertEqual(characters[0]["name"], "林默")

    def test_memory_context_contains_configured_facts(self) -> None:
        root = self.memory_directory()
        store = NovelMemoryStore(root)
        store.initialize()
        (root / "world.md").write_text("月亮永远是红色。", encoding="utf-8")
        context = store.build_context("月亮")
        self.assertIn("月亮永远是红色", context)


if __name__ == "__main__":
    unittest.main()
