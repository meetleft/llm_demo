"""Run the LangChain Q&A and novel-writing agent."""
from __future__ import annotations

import argparse
import sys

from langchain_agent import NovelLangChainAgent
from mcp_client import load_mcp_registry
from settings import DEFAULT_ADAPTER, DEFAULT_MODEL


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="问题或小说创作要求")
    parser.add_argument("--mode", choices=("auto", "qa", "novel"), default="auto")
    parser.add_argument("--memory", default="novel_memory/my_novel")
    parser.add_argument("--save", action="store_true", help="章节生成后保存并更新长期记忆")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--max-new-tokens", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    prompt = args.prompt or input("请输入问题或创作要求：").strip()

    agent = NovelLangChainAgent(
        model_path=args.model,
        adapter_path=args.adapter,
        memory_path=args.memory,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        mcp_registry=load_mcp_registry(),
    )
    response = agent.invoke(prompt, mode=args.mode, save=args.save)
    print(response.content)
    print(f"\n[模式：{response.mode}]")
    if response.saved_path:
        print(f"[已保存：{response.saved_path}；长期记忆已更新]")
    elif args.save and response.mode == "novel":
        print("[本次不是章节正文任务，未自动写入章节记忆]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
