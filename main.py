"""Command-line entry point for Novel LLM."""
import argparse
import subprocess
import sys

def main() -> int:
    p = argparse.ArgumentParser(description="Novel LLM 项目入口")
    p.add_argument("command", choices=("check", "prepare", "generate", "train", "memory", "agent", "mcp", "ui", "codex-ui"), nargs="?", default="check")
    p.add_argument("args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    scripts = {"check": "scripts/check_env.py", "prepare": "scripts/prepare_novel_data.py", "generate": "scripts/generate.py", "train": "scripts/train_lora.py", "memory": "scripts/memory.py", "agent": "scripts/agent.py", "mcp": "scripts/mcp_cli.py"}
    if a.command in {"ui", "codex-ui"}:
        page = "app/web_ui.py" if a.command == "ui" else "app/codex_ui.py"
        return subprocess.call([sys.executable, "-m", "streamlit", "run", page, *a.args])
    return subprocess.call([sys.executable, scripts[a.command], *a.args])

if __name__ == "__main__":
    raise SystemExit(main())
