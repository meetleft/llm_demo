"""Project defaults; edit this file instead of repeating CLI environment flags."""
from __future__ import annotations

import os
from pathlib import Path

# Hugging Face mirror used for all automatic downloads.
HF_ENDPOINT = "https://hf-mirror.com"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Keep writable caches inside this project to avoid Windows permission/lock issues.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_HOME = Path(os.environ.get("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface")))

# Prefer a complete snapshot from the old C: cache, legacy D: cache, or project cache.
_cache_roots = [Path.home() / ".cache" / "huggingface", Path(r"D:\huggingface"), HF_HOME]
_snapshots: list[Path] = []
for root in _cache_roots:
    snapshots = root / "hub" / "models--Qwen--Qwen2.5-1.5B-Instruct" / "snapshots"
    if snapshots.exists():
        _snapshots.extend(p for p in snapshots.glob("*") if p.is_dir() and (p / "config.json").exists() and (p / "model.safetensors").exists())
_snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
DEFAULT_MODEL = str(_snapshots[0]) if _snapshots else MODEL_ID

# Automatically use the finished LoRA adapter when it is present.
_adapter_dir = PROJECT_ROOT / "outputs" / "qwen25-1.5b-novel-lora"
DEFAULT_ADAPTER = str(_adapter_dir) if (_adapter_dir / "adapter_config.json").exists() and (_adapter_dir / "adapter_model.safetensors").exists() else None

def configure_huggingface() -> None:
    """Apply project defaults without overwriting explicit user settings."""
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_HOME / "datasets"))
