"""Minimal Streamlit UI. Run: python main.py ui"""
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from settings import DEFAULT_ADAPTER, DEFAULT_MODEL  # noqa: E402

st.set_page_config(page_title="Novel LLM", layout="wide")
st.title("Novel LLM 小说写作助手")
st.caption("固定环境：Conda python312 · RTX 3060 6GB · Qwen2.5-1.5B + QLoRA")
prompt = st.text_area("创作要求", height=220, placeholder="例如：根据以下章节续写 800 字……")
model = st.text_input("基础模型目录", DEFAULT_MODEL)
use_adapter = st.checkbox("使用训练好的小说 LoRA", value=bool(DEFAULT_ADAPTER))
adapter = st.text_input("LoRA adapter 目录", DEFAULT_ADAPTER or "", disabled=not use_adapter)
max_tokens = st.slider("最大生成长度", 128, 2048, 800, 64)
if st.button("生成", type="primary"):
    if not prompt.strip():
        st.warning("请先输入创作要求")
    else:
        cmd = [sys.executable, str(Path(__file__).parents[1] / "scripts" / "generate.py"), prompt, "--model", model, "--max-new-tokens", str(max_tokens)]
        if use_adapter and adapter: cmd.extend(["--adapter", adapter])
        else: cmd.append("--base-only")
        child_env = os.environ.copy(); child_env["PYTHONIOENCODING"] = "utf-8"; child_env["PYTHONUTF8"] = "1"
        with st.spinner("模型生成中……"):
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=child_env)
        if result.returncode: st.error(result.stderr or result.stdout)
        else: st.markdown(result.stdout)
