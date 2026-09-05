"""Codex-style workbench UI for the LangChain agent. Run: python main.py codex-ui"""
from __future__ import annotations
import json, sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from langchain_agent import LocalQwenRuntime, NovelLangChainAgent  # noqa: E402
from novel_memory_store import NovelMemoryStore  # noqa: E402
from mcp_client import MCPUnavailableError, load_mcp_registry  # noqa: E402
from settings import DEFAULT_ADAPTER, DEFAULT_MODEL  # noqa: E402

st.set_page_config(page_title="Novel Codex", page_icon="✦", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#fff!important;color:#171717!important}
.stApp p,.stApp span,.stApp label,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp li,.stApp div{color:#171717}
[data-testid="stSidebar"],[data-testid="stSidebarContent"]{background:#fafafa!important;color:#171717!important;border-right:1px solid #e5e7eb}
.brand{font-size:1.25rem;font-weight:700;color:#111827!important}.brand-mark{color:#6d5ce7!important}.muted{color:#5f6673!important;font-size:.8rem}.section{color:#5f6673!important;text-transform:uppercase;letter-spacing:.1em;font-size:.72rem;margin:1rem 0 .4rem}
.status{color:#166534!important;padding:.4rem .6rem;border:1px solid #bbf7d0;background:#f0fdf4;border-radius:.45rem}.hint{color:#4b5563!important;font-size:.8rem;padding:.7rem;border:1px solid #e5e7eb;border-radius:.5rem;background:#f9fafb}.file{padding:.35rem;color:#374151!important;border-bottom:1px solid #e5e7eb;font-size:.8rem}
div[data-testid="stChatMessage"]{border:1px solid #e1e4e8!important;border-radius:.7rem;padding:.6rem .8rem;margin:.5rem 0;background:#fff!important;box-shadow:0 1px 2px rgba(0,0,0,.04)}
div[data-testid="stChatMessage"] *{color:#171717!important}div[data-testid="stChatMessage"] [data-testid="stCaptionContainer"] *{color:#606775!important}
div[data-testid="stChatInput"]{background:#fff!important;border:1px solid #cbd0d8!important;border-radius:.75rem!important;box-shadow:0 1px 3px rgba(0,0,0,.08)!important}
div[data-testid="stChatInput"] > div,div[data-testid="stChatInput"] > div > div,div[data-testid="stChatInput"] > div > div > div{background:#fff!important}
div[data-testid="stChatInput"] textarea{background:#fff!important;color:#111827!important;-webkit-text-fill-color:#111827!important;caret-color:#111827!important;font-size:1rem!important}
div[data-testid="stChatInput"] textarea::placeholder{color:#6b7280!important;-webkit-text-fill-color:#6b7280!important;opacity:1!important}
div[data-testid="stChatInput"] button{background:#111827!important;color:#fff!important;border-radius:.5rem!important}div[data-testid="stChatInput"] button *{color:#fff!important}
div[data-testid="stChatInput"] button:disabled{background:#d1d5db!important;color:#6b7280!important}
.st-key-chat_history{height:calc(100vh - 245px)!important;min-height:360px!important;overflow-y:auto!important;padding:.15rem .35rem 1rem!important;scroll-behavior:smooth}
.st-key-chat_history [data-testid="stVerticalBlockBorderWrapper"]{height:100%!important}
div[data-testid="stElementContainer"]:has(> div[data-testid="stChatInput"]){position:sticky!important;bottom:.5rem!important;z-index:20!important;background:#fff!important;padding:.5rem 0!important}
input,textarea,[data-baseweb="input"]>div{background:#fff!important;color:#111827!important;-webkit-text-fill-color:#111827!important;border-color:#d1d5db!important}
[data-testid="stExpander"],details{background:#fff!important;border-color:#e5e7eb!important}button[kind="secondary"]{background:#fff!important;color:#111827!important;border-color:#d1d5db!important}
</style>""", unsafe_allow_html=True)

def path_for(value: str) -> Path:
    p = Path(value.strip() or "novel_memory/my_novel")
    return p if p.is_absolute() else PROJECT_ROOT / p
def read(p: Path, default=""):
    try: return p.read_text(encoding="utf-8") if p.exists() else default
    except OSError as e: return f"读取失败：{e}"
def write(p: Path, value: str): p.parent.mkdir(parents=True, exist_ok=True); p.write_text(value, encoding="utf-8")
@st.cache_resource(show_spinner=False)
def runtime(model: str, adapter: str | None):
    return LocalQwenRuntime(model, adapter)


@st.cache_resource(show_spinner=False)
def mcp_registry(config_path: str):
    return load_mcp_registry(config_path)


def editor(store, filename, label, json_mode=False):
    target = store.root / filename
    value = read(target, "[]" if json_mode else "")
    with st.expander(label):
        value = st.text_area(
            "内容",
            value,
            height=130,
            key=f"edit-{store.root}-{filename}",
            label_visibility="collapsed",
        )
        if st.button("保存", key=f"save-{store.root}-{filename}", use_container_width=True):
            if json_mode:
                try:
                    value = json.dumps(json.loads(value), ensure_ascii=False, indent=2)
                except Exception as exc:
                    st.error(f"JSON 格式错误：{exc}")
                    return
            write(target, value)
            st.success("已保存", icon="✅")

if "messages" not in st.session_state: st.session_state.messages=[]
if "status" not in st.session_state: st.session_state.status="就绪"
with st.sidebar:
    st.markdown("<div class='brand'><span class='brand-mark'>✦</span> Novel Codex</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>本地小说智能体工作台</div>", unsafe_allow_html=True)
    st.markdown("<div class='section'>项目</div>", unsafe_allow_html=True)
    memory = st.text_input("小说记忆目录", "novel_memory/my_novel", label_visibility="collapsed")
    store = NovelMemoryStore(path_for(memory))
    store.initialize()
    if st.button("＋ 初始化/打开项目", use_container_width=True):
        st.session_state.status = f"已打开 {store.root}"
    if st.button("新建会话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.status = "新会话已创建"
    st.markdown("<div class='section'>模式</div>", unsafe_allow_html=True)
    ml = st.radio("工作模式", ("自动识别", "简单问答", "专业小说"), label_visibility="collapsed")
    mode = {"自动识别": "auto", "简单问答": "qa", "专业小说": "novel"}[ml]
    save = st.checkbox("章节完成后保存并更新记忆", True)
    st.markdown("<div class='section'>生成参数</div>", unsafe_allow_html=True)
    tokens = st.slider("最大生成长度", 128, 2048, 800, 64)
    temp = st.slider("创作温度", 0.0, 1.5, 0.8, 0.05)
    with st.expander("模型设置"):
        model = st.text_input("基础模型", DEFAULT_MODEL)
        adapter = st.text_input("小说 LoRA", DEFAULT_ADAPTER or "")
    st.markdown("<div class='section'>章节文件</div>", unsafe_allow_html=True); chapters=sorted((store.root/"chapters").glob("chapter_*.md"), reverse=True)
    if chapters:
        for c in chapters[:12]: st.markdown(f"<div class='file'>📄 {c.name}</div>",unsafe_allow_html=True)
    else: st.caption("还没有保存章节")
    st.markdown("<div class='hint'>右侧可编辑世界观、人物和大纲。小说生成前会自动读取。</div>", unsafe_allow_html=True)

# MCP is deliberately exposed as an explicit, user-triggered tool panel.
# The small local model is never allowed to autonomously execute tools.
with st.sidebar:
    st.markdown("<div class='section'>MCP 工具</div>", unsafe_allow_html=True)
    mcp_config = PROJECT_ROOT / "configs" / "mcp_servers.json"
    st.caption(f"配置：{mcp_config}")
    registry = mcp_registry(str(mcp_config))
    if "mcp_tools" not in st.session_state:
        st.session_state.mcp_tools = []
    if st.button("刷新 MCP 工具", use_container_width=True):
        try:
            st.session_state.mcp_tools = registry.list_tools()
            st.session_state.mcp_status = f"已发现 {len(st.session_state.mcp_tools)} 个工具"
        except MCPUnavailableError as exc:
            st.session_state.mcp_status = str(exc)
        except Exception as exc:
            st.session_state.mcp_status = f"MCP 连接失败：{exc}"
    if st.session_state.get("mcp_status"):
        st.caption(st.session_state.mcp_status)
    if st.session_state.mcp_tools:
        labels = [item.get("name", "") for item in st.session_state.mcp_tools]
        selected = st.selectbox("工具", labels, key="mcp_selected_tool")
        selected_meta = next((item for item in st.session_state.mcp_tools if item.get("name") == selected), {})
        if selected_meta.get("description"):
            st.caption(selected_meta["description"])
        raw_args = st.text_area("参数 JSON", "{}", key="mcp_args", height=90)
        if st.button("执行 MCP 工具", use_container_width=True):
            try:
                args = json.loads(raw_args or "{}")
                if not isinstance(args, dict):
                    raise ValueError("参数必须是 JSON 对象")
                st.session_state.mcp_result = registry.call_tool(selected, args)
            except Exception as exc:
                st.session_state.mcp_result = f"执行失败：{exc}"
    elif mcp_config.exists():
        st.caption("点击“刷新 MCP 工具”连接本地小说记忆服务")
    else:
        st.caption("未找到 configs/mcp_servers.json，可复制示例配置后启用")
    if st.session_state.get("mcp_result") is not None:
        st.code(str(st.session_state.mcp_result), language="text")

left, center, right=st.columns([1,2.2,1.2], gap="large")
with left:
    st.markdown("### 会话")
    st.markdown(f"<div class='status'>● {st.session_state.status}</div>", unsafe_allow_html=True)
    if not st.session_state.messages:
        st.markdown("<div class='hint'>在中间输入问题或创作要求开始。</div>", unsafe_allow_html=True)
with center:
    st.markdown("### 对话与创作")
    chat_history = st.container(
        key="chat_history",
        height=520,
        border=False,
        autoscroll=True,
    )
    with chat_history:
        if not st.session_state.messages:
            st.markdown("<div class='hint'>消息会按时间顺序显示在这里。</div>", unsafe_allow_html=True)
        for item in st.session_state.messages:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])
                if item.get("meta"):
                    st.caption(item["meta"])
    prompt = st.chat_input("输入问题或创作要求……")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_prompt = prompt
        st.rerun()

    pending_prompt = st.session_state.pop("pending_prompt", None)
    if pending_prompt:
        history = [
            (item["role"], item["content"])
            for item in st.session_state.messages[-9:-1]
        ]
        try:
            agent = NovelLangChainAgent(
                runtime=runtime(model, adapter or None),
                memory_path=store.root,
                max_new_tokens=tokens,
                temperature=temp,
                mcp_registry=registry,
            )
            with st.spinner("智能体处理中……"):
                response = agent.invoke(
                    pending_prompt,
                    mode=mode,
                    history=history,
                    save=save,
                )
            meta = f"模式：{'简单问答' if response.mode == 'qa' else '专业小说'}"
            if response.mcp_tool:
                meta += f" · 已查询 MCP：{response.mcp_tool}"
            if response.saved_path:
                meta += f" · 已保存 {response.saved_path.name} 并更新记忆"
                st.session_state.status = "章节和长期记忆已更新"
            else:
                st.session_state.status = "生成完成"
            st.session_state.messages.append(
                {"role": "assistant", "content": response.content, "meta": meta}
            )
            st.rerun()
        except Exception as exc:
            st.session_state.status = "运行失败"
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"智能体运行失败：{exc}",
                    "meta": "错误",
                }
            )
            st.rerun()
with right:
    st.markdown("### 项目资料")
    st.markdown(f"<div class='muted'>{store.root}</div>", unsafe_allow_html=True)
    editor(store, "world.md", "🌍 世界观")
    editor(store, "outline.md", "🧭 故事大纲")
    editor(store, "characters.json", "👥 人物档案", True)
    editor(store, "timeline.json", "🕒 时间线", True)
    editor(store, "foreshadowing.json", "🪝 伏笔", True)
    editor(store, "chapter_summaries.json", "📝 章节摘要", True)
    with st.expander("最近章节预览"):
        st.markdown(read(chapters[0],"暂无章节") if chapters else "暂无章节")
