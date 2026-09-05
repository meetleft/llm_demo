# MCP 接入说明

项目已内置一个只读的小说记忆 MCP Server，位于 `scripts/novel_memory_mcp.py`。默认配置是 `configs/mcp_servers.json`，它只暴露当前项目记忆目录中的查询工具，不提供任意文件写入。对话 Agent 会先让 Qwen 根据语义判断是否需要 MCP，不依赖关键词；只有输出合法调用计划且工具在白名单中时才会执行一次调用。

## 页面操作

运行 `python main.py codex-ui`（或双击“启动Novel Codex.bat”）。正常输入问题即可触发语义判断；若调用了 MCP，结果会作为“外部资料”交给模型参考。左侧 **MCP 工具** 区域仍可用于手动测试工具。

## 命令行

```powershell
python main.py mcp list
python main.py mcp call list_chapters
python main.py mcp call get_character --args "{\"name\":\"林默\"}"
```

## 接入其他 MCP Server

复制 `configs/mcp_servers.example.json`，在 `configs/mcp_servers.json` 的 `mcpServers` 中增加服务器配置。支持的传输方式取决于已安装的 MCP 客户端（stdio、SSE 或 streamable HTTP）。安装依赖后即可使用 LangChain MCP adapter；即使 adapter 暂时无法安装，内置 stdio 客户端仍可连接本地服务。

MCP 返回内容属于外部资料，调用结果会限制长度；不要把未验证的工具输出当作系统指令或直接写入记忆文件。
