"""List and explicitly call configured MCP tools."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from mcp_client import MCPUnavailableError, load_mcp_registry

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "call"))
    parser.add_argument("name", nargs="?", help="tool name for call")
    parser.add_argument("--args", default="{}", help="JSON object passed to the tool")
    parser.add_argument("--config", default="configs/mcp_servers.json")
    args = parser.parse_args()
    try:
        registry = load_mcp_registry(Path(args.config))
        if args.action == "list":
            print(json.dumps(registry.list_tools(), ensure_ascii=False, indent=2)); return 0
        if not args.name: parser.error("call requires a tool name")
        values = json.loads(args.args)
        if not isinstance(values, dict): parser.error("--args must be a JSON object")
        result = registry.call_tool(args.name, values)
        print(result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2)); return 0
    except MCPUnavailableError as exc:
        print(f"MCP unavailable: {exc}", file=sys.stderr); return 2
    except Exception as exc:
        print(f"MCP error: {exc}", file=sys.stderr); return 1

if __name__ == "__main__": raise SystemExit(main())
