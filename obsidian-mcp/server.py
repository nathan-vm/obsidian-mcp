import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP

from shared.config import config
from tools.notes import register as register_notes
from tools.search import register as register_search

mcp = FastMCP("obsidian", host="0.0.0.0", port=config.mcp_port)
register_notes(mcp, config)
register_search(mcp, config)

if __name__ == "__main__":
    mcp.run(transport="sse")
