#!/usr/bin/env python3
"""Add the obsidian MCP server to Claude Code's settings."""
import json
import os
import pathlib

MCP_PORT = os.environ.get("MCP_PORT", "55000")
SETTINGS_PATH = pathlib.Path.home() / ".claude" / "settings.json"

SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

settings: dict = {}
if SETTINGS_PATH.exists():
    settings = json.loads(SETTINGS_PATH.read_text())

settings.setdefault("mcpServers", {})
settings["mcpServers"]["obsidian"] = {
    "url": f"http://localhost:{MCP_PORT}/sse"
}

SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
print(f"Updated {SETTINGS_PATH}")
print(f"  obsidian MCP → http://localhost:{MCP_PORT}/sse")
print()
print("Restart Claude Code to load the new server.")
