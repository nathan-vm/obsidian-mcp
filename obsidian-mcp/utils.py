from pathlib import Path


def safe_path(vault_path: Path, relative: str) -> Path:
    resolved = (vault_path / relative).resolve()
    if not str(resolved).startswith(str(vault_path.resolve())):
        raise ValueError(f"Path escapes vault: {relative}")
    return resolved
