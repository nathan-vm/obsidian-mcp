import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    vault_path: Path
    qdrant_url: str
    lm_studio_url: str
    embedding_model: str
    collection_name: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int
    mcp_port: int
    observer_interval: float


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise EnvironmentError(f"{name} must be an integer, got {raw!r}")
    if value <= 0:
        raise EnvironmentError(f"{name} must be positive, got {value}")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError:
        raise EnvironmentError(f"{name} must be a number, got {raw!r}")
    if value <= 0:
        raise EnvironmentError(f"{name} must be positive, got {value}")
    return value


def load_config() -> Config:
    return Config(
        vault_path=Path(os.environ.get("VAULT_PATH", "/vault")),
        qdrant_url=os.environ.get("QDRANT_URL", "http://qdrant:6333"),
        lm_studio_url=os.environ.get("LM_STUDIO_URL", "http://host.docker.internal:1234/v1"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "nomic-embed-text"),
        collection_name=os.environ.get("COLLECTION_NAME", "obsidian_vault"),
        embedding_dim=_positive_int("EMBEDDING_DIM", 768),
        chunk_size=_positive_int("CHUNK_SIZE", 500),
        chunk_overlap=_positive_int("CHUNK_OVERLAP", 50),
        mcp_port=_positive_int("MCP_PORT", 55000),
        observer_interval=_positive_float("OBSERVER_INTERVAL", 1.0),
    )


config = load_config()
