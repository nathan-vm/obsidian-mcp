import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    vault_path: Path
    qdrant_url: str
    embedding_provider: str  # lm_studio | openai | gemini
    embedding_model: str
    lm_studio_url: str
    openai_api_key: str
    gemini_api_key: str
    collection_name: str  # base name; active collection is derived per model
    chunk_size: int
    chunk_overlap: int
    mcp_port: int
    observer_interval: float

    @property
    def active_collection(self) -> str:
        """Collection name scoped to the current embedding model.

        Changing EMBEDDING_MODEL automatically uses a separate collection,
        so you can test multiple models side-by-side without data corruption.
        """
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", self.embedding_model).strip("_").lower()
        return f"{self.collection_name}__{slug}"


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
        embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "lm_studio"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "nomic-embed-text"),
        lm_studio_url=os.environ.get("LM_STUDIO_URL", "http://host.docker.internal:1234/v1"),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        collection_name=os.environ.get("COLLECTION_NAME", "obsidian_vault"),
        chunk_size=_positive_int("CHUNK_SIZE", 500),
        chunk_overlap=_positive_int("CHUNK_OVERLAP", 50),
        mcp_port=_positive_int("MCP_PORT", 55000),
        observer_interval=_positive_float("OBSERVER_INTERVAL", 1.0),
    )


config = load_config()
