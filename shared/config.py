import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    vault_path: Path
    data_path: Path
    embedding_model: str
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    observer_interval: float

    @property
    def qdrant_path(self) -> Path:
        return self.data_path / "qdrant"

    @property
    def active_collection(self) -> str:
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
        data_path=Path(os.environ.get("DATA_PATH", "/data")),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5"),
        collection_name=os.environ.get("COLLECTION_NAME", "obsidian_vault"),
        chunk_size=_positive_int("CHUNK_SIZE", 500),
        chunk_overlap=_positive_int("CHUNK_OVERLAP", 50),
        observer_interval=_positive_float("OBSERVER_INTERVAL", 1.0),
    )


config = load_config()
