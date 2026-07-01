import os
from pathlib import Path
from unittest.mock import patch

import pytest

from shared.config import Config, _positive_float, _positive_int, load_config


def _make_config(**kwargs) -> Config:
    defaults = dict(
        vault_path=Path("/vault"),
        data_path=Path("/data"),
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
        collection_name="obsidian_vault",
        chunk_size=500,
        chunk_overlap=50,
        observer_interval=1.0,
    )
    return Config(**{**defaults, **kwargs})


class TestConfigProperties:
    def test_qdrant_path(self):
        cfg = _make_config(data_path=Path("/mydata"))
        assert cfg.qdrant_path == Path("/mydata/qdrant")

    def test_active_collection_slugifies_model_name(self):
        cfg = _make_config(
            collection_name="my_vault",
            embedding_model="nomic-ai/nomic-embed-text-v1.5",
        )
        assert cfg.active_collection == "my_vault__nomic_ai_nomic_embed_text_v1_5"

    def test_active_collection_strips_leading_trailing_underscores(self):
        cfg = _make_config(collection_name="vault", embedding_model="/odd-model/")
        assert not cfg.active_collection.split("__")[1].startswith("_")
        assert not cfg.active_collection.split("__")[1].endswith("_")


class TestPositiveInt:
    def test_returns_default_when_env_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MY_INT", None)
            assert _positive_int("MY_INT", 42) == 42

    def test_reads_from_env(self):
        with patch.dict(os.environ, {"MY_INT": "7"}):
            assert _positive_int("MY_INT", 1) == 7

    def test_raises_on_non_integer(self):
        with patch.dict(os.environ, {"MY_INT": "abc"}):
            with pytest.raises(EnvironmentError, match="must be an integer"):
                _positive_int("MY_INT", 1)

    def test_raises_on_zero(self):
        with patch.dict(os.environ, {"MY_INT": "0"}):
            with pytest.raises(EnvironmentError, match="must be positive"):
                _positive_int("MY_INT", 1)

    def test_raises_on_negative(self):
        with patch.dict(os.environ, {"MY_INT": "-5"}):
            with pytest.raises(EnvironmentError, match="must be positive"):
                _positive_int("MY_INT", 1)


class TestPositiveFloat:
    def test_returns_default_when_env_unset(self):
        os.environ.pop("MY_FLOAT", None)
        assert _positive_float("MY_FLOAT", 1.5) == 1.5

    def test_reads_from_env(self):
        with patch.dict(os.environ, {"MY_FLOAT": "2.5"}):
            assert _positive_float("MY_FLOAT", 1.0) == 2.5

    def test_raises_on_non_float(self):
        with patch.dict(os.environ, {"MY_FLOAT": "nope"}):
            with pytest.raises(EnvironmentError, match="must be a number"):
                _positive_float("MY_FLOAT", 1.0)

    def test_raises_on_zero(self):
        with patch.dict(os.environ, {"MY_FLOAT": "0"}):
            with pytest.raises(EnvironmentError, match="must be positive"):
                _positive_float("MY_FLOAT", 1.0)

    def test_raises_on_negative(self):
        with patch.dict(os.environ, {"MY_FLOAT": "-1.0"}):
            with pytest.raises(EnvironmentError, match="must be positive"):
                _positive_float("MY_FLOAT", 1.0)


class TestLoadConfig:
    def test_defaults(self):
        env = {k: v for k, v in os.environ.items()
               if k not in {"VAULT_PATH", "DATA_PATH", "EMBEDDING_MODEL",
                            "COLLECTION_NAME", "CHUNK_SIZE", "CHUNK_OVERLAP", "OBSERVER_INTERVAL"}}
        with patch.dict(os.environ, env, clear=True):
            cfg = load_config()
        assert cfg.vault_path == Path("/vault")
        assert cfg.data_path == Path("/data")
        assert cfg.chunk_size == 500
        assert cfg.chunk_overlap == 50
        assert cfg.observer_interval == 1.0

    def test_env_overrides(self):
        with patch.dict(os.environ, {
            "VAULT_PATH": "/my/vault",
            "DATA_PATH": "/my/data",
            "EMBEDDING_MODEL": "my-model",
            "COLLECTION_NAME": "my_col",
            "CHUNK_SIZE": "200",
            "CHUNK_OVERLAP": "20",
            "OBSERVER_INTERVAL": "0.5",
        }):
            cfg = load_config()
        assert cfg.vault_path == Path("/my/vault")
        assert cfg.chunk_size == 200
        assert cfg.observer_interval == 0.5
