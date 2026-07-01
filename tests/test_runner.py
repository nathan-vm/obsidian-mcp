from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


@pytest.fixture
def mock_store() -> MagicMock:
    store = MagicMock()
    store.ensure_collection.return_value = True
    return mock_store


def _run(store, vault_path, interval=1.0):
    """Run run_indexer with Observer and time.sleep mocked out."""
    mock_observer = MagicMock()

    # Make sleep raise on first call so the while-loop exits immediately
    with (
        patch("runner.Observer", return_value=mock_observer),
        patch("runner.time.sleep", side_effect=Exception("stop")),
    ):
        from runner import run_indexer

        run_indexer(store, vault_path, interval)

    return mock_observer


class TestRunIndexer:
    def test_full_reindex_when_collection_needs_it(self, tmp_path):
        store = MagicMock()
        store.ensure_collection.return_value = True

        _run(store, tmp_path)

        store.full_reindex.assert_called_once_with(tmp_path)

    def test_skips_reindex_when_collection_ready(self, tmp_path):
        store = MagicMock()
        store.ensure_collection.return_value = False

        _run(store, tmp_path)

        store.full_reindex.assert_not_called()

    def test_observer_is_started_and_stopped(self, tmp_path):
        store = MagicMock()
        store.ensure_collection.return_value = False

        observer = _run(store, tmp_path)

        observer.start.assert_called_once()
        observer.stop.assert_called_once()
        observer.join.assert_called_once()

    def test_observer_schedules_vault_path(self, tmp_path):
        store = MagicMock()
        store.ensure_collection.return_value = False

        observer = _run(store, tmp_path)

        observer.schedule.assert_called_once()
        _, kwargs = observer.schedule.call_args
        # path is passed as positional arg
        args = observer.schedule.call_args[0]
        assert str(tmp_path) in args
