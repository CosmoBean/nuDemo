import argparse
import json

from nudemo.cli import command_storage
from nudemo.storage.base import StorageWriteResult


class _FakeBackend:
    def write_samples(self, _samples):
        return StorageWriteResult(
            backend="fake",
            samples_written=3,
            elapsed_sec=2.0,
            bytes_written=30,
        )


def test_command_storage_serializes_slotted_write_result(monkeypatch, capsys):
    monkeypatch.setattr("nudemo.cli.AppConfig.load", lambda _config: object())
    monkeypatch.setattr("nudemo.cli.make_backends", lambda _config: {"fake": _FakeBackend()})
    monkeypatch.setattr(
        "nudemo.cli._iter_samples",
        lambda _config, _provider, _limit, _scene_limit=None: iter(()),
    )

    status = command_storage(
        argparse.Namespace(
            config=None,
            backend="fake",
            provider="real",
            limit=1,
            scene_limit=None,
        )
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "backend": "fake",
        "samples_written": 3,
        "elapsed_sec": 2.0,
        "bytes_written": 30,
        "throughput": 1.5,
    }
