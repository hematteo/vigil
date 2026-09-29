from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from vigil.__main__ import main


def _write_config(tmp_path, text: str):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_watch_test_alert_uses_cli_overrides(tmp_path, capsys):
    cfg = _write_config(tmp_path, "api_key: k\n")
    with patch("vigil.alerts.post_webhook_alert", new=AsyncMock()) as post:
        main(["watch", "-c", str(cfg), "--webhook", "https://ntfy.sh/t", "--webhook-format", "ntfy", "--test-alert"])
    args, kwargs = post.call_args
    assert args[0] == "https://ntfy.sh/t"
    assert kwargs["format"] == "ntfy"
    assert "Sent test alert" in capsys.readouterr().out


def test_watch_test_alert_requires_webhook(tmp_path):
    cfg = _write_config(tmp_path, "api_key: k\n")
    with pytest.raises(SystemExit):
        main(["watch", "-c", str(cfg), "--test-alert"])


def test_watch_requires_api_key(tmp_path, monkeypatch):
    for var in ("VIGIL_API_KEY", "VAST_API_KEY", "RUNPOD_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("vigil.providers.vast.VastProvider.api_key_file", lambda self: None)
    monkeypatch.setattr("vigil.config.DEFAULT_API_KEY_PATH", tmp_path / "missing")
    cfg = _write_config(tmp_path, "provider: runpod\n")
    with pytest.raises(SystemExit, match="RunPod API key"):
        main(["watch", "-c", str(cfg)])


def test_watch_runs_watcher_with_config_provider(tmp_path):
    cfg = _write_config(tmp_path, "api_key: k\nprovider: runpod\nalert_webhook_url: https://x\n")
    with patch("vigil.watch.Watcher") as watcher_cls:
        watcher_cls.return_value.run = AsyncMock()
        main(["watch", "-c", str(cfg), "--stall-minutes", "20"])
    config, provider = watcher_cls.call_args[0]
    assert provider.name == "runpod"
    assert config.stall_threshold_minutes == 20
