from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from vigil.alerts import (
    _format_discord,
    _format_raw,
    _format_slack,
    post_webhook_alert,
)
from vigil.discovery import InstanceInfo


def _make_instance(**overrides) -> InstanceInfo:
    defaults = {
        "id": 12345,
        "ssh_host": "ssh5.vast.ai",
        "ssh_port": 22222,
        "gpu_name": "RTX 4090",
        "num_gpus": 2,
        "status": "running",
        "machine_id": 9999,
        "label": None,
        "dph_total": 0.452,
    }
    defaults.update(overrides)
    return InstanceInfo(**defaults)


# ---------------------------------------------------------------------------
# _format_raw
# ---------------------------------------------------------------------------


class TestFormatRaw:
    def test_basic_payload(self):
        before = time.time()
        result = _format_raw(42, "error", "GPU overheating")
        after = time.time()

        assert result["instance_id"] == 42
        assert result["alert_type"] == "error"
        assert result["message"] == "GPU overheating"
        assert before <= result["timestamp"] <= after
        assert "metrics" not in result
        assert "gpu_name" not in result

    def test_with_metrics(self):
        metrics = {"gpu_util": "12%", "temp": "95C"}
        result = _format_raw(1, "low_gpu", "Low utilization", metrics=metrics)

        assert result["metrics"] is metrics
        assert result["metrics"]["gpu_util"] == "12%"

    def test_with_instance_info(self):
        inst = _make_instance(gpu_name="A100", num_gpus=8, dph_total=1.234)
        result = _format_raw(7, "error", "Crashed", instance=inst)

        assert result["gpu_name"] == "A100"
        assert result["num_gpus"] == 8
        assert result["dph_total"] == 1.234

    def test_with_metrics_and_instance(self):
        inst = _make_instance()
        metrics = {"mem": "80%"}
        result = _format_raw(7, "warn", "High mem", metrics=metrics, instance=inst)

        assert result["metrics"] == metrics
        assert result["gpu_name"] == inst.gpu_name

    def test_empty_metrics_dict_included(self):
        result = _format_raw(1, "info", "test", metrics={})
        assert result["metrics"] == {}

    def test_none_instance_omits_gpu_fields(self):
        result = _format_raw(1, "info", "test", instance=None)
        assert "gpu_name" not in result
        assert "num_gpus" not in result
        assert "dph_total" not in result


# ---------------------------------------------------------------------------
# _format_slack
# ---------------------------------------------------------------------------


class TestFormatSlack:
    def test_basic_structure(self):
        result = _format_slack(99, "error", "Something broke")

        assert result["text"] == "vigil alert: #99"
        assert len(result["blocks"]) == 1
        block = result["blocks"][0]
        assert block["type"] == "section"
        assert block["text"]["type"] == "mrkdwn"
        assert "*ERROR*" in block["text"]["text"]
        assert "Instance #99" in block["text"]["text"]
        assert "Something broke" in block["text"]["text"]

    def test_with_instance(self):
        inst = _make_instance(gpu_name="H100", num_gpus=4, dph_total=2.5)
        result = _format_slack(10, "warn", "Check GPU", instance=inst)

        text = result["blocks"][0]["text"]["text"]
        assert "GPU: H100 x4" in text
        assert "$2.500/hr" in text

    def test_with_metrics(self):
        metrics = {"util": "5%", "temp": "30C"}
        result = _format_slack(10, "low_gpu", "Low util", metrics=metrics)

        text = result["blocks"][0]["text"]["text"]
        assert "Metrics:" in text
        assert "util=5%" in text
        assert "temp=30C" in text

    def test_with_both_instance_and_metrics(self):
        inst = _make_instance()
        metrics = {"gpu": "10%"}
        result = _format_slack(5, "slow", "Slow training", metrics=metrics, instance=inst)

        text = result["blocks"][0]["text"]["text"]
        assert "GPU:" in text
        assert "Metrics:" in text

    def test_without_optional_fields(self):
        result = _format_slack(1, "info", "All good")

        text = result["blocks"][0]["text"]["text"]
        assert "GPU:" not in text
        assert "Metrics:" not in text

    def test_alert_type_uppercased(self):
        result = _format_slack(1, "low_gpu", "test")
        text = result["blocks"][0]["text"]["text"]
        assert "*LOW_GPU*" in text


# ---------------------------------------------------------------------------
# _format_discord
# ---------------------------------------------------------------------------


class TestFormatDiscord:
    def test_basic_structure(self):
        result = _format_discord(42, "error", "GPU crash")

        assert len(result["embeds"]) == 1
        embed = result["embeds"][0]
        assert embed["title"] == "vigil alert: #42"
        assert embed["description"] == "GPU crash"
        assert embed["footer"] == {"text": "error"}

    def test_error_type_red_color(self):
        result = _format_discord(1, "error", "msg")
        assert result["embeds"][0]["color"] == 16711680

    def test_unknown_type_red_color(self):
        result = _format_discord(1, "crash", "msg")
        assert result["embeds"][0]["color"] == 16711680

    def test_warning_type_low_gpu_orange(self):
        result = _format_discord(1, "low_gpu", "msg")
        assert result["embeds"][0]["color"] == 16744448

    def test_warning_type_slow_orange(self):
        result = _format_discord(1, "slow", "msg")
        assert result["embeds"][0]["color"] == 16744448

    def test_with_instance_fields(self):
        inst = _make_instance(gpu_name="A100", num_gpus=8, dph_total=3.141)
        result = _format_discord(5, "error", "oops", instance=inst)

        fields = result["embeds"][0]["fields"]
        assert len(fields) == 2
        assert fields[0] == {"name": "GPU", "value": "A100 x8", "inline": True}
        assert fields[1] == {"name": "Cost", "value": "$3.141/hr", "inline": True}

    def test_with_metrics_fields(self):
        metrics = {"util": "99%"}
        result = _format_discord(5, "error", "hot", metrics=metrics)

        fields = result["embeds"][0]["fields"]
        assert len(fields) == 1
        assert fields[0]["name"] == "Metrics"
        assert fields[0]["value"] == "util=99%"
        assert fields[0]["inline"] is False

    def test_with_instance_and_metrics(self):
        inst = _make_instance()
        metrics = {"temp": "80C"}
        result = _format_discord(5, "error", "hot", metrics=metrics, instance=inst)

        fields = result["embeds"][0]["fields"]
        assert len(fields) == 3
        assert fields[0]["name"] == "GPU"
        assert fields[1]["name"] == "Cost"
        assert fields[2]["name"] == "Metrics"

    def test_no_optional_fields_empty_list(self):
        result = _format_discord(1, "error", "msg")
        assert result["embeds"][0]["fields"] == []


# ---------------------------------------------------------------------------
# post_webhook_alert
# ---------------------------------------------------------------------------


class TestPostWebhookAlert:
    @pytest.mark.anyio
    async def test_uses_provided_client(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=httpx.Response(200))

        await post_webhook_alert(
            "https://hooks.example.com/webhook",
            42,
            "error",
            "GPU crash",
            client=mock_client,
        )

        mock_client.post.assert_awaited_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://hooks.example.com/webhook"
        assert call_args[1]["timeout"] == 10.0
        payload = call_args[1]["json"]
        assert payload["instance_id"] == 42
        assert payload["alert_type"] == "error"

    @pytest.mark.anyio
    async def test_creates_own_client_when_none(self):
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=httpx.Response(200))
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("vigil.alerts.httpx.AsyncClient", return_value=mock_client_instance):
            await post_webhook_alert("https://hooks.example.com/wh", 1, "info", "test")

        mock_client_instance.post.assert_awaited_once()

    @pytest.mark.anyio
    async def test_falls_back_to_raw_for_unknown_format(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=httpx.Response(200))

        await post_webhook_alert(
            "https://hooks.example.com/wh",
            7,
            "error",
            "test msg",
            format="nonexistent_format",
            client=mock_client,
        )

        payload = mock_client.post.call_args[1]["json"]
        assert "instance_id" in payload
        assert "timestamp" in payload
        assert "alert_type" in payload
        assert payload["message"] == "test msg"

    @pytest.mark.anyio
    async def test_slack_format_sends_slack_payload(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=httpx.Response(200))

        await post_webhook_alert(
            "https://hooks.slack.com/services/xxx",
            10,
            "low_gpu",
            "GPU idle",
            format="slack",
            client=mock_client,
        )

        payload = mock_client.post.call_args[1]["json"]
        assert "text" in payload
        assert "blocks" in payload

    @pytest.mark.anyio
    async def test_discord_format_sends_discord_payload(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=httpx.Response(200))

        await post_webhook_alert(
            "https://discord.com/api/webhooks/xxx",
            10,
            "error",
            "GPU crash",
            format="discord",
            client=mock_client,
        )

        payload = mock_client.post.call_args[1]["json"]
        assert "embeds" in payload

    @pytest.mark.anyio
    async def test_swallows_post_exception(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        await post_webhook_alert(
            "https://hooks.example.com/wh",
            1,
            "error",
            "msg",
            client=mock_client,
        )

    @pytest.mark.anyio
    async def test_swallows_timeout_exception(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

        await post_webhook_alert(
            "https://hooks.example.com/wh",
            1,
            "error",
            "msg",
            client=mock_client,
        )

    @pytest.mark.anyio
    async def test_passes_metrics_and_instance_through(self):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=httpx.Response(200))
        inst = _make_instance(gpu_name="V100", num_gpus=1, dph_total=0.1)
        metrics = {"util": "50%"}

        await post_webhook_alert(
            "https://hooks.example.com/wh",
            42,
            "low_gpu",
            "check gpu",
            metrics=metrics,
            instance=inst,
            client=mock_client,
        )

        payload = mock_client.post.call_args[1]["json"]
        assert payload["metrics"] == metrics
        assert payload["gpu_name"] == "V100"
