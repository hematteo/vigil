from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from vigil.discovery import (
    DiscoveryResult,
    InstanceInfo,
    RateLimitError,
)
from vigil.providers.vast import VastProvider

_provider = VastProvider()

API_KEY = "test-api-key"


def _mock_response(
    json_data: dict,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Build a real httpx.Response with the given JSON body and status."""
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        headers=headers or {},
        request=httpx.Request("GET", "https://console.vast.ai/api/v0/instances/"),
    )


def _running_instance(**overrides) -> dict:
    """Return a minimal running instance dict, with optional overrides."""
    base = {
        "id": 12345,
        "ssh_host": "ssh5.vast.ai",
        "ssh_port": 22222,
        "gpu_name": "RTX 4090",
        "num_gpus": 2,
        "actual_status": "running",
        "machine_id": 999,
        "label": "my-job",
        "dph_total": 0.50,
    }
    base.update(overrides)
    return base


@pytest.mark.anyio
async def test_fetch_instances_parses_response():
    payload = {"instances": [_running_instance()]}
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(return_value=_mock_response(payload))

    result = await _provider.fetch_instances(API_KEY, client)

    assert len(result.running) == 1
    inst = result.running[0]
    assert isinstance(inst, InstanceInfo)
    assert inst.id == 12345
    assert inst.ssh_host == "ssh5.vast.ai"
    assert inst.ssh_port == 22222
    assert inst.gpu_name == "RTX 4090"
    assert inst.num_gpus == 2
    assert inst.status == "running"
    assert inst.machine_id == 999
    assert inst.label == "my-job"
    assert inst.dph_total == 0.50


@pytest.mark.anyio
async def test_fetch_instances_skips_non_running():
    payload = {
        "instances": [
            _running_instance(id=1, actual_status="exited"),
            _running_instance(id=2, actual_status="running"),
        ]
    }
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(return_value=_mock_response(payload))

    result = await _provider.fetch_instances(API_KEY, client)

    assert len(result.running) == 1
    assert result.running[0].id == 2
    # "exited" is a terminal status — should NOT appear in stuck
    assert len(result.stuck) == 0


@pytest.mark.anyio
async def test_fetch_instances_skips_no_ssh():
    payload = {
        "instances": [
            _running_instance(id=1, ssh_host=""),
            _running_instance(id=2, ssh_host=None),
            _running_instance(id=3, ssh_host="ssh.vast.ai"),
        ]
    }
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(return_value=_mock_response(payload))

    result = await _provider.fetch_instances(API_KEY, client)

    assert len(result.running) == 1
    assert result.running[0].id == 3
    # Instances with empty/null ssh_host go to stuck
    assert len(result.stuck) == 2
    assert {i.id for i in result.stuck} == {1, 2}


@pytest.mark.anyio
async def test_fetch_instances_handles_malformed():
    payload = {
        "instances": [
            {"garbage": True},  # missing required fields
            _running_instance(id=7),
        ]
    }
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(return_value=_mock_response(payload))

    result = await _provider.fetch_instances(API_KEY, client)

    assert len(result.running) == 1
    assert result.running[0].id == 7
    assert len(result.stuck) == 0


@pytest.mark.anyio
async def test_rate_limit_error():
    response = _mock_response(
        json_data={},
        status_code=429,
        headers={"Retry-After": "30"},
    )
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(return_value=response)

    with pytest.raises(RateLimitError) as exc_info:
        await _provider.fetch_instances(API_KEY, client)

    assert exc_info.value.retry_after == 30.0


@pytest.mark.anyio
async def test_rate_limit_error_default():
    response = _mock_response(
        json_data={},
        status_code=429,
        headers={},
    )
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(return_value=response)

    with pytest.raises(RateLimitError) as exc_info:
        await _provider.fetch_instances(API_KEY, client)

    assert exc_info.value.retry_after == 60.0


@pytest.mark.anyio
async def test_fetch_returns_discovery_result():
    payload = {
        "instances": [
            _running_instance(id=1, actual_status="running"),
            _running_instance(id=2, actual_status="exited"),
            _running_instance(id=3, actual_status="loading", ssh_host=""),
            _running_instance(id=4, actual_status="running"),
        ]
    }
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(return_value=_mock_response(payload))

    result = await _provider.fetch_instances(API_KEY, client)

    assert isinstance(result, DiscoveryResult)
    assert len(result.running) == 2
    assert {i.id for i in result.running} == {1, 4}
    # id=2 is "exited" (terminal) — not stuck; id=3 is "loading" with no SSH — stuck
    assert len(result.stuck) == 1
    assert result.stuck[0].id == 3


# ------------------------------------------------------------------
# fetch_credit tests
# ------------------------------------------------------------------


@pytest.mark.anyio
async def test_fetch_credit_success():
    fetch_credit = _provider.fetch_credit

    response = _mock_response({"credit": 42.5}, status_code=200)
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(return_value=response)

    result = await fetch_credit(API_KEY, client)

    assert result == 42.5


@pytest.mark.anyio
async def test_fetch_credit_rate_limit_returns_none():
    fetch_credit = _provider.fetch_credit

    response = _mock_response({}, status_code=429)
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(return_value=response)

    result = await fetch_credit(API_KEY, client)

    assert result is None


@pytest.mark.anyio
async def test_fetch_credit_error_returns_none():
    fetch_credit = _provider.fetch_credit

    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    result = await fetch_credit(API_KEY, client)

    assert result is None


@pytest.mark.anyio
async def test_fetch_credit_http_error_returns_none():
    fetch_credit = _provider.fetch_credit

    response = _mock_response({"error": "forbidden"}, status_code=403)
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(return_value=response)

    result = await fetch_credit(API_KEY, client)

    assert result is None


@pytest.mark.anyio
async def test_fetch_credit_missing_credit_field():
    fetch_credit = _provider.fetch_credit

    response = _mock_response({"username": "test"}, status_code=200)
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(return_value=response)

    result = await fetch_credit(API_KEY, client)

    assert result == 0.0


# ------------------------------------------------------------------
# destroy_instance tests
# ------------------------------------------------------------------


@pytest.mark.anyio
async def test_destroy_instance_success():
    destroy_instance = _provider.destroy_instance

    response = httpx.Response(
        status_code=200,
        json={"success": True},
        request=httpx.Request("DELETE", "https://console.vast.ai/api/v0/instances/123/"),
    )
    client = AsyncMock(spec=httpx.AsyncClient)
    client.delete = AsyncMock(return_value=response)

    await destroy_instance(API_KEY, 123, client)

    client.delete.assert_called_once()
    call_args = client.delete.call_args
    assert "instances/123/" in call_args.args[0]


@pytest.mark.anyio
async def test_destroy_instance_raises_on_http_error():
    destroy_instance = _provider.destroy_instance

    response = httpx.Response(
        status_code=500,
        json={"error": "internal"},
        request=httpx.Request("DELETE", "https://console.vast.ai/api/v0/instances/456/"),
    )
    client = AsyncMock(spec=httpx.AsyncClient)
    client.delete = AsyncMock(return_value=response)

    with pytest.raises(httpx.HTTPStatusError):
        await destroy_instance(API_KEY, 456, client)


@pytest.mark.anyio
async def test_fetch_instances_parses_start_date():
    payload = {"instances": [_running_instance(start_date=1700000000.5), _running_instance(id=2, start_date=None)]}
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(return_value=_mock_response(payload))

    result = await _provider.fetch_instances(API_KEY, client)

    assert result.running[0].start_time == 1700000000.5
    assert result.running[1].start_time is None


@pytest.mark.anyio
async def test_runpod_start_time_from_uptime():
    from vigil.providers.runpod import RunPodProvider

    pod = {
        "id": "abc123xyz",
        "name": "sweep-7",
        "desiredStatus": "RUNNING",
        "costPerHr": 0.69,
        "gpuCount": 1,
        "machineId": "m1",
        "runtime": {
            "uptimeInSeconds": 3600,
            "ports": [{"ip": "5.6.7.8", "isIpPublic": True, "privatePort": 22, "publicPort": 40022, "type": "tcp"}],
        },
        "machine": {"gpuDisplayName": "RTX 4090"},
    }
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=_mock_response({"data": {"myself": {"pods": [pod]}}}))

    with patch("vigil.providers.runpod.time.time", return_value=10_000.0):
        result = await RunPodProvider().fetch_instances(API_KEY, client)

    inst = result.running[0]
    assert inst.id == "abc123xyz"
    assert inst.ssh_port == 40022
    assert inst.start_time == 6_400.0


@pytest.mark.anyio
async def test_runpod_sends_key_in_header_not_url():
    from vigil.providers.runpod import RunPodProvider

    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=_mock_response({"data": {"myself": {"pods": [], "clientBalance": 1.0}}}))
    provider = RunPodProvider()

    await provider.fetch_instances("rp-secret-key", client)
    await provider.fetch_credit("rp-secret-key", client)
    await provider.destroy_instance("rp-secret-key", "pod1", client)

    for call in client.post.call_args_list:
        assert "rp-secret-key" not in call.args[0]
        assert call.kwargs["headers"] == {"Authorization": "Bearer rp-secret-key"}


def test_redact():
    from vigil.discovery import redact

    msg = "Client error '401' for url 'https://api.runpod.io/graphql?api_key=rp-secret-key'"
    assert "rp-secret-key" not in redact(msg, "rp-secret-key")
    assert redact(msg, "") == msg
    assert redact("abc", "a") == "abc"  # too short to redact safely
