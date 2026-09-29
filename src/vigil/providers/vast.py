from __future__ import annotations

from pathlib import Path

import httpx

from ..config import DEFAULT_LOG_COMMAND
from ..discovery import DiscoveryResult, InstanceInfo, RateLimitError

VAST_API_BASE = "https://console.vast.ai/api/v0"


def _start_time(value: object) -> float | None:
    try:
        ts = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return ts if ts > 0 else None


class VastProvider:
    name: str = "vast"
    display_name: str = "Vast.ai"

    async def fetch_instances(self, api_key: str, client: httpx.AsyncClient) -> DiscoveryResult:
        resp = await client.get(
            f"{VAST_API_BASE}/instances/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "60"))
            raise RateLimitError(retry_after)
        resp.raise_for_status()
        data = resp.json()

        running: list[InstanceInfo] = []
        stuck: list[InstanceInfo] = []
        for inst in data.get("instances", []):
            try:
                is_running = inst.get("actual_status") == "running"
                has_ssh = bool(inst.get("ssh_host")) and bool(inst.get("ssh_port"))
                info = InstanceInfo(
                    id=inst["id"],
                    ssh_host=inst.get("ssh_host") or "",
                    ssh_port=int(inst.get("ssh_port") or 0),
                    gpu_name=inst.get("gpu_name", "Unknown GPU"),
                    num_gpus=int(inst.get("num_gpus", 1)),
                    status=inst.get("actual_status", "unknown"),
                    machine_id=int(inst.get("machine_id", 0)),
                    label=inst.get("label"),
                    dph_total=float(inst.get("dph_total", 0.0)),
                    start_time=_start_time(inst.get("start_date")),
                )
                terminal = info.status in {"exited", "destroyed", "deleting", "error"}
                if is_running and has_ssh:
                    running.append(info)
                elif not terminal:
                    stuck.append(info)
            except (KeyError, ValueError, TypeError):
                continue

        return DiscoveryResult(running=running, stuck=stuck)

    async def fetch_credit(self, api_key: str, client: httpx.AsyncClient) -> float | None:
        try:
            resp = await client.get(
                f"{VAST_API_BASE}/users/current/",
                params={"owner": "me"},
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=httpx.Timeout(10.0, connect=5.0),
            )
            if resp.status_code == 429:
                return None
            resp.raise_for_status()
            return float(resp.json().get("credit", 0.0))
        except Exception:
            return None

    async def destroy_instance(self, api_key: str, instance_id: int | str, client: httpx.AsyncClient) -> None:
        resp = await client.delete(
            f"{VAST_API_BASE}/instances/{instance_id}/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        resp.raise_for_status()

    def env_var_names(self) -> list[str]:
        return ["VAST_API_KEY"]

    def api_key_file(self) -> Path | None:
        return Path.home() / ".vast_api_key"

    def default_ssh_key_path(self) -> Path:
        return Path.home() / ".ssh" / "id_vastai"

    def default_ssh_username(self) -> str:
        return "root"

    def default_log_command(self) -> str:
        return DEFAULT_LOG_COMMAND
