from __future__ import annotations

import time
from pathlib import Path

import httpx

from ..discovery import DiscoveryResult, InstanceInfo, RateLimitError

RUNPOD_API_BASE = "https://api.runpod.io/graphql"

_PODS_QUERY = """
query {
  myself {
    pods {
      id
      name
      desiredStatus
      costPerHr
      gpuCount
      machineId
      runtime {
        uptimeInSeconds
        ports {
          ip
          isIpPublic
          privatePort
          publicPort
          type
        }
      }
      machine {
        gpuDisplayName
      }
    }
  }
}
"""

_CREDIT_QUERY = """
query {
  myself {
    clientBalance
  }
}
"""

_TERMINATE_MUTATION = """
mutation ($id: String!) {
  podTerminate(input: { podId: $id })
}
"""

_DEFAULT_RUNPOD_LOG_COMMAND = (
    "bash -c '"
    'for PID in $(pgrep -f "python" 2>/dev/null | head -5); do '
    "TARGET=$(readlink /proc/$PID/fd/1 2>/dev/null); "
    '[ -f "$TARGET" ] && exec tail -n 100 -f "$TARGET"; '
    "done; "
    'LOG=$(find /root /workspace -maxdepth 6 \\( -name "*.log" -o -name "*.txt" \\) '
    "-newer /proc/1/cmdline 2>/dev/null | xargs ls -t 2>/dev/null | head -1); "
    '[ -n "$LOG" ] && exec tail -n 100 -f "$LOG"; '
    'echo "No log files found. Set log_command in config."'
    "'"
)


def _auth_headers(api_key: str) -> dict[str, str]:
    # Header rather than the ?api_key= query param, so the key never appears in
    # URLs (which httpx includes in error messages that end up in logs).
    return {"Authorization": f"Bearer {api_key}"}


def _check_graphql_errors(body: dict) -> None:
    errors = body.get("errors")
    if errors:
        msg = errors[0].get("message", "Unknown GraphQL error")
        raise RuntimeError(f"RunPod GraphQL error: {msg}")


class RunPodProvider:
    name: str = "runpod"
    display_name: str = "RunPod"

    async def fetch_instances(self, api_key: str, client: httpx.AsyncClient) -> DiscoveryResult:
        resp = await client.post(
            RUNPOD_API_BASE,
            headers=_auth_headers(api_key),
            json={"query": _PODS_QUERY},
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "60"))
            raise RateLimitError(retry_after)
        resp.raise_for_status()
        body = resp.json()
        _check_graphql_errors(body)

        pods = body.get("data", {}).get("myself", {}).get("pods", [])
        running: list[InstanceInfo] = []
        stuck: list[InstanceInfo] = []

        for pod in pods:
            try:
                desired = pod.get("desiredStatus", "")
                # Skip terminal states
                if desired not in ("RUNNING", "CREATED"):
                    continue

                # Extract SSH port info
                ssh_host = ""
                ssh_port = 0
                ports = (pod.get("runtime") or {}).get("ports") or []
                for port_info in ports:
                    if port_info.get("privatePort") == 22 and port_info.get("isIpPublic") is True:
                        ssh_host = port_info.get("ip", "")
                        ssh_port = int(port_info.get("publicPort", 0))
                        break

                has_ssh = bool(ssh_host) and ssh_port > 0

                start_time = None
                uptime = (pod.get("runtime") or {}).get("uptimeInSeconds")
                if isinstance(uptime, (int, float)) and uptime > 0:
                    start_time = time.time() - uptime

                machine = pod.get("machine") or {}

                try:
                    machine_id = int(pod.get("machineId", 0))
                except (ValueError, TypeError):
                    machine_id = 0

                info = InstanceInfo(
                    id=pod["id"],
                    ssh_host=ssh_host,
                    ssh_port=ssh_port,
                    gpu_name=machine.get("gpuDisplayName", "Unknown GPU"),
                    num_gpus=int(pod.get("gpuCount", 1)),
                    status=desired.lower(),
                    machine_id=machine_id,
                    label=pod.get("name"),
                    dph_total=float(pod.get("costPerHr", 0.0)),
                    start_time=start_time,
                )

                if desired == "RUNNING" and has_ssh:
                    running.append(info)
                else:
                    stuck.append(info)
            except (KeyError, ValueError, TypeError):
                continue

        return DiscoveryResult(running=running, stuck=stuck)

    async def fetch_credit(self, api_key: str, client: httpx.AsyncClient) -> float | None:
        try:
            resp = await client.post(
                RUNPOD_API_BASE,
                headers=_auth_headers(api_key),
                json={"query": _CREDIT_QUERY},
                timeout=httpx.Timeout(10.0, connect=5.0),
            )
            if resp.status_code == 429:
                return None
            resp.raise_for_status()
            body = resp.json()
            _check_graphql_errors(body)
            return float(body.get("data", {}).get("myself", {}).get("clientBalance", 0.0))
        except Exception:
            return None

    async def destroy_instance(self, api_key: str, instance_id: int | str, client: httpx.AsyncClient) -> None:
        resp = await client.post(
            RUNPOD_API_BASE,
            headers=_auth_headers(api_key),
            json={"query": _TERMINATE_MUTATION, "variables": {"id": str(instance_id)}},
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "60"))
            raise RateLimitError(retry_after)
        resp.raise_for_status()
        body = resp.json()
        _check_graphql_errors(body)

    def env_var_names(self) -> list[str]:
        return ["RUNPOD_API_KEY"]

    def api_key_file(self) -> Path | None:
        return None

    def default_ssh_key_path(self) -> Path:
        return Path.home() / ".ssh" / "id_ed25519"

    def default_ssh_username(self) -> str:
        return "root"

    def default_log_command(self) -> str:
        return _DEFAULT_RUNPOD_LOG_COMMAND
