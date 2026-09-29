from __future__ import annotations

from dataclasses import dataclass


def redact(text: str, secret: str | None) -> str:
    """Remove *secret* (e.g. an API key) from text that is about to be shown or logged."""
    if secret and len(secret) >= 4:
        return text.replace(secret, "***")
    return text


class RateLimitError(Exception):
    def __init__(self, retry_after: float = 60.0):
        self.retry_after = retry_after


@dataclass
class InstanceInfo:
    id: int | str
    ssh_host: str
    ssh_port: int
    gpu_name: str
    num_gpus: int
    status: str
    machine_id: int = 0
    label: str | None = None
    dph_total: float = 0.0
    start_time: float | None = None  # Unix time the instance started, if the provider reports it


@dataclass
class DiscoveryResult:
    running: list[InstanceInfo]
    stuck: list[InstanceInfo]
