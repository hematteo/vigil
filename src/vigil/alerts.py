from __future__ import annotations

import time
from typing import Any

import httpx

from .discovery import InstanceInfo


def _format_raw(
    instance_id: int | str,
    alert_type: str,
    message: str,
    metrics: dict[str, str] | None = None,
    instance: InstanceInfo | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp": time.time(),
        "instance_id": instance_id,
        "alert_type": alert_type,
        "message": message,
    }
    if metrics is not None:
        payload["metrics"] = metrics
    if instance:
        payload["gpu_name"] = instance.gpu_name
        payload["num_gpus"] = instance.num_gpus
        payload["dph_total"] = instance.dph_total
    return payload


def _format_slack(
    instance_id: int | str,
    alert_type: str,
    message: str,
    metrics: dict[str, str] | None = None,
    instance: InstanceInfo | None = None,
) -> dict[str, Any]:
    title = f"vigil alert: #{instance_id}"
    details = message
    if instance:
        details += f"\nGPU: {instance.gpu_name} x{instance.num_gpus} | ${instance.dph_total:.3f}/hr"
    if metrics is not None:
        metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
        details += f"\nMetrics: {metric_str}"

    return {
        "text": title,
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{alert_type.upper()}* — Instance #{instance_id}\n{details}",
                },
            }
        ],
    }


def _format_discord(
    instance_id: int | str,
    alert_type: str,
    message: str,
    metrics: dict[str, str] | None = None,
    instance: InstanceInfo | None = None,
) -> dict[str, Any]:
    # Green for recoveries, orange for known warning types, red for everything else (errors + unknown)
    _WARN_TYPES = {"low_gpu", "slow", "busy_silent", "unreachable"}
    if alert_type == "recovered":
        color = 3066993
    elif alert_type in _WARN_TYPES:
        color = 16744448
    else:
        color = 16711680
    fields = []
    if instance:
        fields.append({"name": "GPU", "value": f"{instance.gpu_name} x{instance.num_gpus}", "inline": True})
        fields.append({"name": "Cost", "value": f"${instance.dph_total:.3f}/hr", "inline": True})
    if metrics is not None:
        metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
        fields.append({"name": "Metrics", "value": metric_str, "inline": False})

    return {
        "embeds": [
            {
                "title": f"vigil alert: #{instance_id}",
                "description": message,
                "color": color,
                "fields": fields,
                "footer": {"text": alert_type},
            }
        ]
    }


def _ntfy_request(
    instance_id: int | str,
    alert_type: str,
    message: str,
    instance: InstanceInfo | None = None,
) -> tuple[str, dict[str, str]]:
    """Build an ntfy.sh publish: plain-text body plus Title/Tags/Priority headers."""
    title = f"vigil #{instance_id}: {alert_type}"
    if instance and instance.label:
        title += f" ({instance.label})"
    # HTTP headers must be latin-1; drop anything else rather than fail the send
    title = title.encode("ascii", errors="ignore").decode()
    if alert_type == "recovered":
        tags, priority = "white_check_mark", "default"
    else:
        tags, priority = "warning", "high"
    return message, {"Title": title, "Tags": tags, "Priority": priority}


_FORMATTERS = {
    "raw": _format_raw,
    "slack": _format_slack,
    "discord": _format_discord,
}


async def post_webhook_alert(
    url: str,
    instance_id: int | str,
    alert_type: str,
    message: str,
    metrics: dict[str, str] | None = None,
    instance: InstanceInfo | None = None,
    format: str = "raw",
    client: httpx.AsyncClient | None = None,
) -> None:
    """POST an alert payload to a webhook URL. Best-effort, never raises."""
    if format == "ntfy":
        body, headers = _ntfy_request(instance_id, alert_type, message, instance)
        kwargs: dict[str, Any] = {"content": body.encode(), "headers": headers}
    else:
        formatter = _FORMATTERS.get(format, _format_raw)
        kwargs = {"json": formatter(instance_id, alert_type, message, metrics, instance)}

    try:
        if client is None:
            async with httpx.AsyncClient() as c:
                await c.post(url, timeout=10.0, **kwargs)
        else:
            await client.post(url, timeout=10.0, **kwargs)
    except Exception:
        pass  # Best effort — never disrupt the app
