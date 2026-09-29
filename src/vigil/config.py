from __future__ import annotations

import contextlib
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "vigil" / "config.yaml"
DEFAULT_LOG_DIR = Path.home() / ".vigil-logs"
DEFAULT_API_KEY_PATH = Path.home() / ".vast_api_key"

DEFAULT_LOG_COMMAND = (
    "bash -c '"
    # 1. Find training process stdout → if it points to a file, tail it
    'for PID in $(pgrep -f "python" 2>/dev/null | head -5); do '
    "TARGET=$(readlink /proc/$PID/fd/1 2>/dev/null); "
    '[ -f "$TARGET" ] && exec tail -n 100 -f "$TARGET"; '
    "done; "
    # 2. Fall back to most recently modified log/txt file
    'LOG=$(find /root /workspace -maxdepth 6 \\( -name "*.log" -o -name "*.txt" \\) '
    "-newer /proc/1/cmdline 2>/dev/null | xargs ls -t 2>/dev/null | head -1); "
    '[ -n "$LOG" ] && exec tail -n 100 -f "$LOG"; '
    # 3. Last resort
    "tail -n 100 -f /var/log/vastai/onstart.log 2>/dev/null || "
    'echo "No log files found. Set log_command in config."'
    "'"
)

DEFAULT_METRIC_PATTERNS: list[str] = [
    r"loss[:\s=]+(?P<loss>[\d.]+)",
    r"step[:\s=]+(?P<step>\d+(?:,\d{3})*)",
    r"epoch[:\s=]+(?P<epoch>[\d./]+)",
    r"(?<![a-z])lr[:\s=]+(?P<lr>[\d.e\-]+)",
    r"reward[:\s=]+(?P<reward>[\d.\-e]+)",
    r"accuracy[:\s=]+(?P<accuracy>[\d.]+)",
    r"rssm[:\s=]+(?P<rssm>[\d.\-e]+)",
    r"(?<![a-z])vc[:\s=]+(?P<vc>[\d.\-e]+)",
    r"(?<![a-z])tom[:\s=]+(?P<tom>[\d.\-e]+)",
    r"actor[:\s=]+(?P<actor>[\d.\-e]+)",
]


@dataclass
class NotificationConfig:
    nan: bool = True
    plateau: bool = True
    stall: bool = True
    desktop: bool = True
    webhook: bool = True


@dataclass
class InstanceConfig:
    log_command: str | None = None
    stall_threshold_minutes: int | None = None
    ssh_username: str | None = None
    watch: bool | None = None  # False = `vigil watch` ignores this instance


@dataclass
class Config:
    api_key: str = ""
    provider: str = "vast"
    poll_interval: int = 30
    log_dir: Path = field(default_factory=lambda: DEFAULT_LOG_DIR)
    default_log_command: str = DEFAULT_LOG_COMMAND
    ssh_key_path: Path = field(default_factory=lambda: Path.home() / ".ssh" / "id_vastai")
    instances: dict[str, InstanceConfig] = field(default_factory=dict)
    metric_patterns: list[str] = field(
        default_factory=lambda: DEFAULT_METRIC_PATTERNS.copy()
    )
    stall_threshold_minutes: int = 5
    plateau_window: int = 8
    plateau_threshold: float = 1e-4
    sparkline_history: int = 60
    desktop_notifications: bool = True
    alert_webhook_url: str | None = None
    alert_webhook_format: str = "raw"
    highlight_logs: bool = False
    decrease_good: set[str] = field(default_factory=lambda: {"loss", "rssm", "vc", "tom"})
    increase_good: set[str] = field(default_factory=lambda: {"reward", "accuracy"})
    counters: set[str] = field(default_factory=lambda: {"step", "epoch"})
    plateau_metrics: list[str] = field(default_factory=lambda: ["loss"])
    extra_metric_patterns: list[str] = field(default_factory=list)
    ssh_username: str = "root"
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    max_grid_columns: int = 3
    log_buffer_lines: int = 5000
    log_display_lines: int = 2000
    ssh_login_timeout: int = 10
    ssh_keepalive_interval: int = 15
    reconnect_backoff_max: int = 60
    log_retention_days: int = 0       # 0 = keep forever
    log_max_size_mb: int = 0          # 0 = unlimited
    # `vigil watch` (headless hang watchdog)
    watch_gpu_poll_seconds: int = 60
    watch_gpu_idle_percent: float = 5.0
    watch_busy_silence_minutes: int = 30
    watch_reminder_minutes: int = 60  # 0 = alert once per episode
    watch_skip_labels: list[str] = field(default_factory=list)  # label substrings `vigil watch` ignores
    _config_path: Path | None = field(default=None, repr=False, compare=False)

    def log_command_for(self, instance_id: int | str) -> str:
        inst = self.instances.get(str(instance_id))
        if inst and inst.log_command:
            return inst.log_command
        return self.default_log_command

    def stall_threshold_for(self, instance_id: int | str) -> int:
        inst = self.instances.get(str(instance_id))
        if inst and inst.stall_threshold_minutes is not None:
            return inst.stall_threshold_minutes
        return self.stall_threshold_minutes

    def ssh_username_for(self, instance_id: int | str) -> str:
        inst = self.instances.get(str(instance_id))
        if inst and inst.ssh_username:
            return inst.ssh_username
        return self.ssh_username

    def should_watch(self, instance_id: int | str, label: str | None) -> bool:
        """Whether `vigil watch` should monitor this instance."""
        inst = self.instances.get(str(instance_id))
        if inst and inst.watch is not None:
            return inst.watch
        lowered = (label or "").lower()
        return not any(skip.lower() in lowered for skip in self.watch_skip_labels if skip)

    def save_config(self, path: Path | None = None) -> None:
        """Write current config to YAML, preserving only non-default fields."""
        target = path or self._config_path or DEFAULT_CONFIG_PATH
        target.parent.mkdir(parents=True, exist_ok=True)

        defaults = Config()
        data: dict = {}

        if self.api_key and self.api_key != defaults.api_key:
            data["api_key"] = self.api_key
        if self.provider != defaults.provider:
            data["provider"] = self.provider
        if self.poll_interval != defaults.poll_interval:
            data["poll_interval"] = self.poll_interval
        if str(self.log_dir) != str(defaults.log_dir):
            data["log_dir"] = str(self.log_dir)
        if str(self.ssh_key_path) != str(defaults.ssh_key_path):
            data["ssh_key_path"] = str(self.ssh_key_path)
        if self.default_log_command != defaults.default_log_command:
            data["default_log_command"] = self.default_log_command
        if self.stall_threshold_minutes != defaults.stall_threshold_minutes:
            data["stall_threshold_minutes"] = self.stall_threshold_minutes
        if self.plateau_window != defaults.plateau_window:
            data["plateau_window"] = self.plateau_window
        if self.plateau_threshold != defaults.plateau_threshold:
            data["plateau_threshold"] = self.plateau_threshold
        if self.sparkline_history != defaults.sparkline_history:
            data["sparkline_history"] = self.sparkline_history
        # Note: desktop_notifications is legacy — we write the canonical
        # notifications block instead (see below). Do NOT write the legacy field.
        if self.alert_webhook_url:
            data["alert_webhook_url"] = self.alert_webhook_url
        if self.alert_webhook_format != defaults.alert_webhook_format:
            data["alert_webhook_format"] = self.alert_webhook_format
        if self.highlight_logs != defaults.highlight_logs:
            data["highlight_logs"] = self.highlight_logs
        if self.metric_patterns != defaults.metric_patterns:
            data["metric_patterns"] = self.metric_patterns
        if self.decrease_good != defaults.decrease_good:
            data["decrease_good"] = sorted(self.decrease_good)
        if self.increase_good != defaults.increase_good:
            data["increase_good"] = sorted(self.increase_good)
        if self.counters != defaults.counters:
            data["counters"] = sorted(self.counters)
        if self.plateau_metrics != defaults.plateau_metrics:
            data["plateau_metrics"] = self.plateau_metrics
        if self.extra_metric_patterns:
            data["extra_metric_patterns"] = self.extra_metric_patterns
        if self.ssh_username != defaults.ssh_username:
            data["ssh_username"] = self.ssh_username
        if self.max_grid_columns != defaults.max_grid_columns:
            data["max_grid_columns"] = self.max_grid_columns
        if self.log_buffer_lines != defaults.log_buffer_lines:
            data["log_buffer_lines"] = self.log_buffer_lines
        if self.log_display_lines != defaults.log_display_lines:
            data["log_display_lines"] = self.log_display_lines
        if self.ssh_login_timeout != defaults.ssh_login_timeout:
            data["ssh_login_timeout"] = self.ssh_login_timeout
        if self.ssh_keepalive_interval != defaults.ssh_keepalive_interval:
            data["ssh_keepalive_interval"] = self.ssh_keepalive_interval
        if self.reconnect_backoff_max != defaults.reconnect_backoff_max:
            data["reconnect_backoff_max"] = self.reconnect_backoff_max
        if self.log_retention_days != defaults.log_retention_days:
            data["log_retention_days"] = self.log_retention_days
        if self.log_max_size_mb != defaults.log_max_size_mb:
            data["log_max_size_mb"] = self.log_max_size_mb
        for key in (
            "watch_gpu_poll_seconds", "watch_gpu_idle_percent",
            "watch_busy_silence_minutes", "watch_reminder_minutes",
        ):
            if getattr(self, key) != getattr(defaults, key):
                data[key] = getattr(self, key)
        if self.watch_skip_labels:
            data["watch_skip_labels"] = self.watch_skip_labels

        nc_defaults = NotificationConfig()
        nc_data = {}
        for key in ("nan", "plateau", "stall", "desktop", "webhook"):
            if getattr(self.notifications, key) != getattr(nc_defaults, key):
                nc_data[key] = getattr(self.notifications, key)
        if nc_data:
            data["notifications"] = nc_data

        if self.instances:
            instances_data: dict = {}
            for iid, icfg in self.instances.items():
                inst_data: dict = {}
                if icfg.log_command:
                    inst_data["log_command"] = icfg.log_command
                if icfg.stall_threshold_minutes is not None:
                    inst_data["stall_threshold_minutes"] = icfg.stall_threshold_minutes
                if icfg.ssh_username:
                    inst_data["ssh_username"] = icfg.ssh_username
                if icfg.watch is not None:
                    inst_data["watch"] = icfg.watch
                if inst_data:
                    instances_data[str(iid)] = inst_data
            if instances_data:
                data["instances"] = instances_data

        # Atomic write: write to temp file, then rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(target.parent), suffix=".tmp", prefix=".config_"
        )
        replaced = False
        try:
            with os.fdopen(fd, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            os.replace(tmp_path, target)
            replaced = True
        except BaseException:
            if not replaced:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
            raise


def _parse_instance(idata: dict) -> InstanceConfig:
    stall = idata.get("stall_threshold_minutes")
    return InstanceConfig(
        log_command=idata.get("log_command"),
        stall_threshold_minutes=int(stall) if stall is not None else None,
        ssh_username=idata.get("ssh_username"),
        watch=bool(idata["watch"]) if idata.get("watch") is not None else None,
    )


def _set_if_present(obj: object, data: dict, keys: tuple[str, ...], cast: type) -> None:
    """Set attributes on *obj* for each key present in *data* (respects falsy values like 0)."""
    for key in keys:
        if key in data and data[key] is not None:
            setattr(obj, key, cast(data[key]))


def _apply_yaml_fields(config: Config, data: dict) -> None:
    """Apply top-level YAML fields onto an existing Config."""
    if "vast_api_key" in data and "api_key" not in data:
        data["api_key"] = data["vast_api_key"]

    _set_if_present(
        config, data,
        ("api_key", "provider", "default_log_command", "alert_webhook_url", "ssh_username", "alert_webhook_format"),
        str,
    )
    _set_if_present(config, data, (
        "poll_interval", "stall_threshold_minutes", "plateau_window", "sparkline_history",
        "max_grid_columns", "log_buffer_lines", "log_display_lines",
        "ssh_login_timeout", "ssh_keepalive_interval", "reconnect_backoff_max",
        "log_retention_days", "log_max_size_mb",
        "watch_gpu_poll_seconds", "watch_busy_silence_minutes", "watch_reminder_minutes",
    ), int)
    _set_if_present(config, data, ("plateau_threshold", "watch_gpu_idle_percent"), float)

    # Path fields need expanduser()
    for key in ("log_dir", "ssh_key_path"):
        if data.get(key):
            setattr(config, key, Path(data[key]).expanduser())

    # Bool fields must check "in" so False values are respected
    for key in ("desktop_notifications", "highlight_logs"):
        if key in data:
            setattr(config, key, bool(data[key]))

    has_explicit_patterns = "metric_patterns" in data and isinstance(data["metric_patterns"], list)
    if has_explicit_patterns:
        config.metric_patterns = [str(p) for p in data["metric_patterns"]]

    if "extra_metric_patterns" in data and isinstance(data["extra_metric_patterns"], list):
        extras = [str(p) for p in data["extra_metric_patterns"]]
        config.extra_metric_patterns = extras
        # Only extend if metric_patterns was NOT explicitly set in the same file
        # (if it was, it already includes the extras from a prior save)
        if not has_explicit_patterns:
            config.metric_patterns.extend(extras)

    for key in ("decrease_good", "increase_good", "counters"):
        if key in data and isinstance(data[key], list):
            setattr(config, key, {str(v) for v in data[key]})

    if "watch_skip_labels" in data and isinstance(data["watch_skip_labels"], list):
        config.watch_skip_labels = [str(v) for v in data["watch_skip_labels"]]

    if "plateau_metrics" in data and isinstance(data["plateau_metrics"], list):
        config.plateau_metrics = [str(v) for v in data["plateau_metrics"]]

    for iid, idata in (data.get("instances") or {}).items():
        if isinstance(idata, dict):
            config.instances[str(iid)] = _parse_instance(idata)

    # Notification config
    if "notifications" in data and isinstance(data["notifications"], dict):
        nc = data["notifications"]
        for key in ("nan", "plateau", "stall", "desktop", "webhook"):
            if key in nc:
                setattr(config.notifications, key, bool(nc[key]))

    # Backward compat: desktop_notifications overrides notifications.desktop
    # The existing desktop_notifications bool parsing is already handled above.
    # Apply it to notifications.desktop if notifications block was NOT provided:
    if "desktop_notifications" in data and "notifications" not in data:
        config.notifications.desktop = bool(data["desktop_notifications"])


def _resolve_api_key(config: Config, provider=None) -> None:
    """Apply env-var and file-based fallbacks for the API key."""
    env_key = os.environ.get("VIGIL_API_KEY")
    if not env_key and provider:
        for var in provider.env_var_names():
            env_key = os.environ.get(var)
            if env_key:
                break
    if env_key:
        config.api_key = env_key
    if not config.api_key:
        key_file = provider.api_key_file() if provider else DEFAULT_API_KEY_PATH
        if key_file and key_file.exists():
            config.api_key = key_file.read_text().strip()


def load_config(config_path: Path | None = None, provider=None) -> Config:
    config = Config()

    path = config_path or DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _apply_yaml_fields(config, data)

    _resolve_api_key(config, provider)
    config._config_path = path
    return config
