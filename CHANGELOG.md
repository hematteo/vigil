# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `vigil watch`: headless hang watchdog for always-on machines. Combines log silence with `nvidia-smi` sampling to alert on hung, exited, silently-busy (possible NCCL deadlock), unreachable and preempted instances, with the cost of the silence in every alert, reminders, and recovery notices
- `ntfy` webhook format for phone push notifications
- Instance start time from the Vast.ai and RunPod APIs, used for total cost in alerts

- Provider abstraction layer — vigil now supports multiple GPU cloud providers
- RunPod provider with full support (auto-discovery, cost tracking, instance management)
- `--provider` flag now accepts `runpod` in addition to `vast`
- `api_key` config field replaces `vast_api_key` (backward compatible — existing configs still work)
- `--demo` mode for showcasing the TUI without a Vast.ai account or running instances
- `--provider` CLI flag (currently: `vast`)
- `VIGIL_API_KEY` env var support (alongside existing `VAST_API_KEY`)
- Config-driven metric semantics: `decrease_good`, `increase_good`, `counters`, `plateau_metrics`
- `extra_metric_patterns` for appending patterns without replacing defaults
- SSH username configurable globally and per-instance
- Per-alert notification control (NaN, plateau, stall, desktop, webhook independently togglable)
- Slack and Discord webhook formats (`alert_webhook_format: "slack" | "discord" | "raw"`)
- NaN and plateau alerts now fire webhooks (previously only stall did)
- Instance manager screen (`i` key) with GPU details, costs, and SSH commands
- Log rotation with configurable retention (`log_retention_days`, `log_max_size_mb`)
- Exposed hardcoded values as config: `max_grid_columns`, `log_buffer_lines`, `log_display_lines`, `ssh_login_timeout`, `ssh_keepalive_interval`, `reconnect_backoff_max`
- Setup wizard (`S` key) with API key validation, SSH key auto-detection, framework presets (HuggingFace, PyTorch Lightning, Dreamer/RL), and alert configuration
- Framework auto-detection by scanning running instance logs
- Contextual onboarding hints (non-intrusive, dismissable, persistent)
- `--reset-hints` CLI flag to re-show onboarding hints

### Changed

- Webhook alert titles say "vigil alert" instead of "Vast.ai Alert" (they were wrong for RunPod)
- `--provider` now defaults to the `provider` set in the config file instead of always `vast`

### Fixed

- Tab focus now correctly sets `_focused_id` (unifies Tab/click/1-9)
- Alert borders auto-clear when NaN/plateau conditions resolve
- NaN alert deduplication prevents notification spam
- Panel `[N]` labels update correctly after instance removal
- `_set_if_present` respects falsy values like `0` in config
- Atomic config writes prevent corruption on crash
- Global search shows truncation indicator
- Double-press panel number always unfocuses (even with search open)

## [0.1.0] - 2026-03-25

### Added

- Auto-discovery of running GPU instances via provider APIs (Vast.ai)
- Real-time SSH log streaming with exponential backoff reconnection
- Configurable regex-based metric parsing (loss, step, epoch, lr, reward, accuracy, and more)
- Unicode sparkline visualization per metric with directional color coding
- NaN/Inf detection with alert notification
- Loss plateau detection (configurable window and threshold)
- Stall detection with webhook alerts
- Desktop notifications (macOS and Linux)
- Local log persistence with crash resilience
- Historical log browser with search
- Metrics comparison table with auto-refresh and color coding
- Global cross-instance search (live, regex-supported)
- Inline log filtering with debounce
- nvidia-smi remote execution
- Instance destruction with confirmation
- Dynamic grid layout (auto-adjusts 1-3 columns)
- Panel focus/zoom (1-9 keys)
- Configurable per-instance log commands and stall thresholds
- Config persistence (runtime changes saved to YAML)
- Comprehensive config.example.yaml documenting all options
