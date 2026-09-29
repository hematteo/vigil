from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, Config, load_config
from .providers import Provider, get_provider

WEBHOOK_FORMATS = ["raw", "slack", "discord", "ntfy"]


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help=f"Config file path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Provider API key (overrides config/env)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["vast", "runpod"],
        help="GPU cloud provider (default: from config, else vast)",
    )


def _load(args: argparse.Namespace) -> tuple[Config, Provider]:
    config = load_config(args.config)
    if args.provider:
        config.provider = args.provider
    try:
        provider = get_provider(config.provider)
    except ValueError as exc:
        sys.exit(f"vigil: {exc}")

    # Re-resolve API key now that provider is known (checks RUNPOD_API_KEY, etc.)
    if not config.api_key:
        from .config import _resolve_api_key
        _resolve_api_key(config, provider)

    if args.api_key:
        config.api_key = args.api_key
    return config, provider


def _watch_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="vigil watch",
        description=(
            "Headless hang watchdog. Streams logs from every running instance and alerts "
            "when output stops, using GPU utilization to tell hung, exited and busy runs apart."
        ),
    )
    _add_common_args(parser)
    parser.add_argument("--webhook", type=str, default=None, help="Alert webhook URL (overrides config)")
    parser.add_argument(
        "--webhook-format",
        type=str,
        default=None,
        choices=WEBHOOK_FORMATS,
        help="Webhook payload format (overrides config)",
    )
    parser.add_argument(
        "--stall-minutes",
        type=int,
        default=None,
        help="Minutes without output before a run is checked for a hang (overrides config)",
    )
    parser.add_argument(
        "--test-alert",
        action="store_true",
        help="Send one test alert to the configured webhook and exit",
    )
    args = parser.parse_args(argv)

    config, provider = _load(args)
    if args.webhook:
        config.alert_webhook_url = args.webhook
    if args.webhook_format:
        config.alert_webhook_format = args.webhook_format
    if args.stall_minutes is not None:
        config.stall_threshold_minutes = args.stall_minutes

    if args.test_alert:
        if not config.alert_webhook_url:
            sys.exit("vigil watch: no webhook configured (use --webhook or alert_webhook_url in config)")
        from .alerts import post_webhook_alert
        asyncio.run(post_webhook_alert(
            config.alert_webhook_url,
            "test",
            "test",
            "Test alert from vigil watch. If you can read this, alerts are working.",
            format=config.alert_webhook_format,
        ))
        print(f"Sent test alert ({config.alert_webhook_format}) to {config.alert_webhook_url}")
        return

    if not config.api_key:
        names = ", ".join(provider.env_var_names())
        sys.exit(f"vigil watch: no {provider.display_name} API key (set {names}, --api-key, or api_key in config)")
    if not config.alert_webhook_url:
        print("warning: no webhook configured; alerts will only be printed here", file=sys.stderr, flush=True)

    from .watch import Watcher

    print(
        f"vigil watch: {provider.display_name}, stall after {config.stall_threshold_minutes}m, "
        f"GPU idle below {config.watch_gpu_idle_percent:g}%, "
        f"alerts -> {config.alert_webhook_format if config.alert_webhook_url else 'stdout'}",
        flush=True,
    )
    try:
        asyncio.run(Watcher(config, provider).run())
    except KeyboardInterrupt:
        pass


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "watch":
        _watch_main(argv[1:])
        return

    parser = argparse.ArgumentParser(
        description="Real-time TUI for monitoring cloud GPU training instances",
        epilog="Run `vigil watch --help` for the headless hang watchdog.",
    )
    _add_common_args(parser)
    parser.add_argument(
        "--reset-hints",
        action="store_true",
        help="Reset onboarding hints so they appear again",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with simulated GPU instances (no API key or SSH needed)",
    )
    args = parser.parse_args(argv)

    # Handle --reset-hints
    if args.reset_hints:
        from .state import load_state
        state = load_state()
        state.reset()
        print("Onboarding hints have been reset.")
        return

    config, provider = _load(args)

    # Load UI state
    from .state import load_state
    state = load_state()

    # No longer exit on missing API key — the wizard will handle it
    from .app import Dashboard

    app = Dashboard(config, state, provider=provider, demo=args.demo)
    app.run()


if __name__ == "__main__":
    main()
