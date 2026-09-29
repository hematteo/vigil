from __future__ import annotations

import asyncio

import httpx
from textual import events
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Footer, Header, Input, Static
from textual.worker import Worker

from .alerts import post_webhook_alert
from .collector import stream_instance_logs
from .config import Config, InstanceConfig
from .discovery import InstanceInfo, RateLimitError, redact
from .providers import Provider
from .state import AppState
from .parser import MetricParser
from .storage import LogStorage
from .widgets.command_input import CommandInputScreen
from .widgets.confirm_destroy import ConfirmDestroyScreen
from .widgets.global_search import GlobalSearchScreen
from .widgets.help_overlay import HelpOverlay
from .widgets.instance_manager import InstanceManagerScreen
from .widgets.instance_panel import InstancePanel
from .widgets.log_viewer import LogViewerScreen
from .widgets.metrics_overview import MetricsOverviewScreen
from .widgets.nvidia_smi import NvidiaSmiScreen
from .widgets.setup_wizard import SetupWizardScreen
from .widgets.hint_bar import HintBar
from .widgets.stuck_bar import StuckBar


_FOCUS_HINT = "Focus a panel first (press 1-9)"


class Dashboard(App):
    """Real-time TUI dashboard for GPU cloud training instances."""

    TITLE = "GPU Dashboard"

    CSS = """
    #grid {
        layout: grid;
        grid-size: 2;
        grid-gutter: 1;
        padding: 0 1;
    }

    #empty-state {
        width: 100%;
        height: auto;
        content-align: center middle;
        text-align: center;
        color: $text-muted;
        padding: 4;
    }

    #search-input {
        dock: bottom;
        margin: 0 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("1", "focus_panel(1)", ""),
        ("2", "focus_panel(2)", ""),
        ("3", "focus_panel(3)", ""),
        ("4", "focus_panel(4)", ""),
        ("5", "focus_panel(5)", ""),
        ("6", "focus_panel(6)", ""),
        ("7", "focus_panel(7)", ""),
        ("8", "focus_panel(8)", ""),
        ("9", "focus_panel(9)", ""),
        ("escape", "unfocus", "Back"),
        ("tab", "focus_next", "Next"),
        ("slash", "search", "Search"),
        ("l", "log_viewer", "Logs"),
        ("c", "change_command", "Cmd"),
        ("r", "reconnect", "Reconnect"),
        ("p", "toggle_pause", "Pause"),
        ("f", "toggle_follow", "Follow"),
        ("i", "instance_manager", "Instances"),
        ("m", "metrics_overview", "Metrics"),
        ("n", "nvidia_smi", "nvidia-smi"),
        ("s", "toggle_stuck", "Stuck"),
        ("shift+d", "destroy_instance", "Destroy"),
        ("shift+g", "global_search", "Search All"),
        ("shift+s", "setup_wizard", "Setup"),
        ("question_mark", "help", "Help"),
    ]

    def __init__(
        self, config: Config, state: AppState | None = None, provider: Provider | None = None, demo: bool = False
    ) -> None:
        super().__init__()
        self.app_config = config
        self.app_state = state or AppState()
        self.provider = provider
        self._demo = demo
        if demo:
            config.api_key = config.api_key or "demo"
            config.poll_interval = 600
            config.stall_threshold_minutes = 120
        self.title = f"{self.provider.display_name} Dashboard" if self.provider else "GPU Dashboard"
        self.sub_title = ""  # will be set by discover loop
        self.storage = LogStorage(config.log_dir)
        self.parser = MetricParser(config.metric_patterns)
        self._panels: dict[int | str, InstancePanel] = {}
        self._stream_workers: dict[int | str, Worker] = {}
        self._panel_order: list[int | str] = []
        self._focused_id: int | str | None = None
        self._searching = False
        self._http_client: httpx.AsyncClient | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._hint_bar: HintBar | None = None
        self._stuck_instances: list[InstanceInfo] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield ScrollableContainer(
            Static("Searching for running instances...", id="empty-state"),
            id="grid",
        )
        yield StuckBar()
        yield Footer()

    def on_mount(self) -> None:
        self._http_client = httpx.AsyncClient()

        if not self.app_config.api_key:
            # First launch — push wizard before starting workers
            self.push_screen(
                SetupWizardScreen(self.app_config, self._http_client, self.provider),
                callback=self._on_wizard_complete,
            )
            return

        self._start_workers()

    def _start_workers(self) -> None:
        """Start all background workers. Called after wizard or on normal launch."""
        if self._demo:
            self.run_worker(self._demo_discover_loop(), exclusive=True, group="discovery")
        else:
            self.run_worker(self._discover_loop(), exclusive=True, group="discovery")
        self.run_worker(self._flush_loop(), exclusive=True, group="flusher")
        if self.app_config.log_retention_days > 0 or self.app_config.log_max_size_mb > 0:
            self.run_worker(self._cleanup_loop(), exclusive=True, group="cleanup")
        self._mount_hint_bar()

    def _mount_hint_bar(self) -> None:
        """Mount the hint bar if not already mounted."""
        if self._hint_bar is None:
            self._hint_bar = HintBar(self.app_state)
            self.mount(self._hint_bar, before=self.query_one("Footer"))

    def _on_wizard_complete(self, completed: bool) -> None:
        """Called when the setup wizard is dismissed."""
        if completed:
            self.parser = MetricParser(self.app_config.metric_patterns)
        if self.app_config.api_key:
            self._start_workers()

    def _rebuild_runtime(self) -> None:
        """Recreate parser and restart workers after config changes."""
        # Recreate parser with updated patterns
        self.parser = MetricParser(self.app_config.metric_patterns)
        # Cancel all existing collectors and restart them with new config
        for worker in self._stream_workers.values():
            worker.cancel()
        self._stream_workers.clear()
        # Restart streams for all existing panels
        for iid, panel in self._panels.items():
            panel.clear_log()
            worker = self.run_worker(
                self._stream_instance(panel.instance_info, panel),
                group="collectors",
                exit_on_error=False,
            )
            self._stream_workers[iid] = worker

    # ------------------------------------------------------------------
    # Focus tracking — unifies Tab, click, and 1-9 key focus
    # ------------------------------------------------------------------

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        widget = event.widget
        while widget is not None:
            if isinstance(widget, InstancePanel):
                self._focused_id = widget.instance_info.id
                return
            widget = widget.parent
        self._focused_id = None

    # ------------------------------------------------------------------
    # Instance discovery
    # ------------------------------------------------------------------

    async def _discover_loop(self) -> None:
        first_run = True
        while True:
            try:
                result = await self.provider.fetch_instances(
                    self.app_config.api_key, self._http_client
                )
                credit = await self.provider.fetch_credit(
                    self.app_config.api_key, self._http_client
                )
                self._reconcile_panels(result.running)
                self._stuck_instances = result.stuck
                self.query_one(StuckBar).update_instances(result.stuck)
                self._update_cost_display(result.running, result.stuck, credit)
            except RateLimitError as exc:
                self.notify(
                    f"Rate limited, retry in {exc.retry_after}s",
                    severity="warning",
                    timeout=5,
                )
                await asyncio.sleep(exc.retry_after)
                continue
            except Exception as exc:
                msg = redact(str(exc), self.app_config.api_key)
                if first_run:
                    self._show_empty(f"Discovery error: {msg}")
                self.notify(f"Discovery: {msg}", severity="error", timeout=5)

            first_run = False
            await asyncio.sleep(self.app_config.poll_interval)

    async def _demo_discover_loop(self) -> None:
        from .demo import DEMO_CREDIT, DEMO_INSTANCES, DEMO_STREAMS, DEMO_TOTAL_DPH

        self._reconcile_panels(DEMO_INSTANCES)
        self.sub_title = (
            f"{len(DEMO_INSTANCES)} running | "
            f"${DEMO_TOTAL_DPH:.3f}/hr | Credit: ${DEMO_CREDIT:.2f}"
        )

        # Replace SSH stream workers with demo streams
        for inst, stream_fn in zip(DEMO_INSTANCES, DEMO_STREAMS):
            old_worker = self._stream_workers.pop(inst.id, None)
            if old_worker:
                old_worker.cancel()
            panel = self._panels[inst.id]
            worker = self.run_worker(
                stream_fn(panel, self.parser),
                group="collectors",
                exit_on_error=False,
            )
            self._stream_workers[inst.id] = worker

        # Sleep forever — no polling needed
        await asyncio.sleep(1e9)

    def _reconcile_panels(self, instances: list[InstanceInfo]) -> None:
        grid = self.query_one("#grid")
        live_ids = {inst.id for inst in instances}

        # Remove terminated instances
        for iid in set(self._panels.keys()) - live_ids:
            panel = self._panels.pop(iid)
            panel.mark_removed()
            worker = self._stream_workers.pop(iid, None)
            if worker:
                worker.cancel()
            panel.remove()
            self._panel_order = [x for x in self._panel_order if x != iid]
            self.storage.close(iid)
            self.notify(
                f"Instance #{iid} terminated",
                severity="warning",
                timeout=8,
            )

        # Add new instances
        def make_alert_callback(inst_info: InstanceInfo):
            def on_alert(alert_type: str, message: str, metrics: dict[str, str] | None) -> None:
                nc = self.app_config.notifications
                if not nc.webhook or not self.app_config.alert_webhook_url:
                    return
                if alert_type == "nan" and not nc.nan:
                    return
                if alert_type == "plateau" and not nc.plateau:
                    return
                task = asyncio.create_task(
                    post_webhook_alert(
                        self.app_config.alert_webhook_url,
                        inst_info.id,
                        alert_type,
                        message,
                        metrics=metrics,
                        instance=inst_info,
                        format=self.app_config.alert_webhook_format,
                    )
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            return on_alert

        for inst in instances:
            if inst.id not in self._panels:
                self._panel_order.append(inst.id)
                nc = self.app_config.notifications
                panel = InstancePanel(
                    inst,
                    panel_index=len(self._panel_order),
                    sparkline_history=self.app_config.sparkline_history,
                    plateau_window=self.app_config.plateau_window,
                    plateau_threshold=self.app_config.plateau_threshold,
                    plateau_metrics=self.app_config.plateau_metrics,
                    desktop_notifications=nc.desktop,
                    notify_nan=nc.nan,
                    notify_plateau=nc.plateau,
                    highlight_logs=self.app_config.highlight_logs,
                    decrease_good=self.app_config.decrease_good,
                    increase_good=self.app_config.increase_good,
                    counters=self.app_config.counters,
                    log_buffer_lines=self.app_config.log_buffer_lines,
                    log_display_lines=self.app_config.log_display_lines,
                    on_alert=make_alert_callback(inst),
                )
                self._panels[inst.id] = panel
                grid.mount(panel)

                worker = self.run_worker(
                    self._stream_instance(inst, panel),
                    group="collectors",
                    exit_on_error=False,
                )
                self._stream_workers[inst.id] = worker

        # Remove empty state message if we have panels
        empty = grid.query("#empty-state")
        if self._panels and empty:
            for w in empty:
                w.remove()
        elif not self._panels:
            self._show_empty("No running instances found. Polling...")

        self._update_grid_columns()
        self._update_panel_indices()

        # Notify hint bar when panels first appear
        if self._panels and self._hint_bar:
            self._hint_bar.notify_panels_appeared()

    def _update_panel_indices(self) -> None:
        """Refresh [N] labels so they match current _panel_order positions."""
        for idx, iid in enumerate(self._panel_order):
            panel = self._panels.get(iid)
            if not panel:
                continue
            inst = panel.instance_info
            title = f"[{idx + 1}] #{inst.id}  {inst.gpu_name} x{inst.num_gpus}"
            if inst.label:
                title += f"  [{inst.label}]"
            if inst.dph_total > 0:
                title += f"  ${inst.dph_total:.3f}/hr"
            panel.border_title = title

    def _update_cost_display(
        self,
        running: list[InstanceInfo],
        stuck: list[InstanceInfo],
        credit: float | None = None,
    ) -> None:
        running_dph = sum(inst.dph_total for inst in running)
        stuck_dph = sum(inst.dph_total for inst in stuck)
        total_dph = running_dph + stuck_dph
        parts: list[str] = []
        if running:
            parts.append(f"{len(running)} running")
        if stuck:
            parts.append(f"{len(stuck)} stuck")
        if total_dph > 0:
            parts.append(f"${total_dph:.3f}/hr")
        if credit is not None:
            parts.append(f"Credit: ${credit:.2f}")
        self.sub_title = " | ".join(parts) if parts else "No instances"

    def _show_empty(self, message: str) -> None:
        grid = self.query_one("#grid")
        existing = grid.query("#empty-state")
        if existing:
            existing.first().update(message)
        else:
            grid.mount(Static(message, id="empty-state"))

    def _update_grid_columns(self) -> None:
        grid = self.query_one("#grid")
        n = len(self._panels)
        max_cols = self.app_config.max_grid_columns
        if n <= 1:
            cols = 1
        elif n <= 4:
            cols = min(2, max_cols)
        else:
            cols = min(max_cols, n)
        grid.styles.grid_size_columns = cols

    # ------------------------------------------------------------------
    # Log streaming
    # ------------------------------------------------------------------

    async def _stream_instance(
        self, instance: InstanceInfo, panel: InstancePanel
    ) -> None:
        def on_stall(minutes: int) -> None:
            nc = self.app_config.notifications
            if self.app_config.alert_webhook_url and nc.stall and nc.webhook:
                task = asyncio.create_task(
                    post_webhook_alert(
                        self.app_config.alert_webhook_url,
                        instance.id,
                        "stall",
                        f"No output for {minutes}m",
                        instance=instance,
                        format=self.app_config.alert_webhook_format,
                    )
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

        await stream_instance_logs(
            instance=instance,
            config=self.app_config,
            storage=self.storage,
            parser=self.parser,
            on_line=panel.add_log_line,
            on_status=panel.set_status,
            on_stall=on_stall,
        )

    def _restart_stream(self, instance_id: int | str) -> None:
        """Restart the log stream for an instance (e.g. after command change)."""
        panel = self._panels.get(instance_id)
        if not panel:
            return

        old_worker = self._stream_workers.pop(instance_id, None)
        if old_worker:
            old_worker.cancel()

        panel.clear_log()
        panel.set_status("reconnecting with new command...")

        worker = self.run_worker(
            self._stream_instance(panel.instance_info, panel),
            group="collectors",
            exit_on_error=False,
        )
        self._stream_workers[instance_id] = worker

    # ------------------------------------------------------------------
    # Periodic flush
    # ------------------------------------------------------------------

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(10)
            self.storage.flush_all()

    async def _cleanup_loop(self) -> None:
        """Periodically clean up old log files based on retention config."""
        while True:
            result = self.storage.cleanup(
                retention_days=self.app_config.log_retention_days,
                max_size_mb=self.app_config.log_max_size_mb,
            )
            if result["files_removed"] > 0:
                mb = result["bytes_freed"] / (1024 * 1024)
                self.notify(
                    f"Log cleanup: {result['files_removed']} files, {mb:.1f}MB freed",
                    timeout=5,
                )
            await asyncio.sleep(3600)  # Run every hour

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def action_search(self) -> None:
        if self._hint_bar:
            self._hint_bar.notify_action("search")
        if self._searching:
            self._close_search()
            return

        self._searching = True
        search = Input(placeholder="Filter logs...", id="search-input")
        self.mount(search, before=self.query_one("Footer"))
        search.focus()

    def _close_search(self) -> None:
        self._searching = False
        for w in self.query("#search-input"):
            w.remove()
        for panel in self._panels.values():
            panel.set_search("")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search-input":
            for panel in self._panels.values():
                panel.set_search(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search-input":
            if not event.value:
                self._close_search()

    # ------------------------------------------------------------------
    # Key bindings
    # ------------------------------------------------------------------

    def action_focus_panel(self, index: int) -> None:
        if self._hint_bar:
            self._hint_bar.notify_action("focus_panel")
        if index < 1 or index > len(self._panel_order):
            return

        iid = self._panel_order[index - 1]
        panel = self._panels.get(iid)
        if not panel:
            return

        if self._focused_id == iid:
            # Directly unfocus — don't go through action_unfocus which
            # prioritizes closing search over unfocusing the panel
            self._focused_id = None
            for p in self._panels.values():
                p.display = True
            self._update_grid_columns()
            return

        self._focused_id = iid
        for pid, p in self._panels.items():
            p.display = pid == iid
        grid = self.query_one("#grid")
        grid.styles.grid_size_columns = 1
        panel.focus()

    def action_unfocus(self) -> None:
        # Close search if active
        if self._searching:
            self._close_search()
            return
        # Unfocus panel if focused
        if self._focused_id is not None:
            self._focused_id = None
            for p in self._panels.values():
                p.display = True
            self._update_grid_columns()

    def action_log_viewer(self) -> None:
        self.push_screen(LogViewerScreen(self.storage))

    def action_change_command(self) -> None:
        target_id = self._focused_id
        if target_id is None:
            self.notify(_FOCUS_HINT, severity="warning")
            return

        current_cmd = self.app_config.log_command_for(target_id)

        def on_result(new_command: str | None) -> None:
            if new_command is None or new_command == current_cmd:
                return
            # Update config in memory
            if target_id not in self.app_config.instances:
                self.app_config.instances[target_id] = InstanceConfig()
            self.app_config.instances[target_id].log_command = new_command
            self.app_config.save_config()
            self.notify(f"Command updated for #{target_id}")
            self._restart_stream(target_id)

        self.push_screen(
            CommandInputScreen(target_id, current_cmd),
            callback=on_result,
        )

    def action_reconnect(self) -> None:
        if self._focused_id is None:
            self.notify(_FOCUS_HINT, severity="warning")
            return
        self._restart_stream(self._focused_id)
        self.notify(f"Reconnecting #{self._focused_id}...")

    def action_toggle_pause(self) -> None:
        if self._focused_id is None:
            self.notify(_FOCUS_HINT, severity="warning")
            return
        panel = self._panels.get(self._focused_id)
        if panel:
            paused = panel.toggle_pause()
            self.notify(f"#{self._focused_id}: {'paused' if paused else 'resumed'}")

    def action_toggle_follow(self) -> None:
        if self._focused_id is None:
            self.notify(_FOCUS_HINT, severity="warning")
            return
        panel = self._panels.get(self._focused_id)
        if panel:
            following = panel.toggle_follow()
            self.notify(f"#{self._focused_id}: follow {'on' if following else 'off'}")

    def action_instance_manager(self) -> None:
        self.push_screen(InstanceManagerScreen(
            self._panels,
            self._stuck_instances,
            self.app_config.api_key,
            self._http_client,
            self.provider,
        ))

    def action_metrics_overview(self) -> None:
        if self._hint_bar:
            self._hint_bar.notify_action("metrics_overview")
        self.push_screen(MetricsOverviewScreen(
            self._panels,
            decrease_good=self.app_config.decrease_good,
            increase_good=self.app_config.increase_good,
        ))

    def action_nvidia_smi(self) -> None:
        if self._focused_id is None:
            self.notify(_FOCUS_HINT, severity="warning")
            return
        panel = self._panels.get(self._focused_id)
        if panel:
            self.push_screen(
                NvidiaSmiScreen(
                    panel.instance_info,
                    self.app_config,
                )
            )

    def action_toggle_stuck(self) -> None:
        self.query_one(StuckBar).toggle()

    def action_destroy_instance(self) -> None:
        # Check if stuck bar has a selection
        stuck_bar = self.query_one(StuckBar)
        stuck_id = stuck_bar.selected_instance_id
        if self._focused_id is None and stuck_id is not None:
            inst = next((i for i in self._stuck_instances if str(i.id) == str(stuck_id)), None)
            if inst is None:
                return

            def on_confirm_stuck(confirmed: bool) -> None:
                if confirmed and self._http_client:
                    self.run_worker(
                        self._do_destroy(inst.id), exit_on_error=False
                    )

            self.push_screen(
                ConfirmDestroyScreen(inst.id, inst.gpu_name, inst.dph_total),
                callback=on_confirm_stuck,
            )
            return

        # Existing behavior for running panels
        if self._focused_id is None:
            self.notify(_FOCUS_HINT, severity="warning")
            return
        panel = self._panels.get(self._focused_id)
        if not panel:
            return
        inst = panel.instance_info

        def on_confirm(confirmed: bool) -> None:
            if confirmed and self._http_client:
                panel.set_status("stopping...")
                panel.add_class("stopping")
                self.run_worker(
                    self._do_destroy(inst.id), exit_on_error=False
                )

        self.push_screen(
            ConfirmDestroyScreen(inst.id, inst.gpu_name, inst.dph_total),
            callback=on_confirm,
        )

    async def _do_destroy(self, instance_id: int | str) -> None:
        try:
            await self.provider.destroy_instance(
                self.app_config.api_key,
                instance_id,
                self._http_client,
            )
            self.notify(f"Instance #{instance_id} stop requested")
        except Exception as exc:
            msg = redact(str(exc), self.app_config.api_key)
            self.notify(f"Failed to stop #{instance_id}: {msg}", severity="error")

    def action_global_search(self) -> None:
        self.push_screen(GlobalSearchScreen(self._panels))

    def action_help(self) -> None:
        if self._hint_bar:
            self._hint_bar.notify_action("help")
        self.push_screen(HelpOverlay())

    def action_setup_wizard(self) -> None:
        self.push_screen(
            SetupWizardScreen(self.app_config, self._http_client, self.provider),
            callback=self._on_wizard_reentry,
        )

    def _on_wizard_reentry(self, completed: bool) -> None:
        """Called when setup wizard is dismissed after re-entry via S."""
        if completed:
            self._rebuild_runtime()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def on_unmount(self) -> None:
        # Cancel all workers before closing shared resources
        for worker in self._stream_workers.values():
            worker.cancel()
        self._stream_workers.clear()
        # Cancel discovery/flush/cleanup workers so they don't use closed clients
        self.workers.cancel_group(self, "discovery")
        self.workers.cancel_group(self, "flusher")
        self.workers.cancel_group(self, "cleanup")
        # Brief yield to let cancelled coroutines finish
        await asyncio.sleep(0)
        self.storage.close()
        if self._http_client:
            await self._http_client.aclose()


# Backward-compat alias
VastDashboard = Dashboard
