from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from ..discovery import InstanceInfo, redact
from .confirm_destroy import ConfirmDestroyScreen

if TYPE_CHECKING:
    from ..providers import Provider
    from .instance_panel import InstancePanel


class InstanceManagerScreen(Screen):
    """Detailed view of all instances with management actions."""

    DEFAULT_CSS = """
    InstanceManagerScreen DataTable {
        height: 1fr;
    }

    InstanceManagerScreen .section-label {
        height: 1;
        padding: 0 1;
        color: $text-muted;
        text-style: bold;
    }

    InstanceManagerScreen #stuck-instance-table {
        height: auto;
        max-height: 50%;
    }
    """

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("r", "refresh", "Refresh"),
        ("shift+d", "destroy_stuck", "Destroy"),
    ]

    def __init__(
        self,
        panels: dict[int, InstancePanel],
        stuck_instances: list[InstanceInfo],
        api_key: str,
        http_client: httpx.AsyncClient | None,
        provider: Provider | None = None,
    ) -> None:
        super().__init__()
        self.panels = panels
        self.stuck_instances = stuck_instances
        self._api_key = api_key
        self._http_client = http_client
        self._provider = provider

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Running Instances", classes="section-label")
        yield DataTable(id="instance-table", cursor_type="row")
        yield Static("Stuck Instances", classes="section-label")
        yield DataTable(id="stuck-instance-table", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        self._build_tables()

    def action_refresh(self) -> None:
        self._build_tables()

    def action_destroy_stuck(self) -> None:
        if not self.stuck_instances:
            return
        stuck_table = self.query_one("#stuck-instance-table", DataTable)
        if stuck_table.cursor_row is None or stuck_table.row_count == 0:
            return
        try:
            row = stuck_table.get_row_at(stuck_table.cursor_row)
            instance_id = row[0]
        except (IndexError, ValueError):
            return

        inst = next((i for i in self.stuck_instances if str(i.id) == str(instance_id)), None)
        if inst is None:
            return

        def on_confirm(confirmed: bool) -> None:
            if confirmed and self._http_client:
                self.app.run_worker(
                    self._do_destroy(instance_id), exit_on_error=False
                )

        self.app.push_screen(
            ConfirmDestroyScreen(inst.id, inst.gpu_name, inst.dph_total),
            callback=on_confirm,
        )

    async def _do_destroy(self, instance_id: int | str) -> None:
        try:
            await self._provider.destroy_instance(self._api_key, instance_id, self._http_client)
            if self.is_attached:
                self.app.notify(f"Instance #{instance_id} stop requested")
        except Exception as exc:
            if self.is_attached:
                self.app.notify(f"Failed to stop #{instance_id}: {redact(str(exc), self._api_key)}", severity="error")

    def _build_tables(self) -> None:
        self._build_running_table()
        self._build_stuck_table()

    def _build_running_table(self) -> None:
        table = self.query_one("#instance-table", DataTable)
        table.clear(columns=True)

        if not self.panels:
            table.add_column("Status")
            table.add_row("No running instances")
            return

        table.add_column("ID", key="id")
        table.add_column("GPU", key="gpu")
        table.add_column("Count", key="count")
        table.add_column("Label", key="label")
        table.add_column("$/hr", key="cost")
        table.add_column("Status", key="status")
        table.add_column("SSH", key="ssh")

        for iid, panel in sorted(self.panels.items()):
            inst = panel.instance_info
            try:
                status_widget = panel.query_one(".panel-status")
                status_str = str(getattr(status_widget, "renderable", ""))[:30]
            except Exception:
                status_str = "unknown"

            ssh_cmd = f"ssh root@{inst.ssh_host} -p {inst.ssh_port}"

            table.add_row(
                str(inst.id),
                inst.gpu_name,
                str(inst.num_gpus),
                inst.label or "\u2014",
                f"${inst.dph_total:.3f}",
                status_str,
                ssh_cmd,
            )

    def _build_stuck_table(self) -> None:
        table = self.query_one("#stuck-instance-table", DataTable)
        table.clear(columns=True)

        if not self.stuck_instances:
            table.add_column("Status")
            table.add_row("No stuck instances")
            return

        table.add_column("ID", key="id")
        table.add_column("GPU", key="gpu")
        table.add_column("Count", key="count")
        table.add_column("Label", key="label")
        table.add_column("$/hr", key="cost")
        table.add_column("Status", key="status")

        for inst in self.stuck_instances:
            table.add_row(
                str(inst.id),
                inst.gpu_name,
                str(inst.num_gpus),
                inst.label or "\u2014",
                f"${inst.dph_total:.3f}",
                inst.status,
            )
