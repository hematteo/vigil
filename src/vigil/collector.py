from __future__ import annotations

import asyncio
import re
import time
from typing import Callable

import asyncssh

from .config import Config
from .discovery import InstanceInfo
from .parser import MetricParser
from .ssh import ssh_connect
from .storage import LogStorage


def _write_to_storage(
    storage: LogStorage,
    parser: MetricParser,
    instance_id: int,
    line: str,
) -> dict[str, str]:
    """Write a line to storage, parse metrics, and return them.

    Disk errors are silently ignored so they never tear down the SSH session.
    """
    try:
        storage.write_line(instance_id, line)
    except OSError:
        pass

    metrics = parser.parse_line(line)
    if metrics:
        try:
            storage.write_metrics(instance_id, metrics, time.time())
        except OSError:
            pass

    return metrics


_NEWLINE = re.compile(r"\r\n|\n|\r")

# Emit a line anyway once the buffer gets this big without a line ending.
_MAX_LINE_CHARS = 65536

# Progress-bar redraws (bare \r, as tqdm does) all count as activity, but only
# one per this many seconds is logged and parsed, so bars don't flood the logs.
PROGRESS_EMIT_SECONDS = 10.0


class LineSplitter:
    """Split streamed text on \n, \r\n, and bare \r (progress-bar redraws).

    ``feed`` returns ``(text, is_progress)`` pairs, where ``is_progress`` marks
    a segment ended by a bare \r. A trailing \r is held back until the next
    chunk shows whether it starts a \r\n.
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, data: str) -> list[tuple[str, bool]]:
        self._buf += data
        out: list[tuple[str, bool]] = []
        pos = 0
        for m in _NEWLINE.finditer(self._buf):
            if m.group() == "\r" and m.end() == len(self._buf):
                break
            out.append((self._buf[pos:m.start()], m.group() == "\r"))
            pos = m.end()
        self._buf = self._buf[pos:]
        if len(self._buf) > _MAX_LINE_CHARS:
            out.append((self._buf, False))
            self._buf = ""
        return out

    def flush(self) -> list[tuple[str, bool]]:
        rest, self._buf = self._buf, ""
        if rest.endswith("\r"):
            return [(rest[:-1], True)]
        return [(rest, False)] if rest else []


async def _read_stderr(process: asyncssh.SSHClientProcess) -> str:
    """Best-effort read of stderr after process exit."""
    if not process.stderr:
        return ""
    try:
        data = await asyncio.wait_for(process.stderr.read(4096), timeout=2.0)
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")
        return data
    except Exception:
        return ""


async def _stream_one_session(
    instance: InstanceInfo,
    config: Config,
    storage: LogStorage,
    parser: MetricParser,
    on_line: Callable[[str, dict[str, str]], None],
    on_status: Callable[[str], None],
    update_line_time: Callable[[], None],
) -> None:
    """Run a single SSH connect-stream-disconnect cycle."""
    on_status("connecting...")

    conn = await asyncio.wait_for(
        ssh_connect(instance, config),
        timeout=config.ssh_login_timeout + 10,
    )

    async with conn:
        on_status("connected")
        line_count = 0

        splitter = LineSplitter()
        last_progress_emit = 0.0

        def handle(segments: list[tuple[str, bool]]) -> None:
            nonlocal line_count, last_progress_emit
            for line, is_progress in segments:
                update_line_time()
                if is_progress:
                    now = time.monotonic()
                    if not line.strip() or now - last_progress_emit < PROGRESS_EMIT_SECONDS:
                        continue
                    last_progress_emit = now

                metrics = _write_to_storage(storage, parser, instance.id, line)
                on_line(line, metrics)

                line_count += 1
                if line_count % 100 == 0:
                    storage.flush(instance.id)

        command = config.log_command_for(instance.id)
        async with conn.create_process(command) as process:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        process.stdout.read(65536),
                        timeout=config.stall_threshold_for(instance.id) * 60 + 30,
                    )
                except asyncio.TimeoutError:
                    on_status("stalled — no output, reconnecting")
                    raise
                if not chunk:
                    break
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8", errors="replace")
                handle(splitter.feed(chunk))
            handle(splitter.flush())

            storage.flush(instance.id)
            stderr_text = await _read_stderr(process)

        exit_msg = f"process exited (rc={process.returncode})"
        if stderr_text:
            exit_msg += f": {stderr_text[:200]}"
        on_status(exit_msg)
        # Treat non-zero exit as an error so the outer loop applies backoff
        # instead of tight-looping on a broken log command.
        if process.returncode is not None and process.returncode != 0:
            raise OSError(f"Remote process exited with rc={process.returncode}")
        await asyncio.sleep(2)


async def stream_instance_logs(
    instance: InstanceInfo,
    config: Config,
    storage: LogStorage,
    parser: MetricParser,
    on_line: Callable[[str, dict[str, str]], None],
    on_status: Callable[[str], None],
    on_stall: Callable[[int], None] = lambda _: None,
) -> None:
    """Stream logs from a Vast.ai instance over SSH.

    Reconnects with exponential backoff on failure.
    Runs until cancelled.
    """
    backoff = 1
    max_backoff = config.reconnect_backoff_max
    last_line_time = time.monotonic()
    stall_alerted = False
    stall_threshold_minutes = config.stall_threshold_for(instance.id)

    def update_line_time() -> None:
        nonlocal last_line_time
        last_line_time = time.monotonic()

    async def _check_stall() -> None:
        nonlocal stall_alerted
        while True:
            await asyncio.sleep(30)
            gap_minutes = (time.monotonic() - last_line_time) / 60
            if gap_minutes >= stall_threshold_minutes and not stall_alerted:
                stall_alerted = True
                on_status(f"STALLED — no output for {int(gap_minutes)}m")
                on_stall(int(gap_minutes))
            elif gap_minutes < stall_threshold_minutes and stall_alerted:
                stall_alerted = False

    stall_task = asyncio.create_task(_check_stall())
    try:
        while True:
            last_line_time = time.monotonic()
            stall_alerted = False
            try:
                await _stream_one_session(
                    instance, config, storage, parser,
                    on_line, on_status, update_line_time,
                )
                backoff = 1
            except asyncio.CancelledError:
                raise
            except (asyncssh.Error, OSError, asyncio.TimeoutError) as exc:
                short = type(exc).__name__
                on_status(f"disconnected ({short}), retry in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
            except Exception as exc:
                short = type(exc).__name__
                on_status(f"error ({short}: {exc}), retry in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
    finally:
        stall_task.cancel()
        try:
            await stall_task
        except (asyncio.CancelledError, Exception):
            pass  # Swallow only the stall_task's own exception
