"""Headless hang watchdog behind `vigil watch`.

Runs without the TUI (e.g. on a small always-on box) and alerts when a training
run stops producing output. GPU utilization and memory, sampled over SSH, tell
apart a hung process, one that exited, and one that is just quietly busy.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Awaitable, Callable

import asyncssh
import httpx

from .alerts import post_webhook_alert
from .collector import stream_instance_logs
from .config import Config
from .discovery import InstanceInfo, RateLimitError, redact
from .parser import MetricParser
from .providers import Provider
from .ssh import ssh_connect
from .storage import LogStorage

GPU_QUERY = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"

# Lines arriving this soon after (re)connecting are the log command's backlog
# (`tail -n 100 -f` replays old lines), not new output.
BACKLOG_GRACE_SECONDS = 3.0

# A GPU using less than this fraction of its memory counts as released.
MEMORY_RELEASED_FRACTION = 0.05

EVAL_INTERVAL_SECONDS = 15.0

OK = "ok"
HANG = "hang"
EXITED = "exited"
BUSY_SILENT = "busy_silent"
UNREACHABLE = "unreachable"
RECOVERED = "recovered"
GONE = "gone"


@dataclass
class GpuSample:
    t: float
    util: list[float]
    mem_used: list[float]
    mem_total: list[float]

    @property
    def max_util(self) -> float:
        return max(self.util, default=0.0)

    @property
    def memory_released(self) -> bool:
        return all(
            total <= 0 or used / total < MEMORY_RELEASED_FRACTION
            for used, total in zip(self.mem_used, self.mem_total)
        )


def parse_gpu_query(text: str, t: float) -> GpuSample | None:
    """Parse `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total` CSV output."""
    util: list[float] = []
    used: list[float] = []
    total: list[float] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            u, m, tot = (float(p) for p in parts)
        except ValueError:
            continue  # e.g. "[N/A]" on MIG devices
        util.append(u)
        used.append(m)
        total.append(tot)
    if not util:
        return None
    return GpuSample(t=t, util=util, mem_used=used, mem_total=total)


async def probe_gpu(instance: InstanceInfo, config: Config, now: float) -> GpuSample | None:
    """Sample GPU utilization and memory over a one-shot SSH connection. None on any failure."""
    try:
        conn = await asyncio.wait_for(
            ssh_connect(instance, config, keepalive=False),
            timeout=config.ssh_login_timeout + 10,
        )
        async with conn:
            result = await asyncio.wait_for(conn.run(GPU_QUERY, check=False), timeout=20.0)
    except (asyncssh.Error, OSError, asyncio.TimeoutError):
        return None
    stdout = result.stdout or ""
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    return parse_gpu_query(stdout, now)


class HangDetector:
    """Classify one instance from its output timing and GPU samples.

    All methods take explicit timestamps so the logic is deterministic to test.
    """

    def __init__(
        self,
        *,
        started_at: float,
        stall_minutes: float,
        busy_silence_minutes: float,
        idle_percent: float,
        reminder_minutes: float,
    ) -> None:
        self.stall_seconds = stall_minutes * 60
        self.busy_silence_seconds = busy_silence_minutes * 60
        self.idle_percent = idle_percent
        self.reminder_seconds = reminder_minutes * 60
        self.last_output = started_at
        self.last_connect: float | None = None
        self.last_gpu_ok = started_at
        self.samples: deque[GpuSample] = deque(maxlen=240)
        self.state = OK
        self.last_alert = started_at
        self.silence_started = started_at

    def observe_connected(self, now: float) -> None:
        self.last_connect = now

    def observe_line(self, now: float) -> None:
        if self.last_connect is not None and now - self.last_connect < BACKLOG_GRACE_SECONDS:
            return
        self.last_output = now

    def observe_gpu(self, sample: GpuSample) -> None:
        self.samples.append(sample)
        self.last_gpu_ok = sample.t

    def silence(self, now: float) -> float:
        return now - self.last_output

    def latest_sample(self) -> GpuSample | None:
        return self.samples[-1] if self.samples else None

    def classify(self, now: float) -> str:
        silent = self.silence(now)
        if silent < self.stall_seconds:
            return OK
        if now - self.last_gpu_ok >= self.stall_seconds:
            return UNREACHABLE
        # Need the GPU idle across the whole stall window (and at least two
        # samples) so a brief dip between epochs doesn't trigger an alert.
        recent = [s for s in self.samples if s.t >= now - self.stall_seconds]
        if len(recent) >= 2 and all(s.max_util < self.idle_percent for s in recent):
            return EXITED if recent[-1].memory_released else HANG
        # GPU still busy: often a long eval or checkpoint, but NCCL deadlocks
        # also spin at 100% util, so alert once the silence gets long enough.
        if silent >= self.busy_silence_seconds:
            return BUSY_SILENT
        return OK

    def step(self, now: float) -> tuple[str, bool] | None:
        """Advance the state machine. Returns (alert_type, is_reminder) when an alert is due."""
        new = self.classify(now)
        if new != self.state:
            old, self.state = self.state, new
            self.last_alert = now
            if new == OK:
                return (RECOVERED, False)
            if old == OK:
                self.silence_started = self.last_output
            return (new, False)
        if new != OK and self.reminder_seconds > 0 and now - self.last_alert >= self.reminder_seconds:
            self.last_alert = now
            return (new, True)
        return None


def _fmt_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m"
    return f"{minutes // 60}h{minutes % 60:02d}m"


def _cost_summary(inst: InstanceInfo, silent_seconds: float, now: float, first_seen: float) -> str:
    if inst.dph_total <= 0:
        return ""
    if inst.start_time and inst.start_time <= now:
        hours, label = (now - inst.start_time) / 3600, "total"
    else:
        hours, label = (now - first_seen) / 3600, "since vigil started watching"
    return (
        f" Cost: ~${inst.dph_total * silent_seconds / 3600:.2f} during this silence,"
        f" ~${inst.dph_total * hours:.2f} {label} ({hours:.1f}h @ ${inst.dph_total:.3f}/hr)."
    )


def describe(
    alert_type: str,
    det: HangDetector,
    inst: InstanceInfo,
    now: float,
    first_seen: float,
    *,
    reminder: bool = False,
) -> str:
    """Build the human-readable alert text, including what the silence has cost."""
    silent = det.silence(now)
    quiet = _fmt_duration(silent)
    sample = det.latest_sample()
    util = f"{sample.max_util:.0f}%" if sample else "unknown"

    if alert_type == HANG:
        text = (
            f"No new output for {quiet} and GPU idle ({util} util) while still holding memory."
            " Training looks hung."
        )
    elif alert_type == EXITED:
        text = (
            f"No new output for {quiet}, GPU idle and memory released."
            " The training process looks finished or crashed, but the instance is still billing."
        )
    elif alert_type == BUSY_SILENT:
        text = (
            f"No new output for {quiet} but GPU reports {util} util."
            " Could be a very long eval/checkpoint, or a distributed (NCCL) deadlock, which spins at full util."
        )
    elif alert_type == UNREACHABLE:
        text = f"No new output for {quiet} and nvidia-smi over SSH has been failing. Instance may be down."
    elif alert_type == RECOVERED:
        return f"Output resumed after ~{_fmt_duration(now - det.silence_started)} of silence."
    elif alert_type == GONE:
        return "Instance is no longer listed as running (stopped, destroyed or preempted)."
    else:
        text = f"{alert_type}: no new output for {quiet}."

    if reminder:
        text = "Still unresolved. " + text
    return text + _cost_summary(inst, silent, now, first_seen)


Notifier = Callable[[InstanceInfo, str, str], Awaitable[None]]
StreamFn = Callable[..., Awaitable[None]]
ProbeFn = Callable[[InstanceInfo, Config, float], Awaitable["GpuSample | None"]]


@dataclass
class _Watched:
    info: InstanceInfo
    detector: HangDetector
    first_seen: float
    tasks: list[asyncio.Task[None]] = field(default_factory=list)


class Watcher:
    """Discover instances, stream their logs, sample their GPUs, and alert on hangs."""

    def __init__(
        self,
        config: Config,
        provider: Provider,
        *,
        clock: Callable[[], float] = time.time,
        log: Callable[[str], None] | None = None,
        notify: Notifier | None = None,
        stream: StreamFn = stream_instance_logs,
        probe: ProbeFn = probe_gpu,
    ) -> None:
        self.config = config
        self.provider = provider
        self.clock = clock
        self._log = log or (lambda msg: print(msg, flush=True))
        self._notify = notify or self._post
        self._stream = stream
        self._probe = probe
        self._client: httpx.AsyncClient | None = None
        self.storage = LogStorage(config.log_dir)
        self.parser = MetricParser(config.metric_patterns)
        self.watched: dict[int | str, _Watched] = {}
        self._skipped: set[int | str] = set()

    def log(self, msg: str) -> None:
        self._log(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}")

    async def run(self) -> None:
        async with httpx.AsyncClient() as client:
            self._client = client
            evaluator = asyncio.create_task(self._eval_loop())
            try:
                while True:
                    try:
                        result = await self.provider.fetch_instances(self.config.api_key, client)
                        await self.reconcile(result.running)
                    except RateLimitError as exc:
                        self.log(f"Rate limited by {self.provider.display_name}, retrying in {exc.retry_after:.0f}s")
                        await asyncio.sleep(exc.retry_after)
                        continue
                    except Exception as exc:
                        self.log(redact(f"Discovery error: {type(exc).__name__}: {exc}", self.config.api_key))
                    await asyncio.sleep(self.config.poll_interval)
            finally:
                evaluator.cancel()
                for w in self.watched.values():
                    for task in w.tasks:
                        task.cancel()
                await asyncio.gather(
                    evaluator, *(t for w in self.watched.values() for t in w.tasks),
                    return_exceptions=True,
                )
                self.storage.close()

    async def reconcile(self, running: list[InstanceInfo]) -> None:
        live: dict[int | str, InstanceInfo] = {}
        for inst in running:
            if self.config.should_watch(inst.id, inst.label):
                live[inst.id] = inst
                self._skipped.discard(inst.id)
            elif inst.id not in self._skipped:
                self._skipped.add(inst.id)
                label = f" [{inst.label}]" if inst.label else ""
                self.log(f"Skipping #{inst.id}{label} (excluded by watch settings)")

        for iid in list(self.watched):
            if iid not in live:
                w = self.watched.pop(iid)
                for task in w.tasks:
                    task.cancel()
                self.storage.close(iid)
                if iid not in self._skipped:  # still running, just excluded now (e.g. relabelled)
                    await self._alert(w, GONE, reminder=False)

        now = self.clock()
        for iid, inst in live.items():
            w = self.watched.get(iid)
            if w is not None:
                w.info = inst  # pick up price/label changes
                continue
            det = HangDetector(
                started_at=now,
                stall_minutes=self.config.stall_threshold_for(iid),
                busy_silence_minutes=self.config.watch_busy_silence_minutes,
                idle_percent=self.config.watch_gpu_idle_percent,
                reminder_minutes=self.config.watch_reminder_minutes,
            )
            w = _Watched(info=inst, detector=det, first_seen=now)
            w.tasks = [
                asyncio.create_task(self._stream_loop(w)),
                asyncio.create_task(self._probe_loop(w)),
            ]
            self.watched[iid] = w
            label = f" [{inst.label}]" if inst.label else ""
            self.log(f"Watching #{iid}{label} {inst.gpu_name} x{inst.num_gpus} ${inst.dph_total:.3f}/hr")

    async def _stream_loop(self, w: _Watched) -> None:
        def on_line(_line: str, _metrics: dict[str, str]) -> None:
            w.detector.observe_line(self.clock())

        def on_status(status: str) -> None:
            if status == "connected":
                w.detector.observe_connected(self.clock())

        await self._stream(
            instance=w.info,
            config=self.config,
            storage=self.storage,
            parser=self.parser,
            on_line=on_line,
            on_status=on_status,
        )

    async def _probe_loop(self, w: _Watched) -> None:
        while True:
            sample = await self._probe(w.info, self.config, self.clock())
            if sample is not None:
                w.detector.observe_gpu(sample)
            await asyncio.sleep(self.config.watch_gpu_poll_seconds)

    async def _eval_loop(self) -> None:
        while True:
            await asyncio.sleep(EVAL_INTERVAL_SECONDS)
            await self.evaluate()

    async def evaluate(self) -> None:
        now = self.clock()
        for iid, w in list(self.watched.items()):
            due = w.detector.step(now)
            if due is not None:
                alert_type, reminder = due
                await self._alert(w, alert_type, reminder=reminder, now=now)
            try:
                self.storage.flush(iid)
            except OSError:
                pass

    async def _alert(self, w: _Watched, alert_type: str, *, reminder: bool, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        text = describe(alert_type, w.detector, w.info, now, w.first_seen, reminder=reminder)
        self.log(f"#{w.info.id} {alert_type.upper()}: {text}")
        await self._notify(w.info, alert_type, text)

    async def _post(self, inst: InstanceInfo, alert_type: str, text: str) -> None:
        url = self.config.alert_webhook_url
        if not url or not self.config.notifications.webhook:
            return
        await post_webhook_alert(
            url,
            inst.id,
            alert_type,
            text,
            instance=inst,
            format=self.config.alert_webhook_format,
            client=self._client,
        )
