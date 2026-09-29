from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from vigil.alerts import post_webhook_alert
from vigil.config import Config
from vigil.discovery import InstanceInfo
from vigil.watch import (
    BUSY_SILENT,
    EXITED,
    GONE,
    HANG,
    OK,
    RECOVERED,
    UNREACHABLE,
    GpuSample,
    HangDetector,
    Watcher,
    describe,
    parse_gpu_query,
)

T0 = 1_000_000.0
MIN = 60.0


def _sample(t: float, util: float, used: float = 20000, total: float = 80000) -> GpuSample:
    return GpuSample(t=t, util=[util], mem_used=[used], mem_total=[total])


def _detector(**overrides) -> HangDetector:
    kwargs = dict(
        started_at=T0,
        stall_minutes=10,
        busy_silence_minutes=30,
        idle_percent=5.0,
        reminder_minutes=60,
    )
    kwargs.update(overrides)
    return HangDetector(**kwargs)


def _feed_gpu(det: HangDetector, start: float, end: float, util: float, **kw) -> None:
    t = start
    while t <= end:
        det.observe_gpu(_sample(t, util, **kw))
        t += MIN


def _instance(**overrides) -> InstanceInfo:
    base = dict(
        id=42,
        ssh_host="1.2.3.4",
        ssh_port=22,
        gpu_name="A100",
        num_gpus=4,
        status="running",
        label="llama-ft",
        dph_total=2.0,
    )
    base.update(overrides)
    return InstanceInfo(**base)


# ---------------------------------------------------------------------------
# parse_gpu_query
# ---------------------------------------------------------------------------


class TestParseGpuQuery:
    def test_multi_gpu(self):
        s = parse_gpu_query("98, 70000, 81920\n3, 500, 81920\n", T0)
        assert s is not None
        assert s.util == [98.0, 3.0]
        assert s.max_util == 98.0
        assert not s.memory_released

    def test_skips_na_rows(self):
        s = parse_gpu_query("[N/A], 100, 40000\n0, 100, 40000\n", T0)
        assert s is not None
        assert s.util == [0.0]
        assert s.memory_released

    def test_garbage_returns_none(self):
        assert parse_gpu_query("NVIDIA-SMI has failed", T0) is None
        assert parse_gpu_query("", T0) is None


# ---------------------------------------------------------------------------
# HangDetector
# ---------------------------------------------------------------------------


class TestHangDetector:
    def test_output_flowing_is_ok(self):
        det = _detector()
        _feed_gpu(det, T0, T0 + 20 * MIN, util=0)
        for m in range(0, 21):
            det.observe_line(T0 + m * MIN)
        assert det.step(T0 + 20 * MIN) is None
        assert det.state == OK

    def test_silent_and_idle_with_memory_held_is_hang(self):
        det = _detector()
        _feed_gpu(det, T0, T0 + 11 * MIN, util=0, used=60000)
        assert det.step(T0 + 11 * MIN) == (HANG, False)

    def test_silent_idle_memory_released_is_exited(self):
        det = _detector()
        _feed_gpu(det, T0, T0 + 11 * MIN, util=0, used=300)
        assert det.step(T0 + 11 * MIN) == (EXITED, False)

    def test_brief_idle_dip_does_not_alert(self):
        det = _detector()
        _feed_gpu(det, T0, T0 + 8 * MIN, util=95)
        _feed_gpu(det, T0 + 9 * MIN, T0 + 11 * MIN, util=0)
        assert det.step(T0 + 11 * MIN) is None

    def test_single_sample_is_not_enough(self):
        det = _detector()
        det.observe_gpu(_sample(T0 + 10.5 * MIN, 0))
        assert det.classify(T0 + 11 * MIN) == OK

    def test_busy_but_silent_waits_for_busy_threshold(self):
        det = _detector()
        _feed_gpu(det, T0, T0 + 29 * MIN, util=100)
        assert det.step(T0 + 29 * MIN) is None
        _feed_gpu(det, T0 + 30 * MIN, T0 + 31 * MIN, util=100)
        assert det.step(T0 + 31 * MIN) == (BUSY_SILENT, False)

    def test_unreachable_when_probes_fail(self):
        det = _detector()
        assert det.step(T0 + 11 * MIN) == (UNREACHABLE, False)

    def test_recovers_when_output_resumes(self):
        det = _detector()
        _feed_gpu(det, T0, T0 + 11 * MIN, util=0)
        assert det.step(T0 + 11 * MIN) == (HANG, False)
        det.observe_line(T0 + 12 * MIN)
        assert det.step(T0 + 12 * MIN) == (RECOVERED, False)
        assert det.state == OK

    def test_backlog_after_reconnect_is_ignored(self):
        det = _detector()
        _feed_gpu(det, T0, T0 + 11 * MIN, util=0)
        det.observe_connected(T0 + 10 * MIN)
        det.observe_line(T0 + 10 * MIN + 0.5)  # tail -n 100 replay
        assert det.step(T0 + 11 * MIN) == (HANG, False)

    def test_reminder_repeats_and_can_be_disabled(self):
        det = _detector()
        _feed_gpu(det, T0, T0 + 75 * MIN, util=0)
        assert det.step(T0 + 11 * MIN) == (HANG, False)
        assert det.step(T0 + 40 * MIN) is None
        assert det.step(T0 + 71 * MIN) == (HANG, True)

        quiet = _detector(reminder_minutes=0)
        _feed_gpu(quiet, T0, T0 + 75 * MIN, util=0)
        assert quiet.step(T0 + 11 * MIN) == (HANG, False)
        assert quiet.step(T0 + 75 * MIN) is None

    def test_hang_escalates_to_unreachable(self):
        det = _detector()
        _feed_gpu(det, T0, T0 + 11 * MIN, util=0)
        assert det.step(T0 + 11 * MIN) == (HANG, False)
        # probes stop succeeding
        assert det.step(T0 + 22 * MIN) == (UNREACHABLE, False)


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_hang_includes_costs_from_start_time(self):
        det = _detector()
        _feed_gpu(det, T0, T0 + 30 * MIN, util=0)
        inst = _instance(start_time=T0 - 2.5 * 3600)
        text = describe(HANG, det, inst, T0 + 30 * MIN, first_seen=T0)
        assert "No new output for 30m" in text
        assert "hung" in text
        assert "~$1.00 during this silence" in text
        assert "~$6.00 total (3.0h @ $2.000/hr)" in text

    def test_falls_back_to_watch_time_without_start_time(self):
        det = _detector()
        text = describe(UNREACHABLE, det, _instance(), T0 + 90 * MIN, first_seen=T0)
        assert "since vigil started watching (1.5h" in text

    def test_no_cost_when_price_unknown(self):
        det = _detector()
        text = describe(EXITED, det, _instance(dph_total=0.0), T0 + 30 * MIN, first_seen=T0)
        assert "Cost" not in text
        assert "still billing" in text

    def test_reminder_and_recovered_wording(self):
        det = _detector()
        _feed_gpu(det, T0, T0 + 11 * MIN, util=100)
        assert describe(BUSY_SILENT, det, _instance(), T0 + 2 * 3600, T0, reminder=True).startswith(
            "Still unresolved."
        )
        det.silence_started = T0
        assert describe(RECOVERED, det, _instance(), T0 + 52 * MIN, T0) == (
            "Output resumed after ~52m of silence."
        )
        assert "preempted" in describe(GONE, det, _instance(), T0, T0)


# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _watcher(tmp_path, clock, notify, stream=None, probe=None) -> Watcher:
    config = Config(log_dir=tmp_path, stall_threshold_minutes=10)

    async def idle_stream(**_kwargs):
        await asyncio.Event().wait()

    async def no_probe(_inst, _cfg, _now):
        return None

    return Watcher(
        config,
        MagicMock(),
        clock=clock,
        log=lambda _msg: None,
        notify=notify,
        stream=stream or idle_stream,
        probe=probe or no_probe,
    )


class TestWatcher:
    @pytest.mark.anyio
    async def test_reconcile_starts_and_reports_gone(self, tmp_path):
        sent = []

        async def notify(inst, alert_type, text):
            sent.append((inst.id, alert_type))

        w = _watcher(tmp_path, _Clock(T0), notify)
        await w.reconcile([_instance()])
        assert 42 in w.watched
        tasks = list(w.watched[42].tasks)

        await w.reconcile([])
        assert w.watched == {}
        assert sent == [(42, GONE)]
        await asyncio.gather(*tasks, return_exceptions=True)
        assert all(t.cancelled() for t in tasks)

    @pytest.mark.anyio
    async def test_reconcile_updates_instance_info(self, tmp_path):
        async def notify(*_):
            pass

        w = _watcher(tmp_path, _Clock(T0), notify)
        await w.reconcile([_instance(dph_total=1.0)])
        await w.reconcile([_instance(dph_total=3.0)])
        assert w.watched[42].info.dph_total == 3.0
        for t in w.watched[42].tasks:
            t.cancel()

    @pytest.mark.anyio
    async def test_hang_alert_end_to_end(self, tmp_path):
        clock = _Clock(T0)
        sent = []

        async def notify(inst, alert_type, text):
            sent.append((alert_type, text))

        async def stream(*, on_line, on_status, **_kwargs):
            on_status("connected")
            clock.t += 10  # past the backlog grace
            on_line("step 1 loss 0.5", {})
            await asyncio.Event().wait()

        async def probe(_inst, _cfg, now):
            return _sample(now, 0)

        w = _watcher(tmp_path, clock, notify, stream=stream, probe=probe)
        await w.reconcile([_instance()])
        await asyncio.sleep(0)
        det = w.watched[42].detector
        assert det.last_output == T0 + 10

        # Simulate probe samples across the stall window, then evaluate
        _feed_gpu(det, T0 + MIN, T0 + 12 * MIN, util=0)
        clock.t = T0 + 12 * MIN
        await w.evaluate()
        assert [a for a, _ in sent] == [HANG]
        assert "Training looks hung" in sent[0][1]

        for t in w.watched[42].tasks:
            t.cancel()
        await asyncio.gather(*w.watched[42].tasks, return_exceptions=True)

    @pytest.mark.anyio
    async def test_post_respects_webhook_switch(self, tmp_path):
        w = _watcher(tmp_path, _Clock(T0), None)
        w.config.alert_webhook_url = "https://ntfy.sh/topic"
        w.config.notifications.webhook = False
        with patch("vigil.watch.post_webhook_alert", new=AsyncMock()) as post:
            await w._post(_instance(), HANG, "x")
            post.assert_not_awaited()
            w.config.notifications.webhook = True
            await w._post(_instance(), HANG, "x")
            post.assert_awaited_once()


# ---------------------------------------------------------------------------
# ntfy format
# ---------------------------------------------------------------------------


class TestNtfy:
    @pytest.mark.anyio
    async def test_ntfy_posts_plain_text_with_headers(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post = AsyncMock(return_value=httpx.Response(200))

        await post_webhook_alert(
            "https://ntfy.sh/my-runs",
            42,
            HANG,
            "No new output for 30m — hung",
            instance=_instance(label="llama—ft"),
            format="ntfy",
            client=client,
        )

        kwargs = client.post.call_args[1]
        assert "json" not in kwargs
        assert kwargs["content"].decode() == "No new output for 30m — hung"
        assert kwargs["headers"]["Title"] == "vigil #42: hang (llamaft)"
        assert kwargs["headers"]["Priority"] == "high"
