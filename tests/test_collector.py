from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import asyncssh
import pytest

from vigil.collector import (
    LineSplitter,
    _read_stderr,
    _stream_one_session,
    _write_to_storage,
    stream_instance_logs,
)
from vigil.config import Config
from vigil.discovery import InstanceInfo
from vigil.parser import MetricParser
from vigil.storage import LogStorage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def instance():
    return InstanceInfo(
        id=42,
        ssh_host="1.2.3.4",
        ssh_port=22,
        gpu_name="RTX 4090",
        num_gpus=1,
        status="running",
        machine_id=100,
    )


@pytest.fixture()
def config():
    cfg = Config()
    cfg.ssh_login_timeout = 5
    cfg.reconnect_backoff_max = 8
    cfg.stall_threshold_minutes = 2
    return cfg


@pytest.fixture()
def storage():
    s = MagicMock(spec=LogStorage)
    return s


@pytest.fixture()
def parser():
    p = MagicMock(spec=MetricParser)
    p.parse_line = MagicMock(return_value={})
    return p


# ===========================================================================
# _write_to_storage
# ===========================================================================


class TestWriteToStorage:
    def test_happy_path_with_metrics(self, storage, parser):
        parser.parse_line.return_value = {"loss": "0.42"}
        result = _write_to_storage(storage, parser, 42, "step=10 loss=0.42")

        storage.write_line.assert_called_once_with(42, "step=10 loss=0.42")
        parser.parse_line.assert_called_once_with("step=10 loss=0.42")
        storage.write_metrics.assert_called_once()
        assert storage.write_metrics.call_args[0][0] == 42
        assert storage.write_metrics.call_args[0][1] == {"loss": "0.42"}
        assert result == {"loss": "0.42"}

    def test_happy_path_no_metrics(self, storage, parser):
        parser.parse_line.return_value = {}
        result = _write_to_storage(storage, parser, 42, "some random line")

        storage.write_line.assert_called_once_with(42, "some random line")
        storage.write_metrics.assert_not_called()
        assert result == {}

    def test_oserror_on_write_line_still_parses(self, storage, parser):
        storage.write_line.side_effect = OSError("disk full")
        parser.parse_line.return_value = {"loss": "1.0"}

        result = _write_to_storage(storage, parser, 42, "loss=1.0")

        parser.parse_line.assert_called_once()
        storage.write_metrics.assert_called_once()
        assert result == {"loss": "1.0"}

    def test_oserror_on_write_metrics_returns_metrics(self, storage, parser):
        parser.parse_line.return_value = {"loss": "0.5"}
        storage.write_metrics.side_effect = OSError("disk full")

        result = _write_to_storage(storage, parser, 42, "loss=0.5")

        storage.write_line.assert_called_once()
        assert result == {"loss": "0.5"}

    def test_parser_returns_empty_dict(self, storage, parser):
        parser.parse_line.return_value = {}
        result = _write_to_storage(storage, parser, 7, "no metrics here")

        storage.write_line.assert_called_once_with(7, "no metrics here")
        storage.write_metrics.assert_not_called()
        assert result == {}


# ===========================================================================
# _read_stderr
# ===========================================================================


class TestReadStderr:
    @pytest.mark.anyio()
    async def test_returns_string_data(self):
        process = MagicMock()
        process.stderr = MagicMock()
        process.stderr.read = AsyncMock(return_value="some error text")

        result = await _read_stderr(process)
        assert result == "some error text"

    @pytest.mark.anyio()
    async def test_returns_decoded_bytes(self):
        process = MagicMock()
        process.stderr = MagicMock()
        process.stderr.read = AsyncMock(return_value=b"binary error")

        result = await _read_stderr(process)
        assert result == "binary error"

    @pytest.mark.anyio()
    async def test_no_stderr_returns_empty(self):
        process = MagicMock()
        process.stderr = None

        result = await _read_stderr(process)
        assert result == ""

    @pytest.mark.anyio()
    async def test_timeout_returns_empty(self):
        process = MagicMock()
        process.stderr = MagicMock()

        async def hang(_size):
            await asyncio.sleep(100)

        process.stderr.read = hang

        result = await _read_stderr(process)
        assert result == ""

    @pytest.mark.anyio()
    async def test_exception_returns_empty(self):
        process = MagicMock()
        process.stderr = MagicMock()
        process.stderr.read = AsyncMock(side_effect=RuntimeError("boom"))

        result = await _read_stderr(process)
        assert result == ""


# ===========================================================================
# Helpers for _stream_one_session / stream_instance_logs
# ===========================================================================


def _make_mock_conn(lines: list[str], returncode: int = 0, stderr_data: str = ""):
    """Build a mock SSH connection that yields *lines* from stdout."""
    # stdout mock: read returns each line as a chunk, then "" to signal EOF
    chunks = [line + "\n" for line in lines] + [""]
    mock_stdout = MagicMock()
    mock_stdout.read = AsyncMock(side_effect=chunks)

    # stderr mock
    mock_stderr = MagicMock()
    mock_stderr.read = AsyncMock(return_value=stderr_data)

    # process mock (used as async context manager)
    mock_process = MagicMock()
    mock_process.stdout = mock_stdout
    mock_process.stderr = mock_stderr
    mock_process.returncode = returncode

    # process async context manager
    mock_process.__aenter__ = AsyncMock(return_value=mock_process)
    mock_process.__aexit__ = AsyncMock(return_value=False)

    # connection mock (used as async context manager)
    mock_conn = MagicMock()
    mock_conn.create_process = MagicMock(return_value=mock_process)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    return mock_conn


# ===========================================================================
# _stream_one_session
# ===========================================================================


class TestStreamOneSession:
    @pytest.mark.anyio()
    @patch("vigil.collector.ssh_connect", new_callable=AsyncMock)
    async def test_happy_path_lines_flow(self, mock_ssh_connect, instance, config, storage, parser):
        lines = ["step=1 loss=0.9", "step=2 loss=0.8", "step=3 loss=0.7"]
        mock_conn = _make_mock_conn(lines, returncode=0)
        mock_ssh_connect.return_value = mock_conn

        on_line = MagicMock()
        on_status = MagicMock()
        update_line_time = MagicMock()

        await _stream_one_session(instance, config, storage, parser, on_line, on_status, update_line_time)

        assert on_line.call_count == 3
        assert on_status.call_args_list[0].args[0] == "connecting..."
        assert on_status.call_args_list[1].args[0] == "connected"
        # Final status should contain "process exited (rc=0)"
        final_status = on_status.call_args_list[-1].args[0]
        assert "rc=0" in final_status
        assert update_line_time.call_count == 3
        # flush called at end of loop
        storage.flush.assert_called_with(42)

    @pytest.mark.anyio()
    @patch("vigil.collector.ssh_connect", new_callable=AsyncMock)
    async def test_flushes_every_100_lines(self, mock_ssh_connect, instance, config, storage, parser):
        lines = [f"line {i}" for i in range(150)]
        mock_conn = _make_mock_conn(lines, returncode=0)
        mock_ssh_connect.return_value = mock_conn

        on_line = MagicMock()
        on_status = MagicMock()
        update_line_time = MagicMock()

        await _stream_one_session(instance, config, storage, parser, on_line, on_status, update_line_time)

        # flush called at line 100, then at end-of-stream
        flush_calls = [c for c in storage.flush.call_args_list if c.args == (42,)]
        assert len(flush_calls) == 2

    @pytest.mark.anyio()
    @patch("vigil.collector.ssh_connect", new_callable=AsyncMock)
    async def test_connect_timeout_raises(self, mock_ssh_connect, instance, config, storage, parser):
        async def slow_connect(*_args, **_kwargs):
            await asyncio.sleep(100)

        mock_ssh_connect.side_effect = slow_connect
        config.ssh_login_timeout = 0  # timeout = 0 + 10 = 10, but we use a very slow connect

        on_line = MagicMock()
        on_status = MagicMock()
        update_line_time = MagicMock()

        # Override timeout to be very short
        with patch("vigil.collector.asyncio.wait_for", side_effect=asyncio.TimeoutError):
            with pytest.raises(asyncio.TimeoutError):
                await _stream_one_session(instance, config, storage, parser, on_line, on_status, update_line_time)

        on_status.assert_any_call("connecting...")

    @pytest.mark.anyio()
    @patch("vigil.collector.ssh_connect", new_callable=AsyncMock)
    async def test_read_timeout_stall(self, mock_ssh_connect, instance, config, storage, parser):
        mock_stdout = MagicMock()
        # First read succeeds, second times out
        mock_stdout.read = AsyncMock(side_effect=["line1\n", asyncio.TimeoutError()])

        mock_stderr = MagicMock()
        mock_stderr.read = AsyncMock(return_value="")

        mock_process = MagicMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.returncode = 0
        mock_process.__aenter__ = AsyncMock(return_value=mock_process)
        mock_process.__aexit__ = AsyncMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.create_process = MagicMock(return_value=mock_process)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        mock_ssh_connect.return_value = mock_conn

        on_line = MagicMock()
        on_status = MagicMock()
        update_line_time = MagicMock()

        with pytest.raises(asyncio.TimeoutError):
            await _stream_one_session(instance, config, storage, parser, on_line, on_status, update_line_time)

        on_status.assert_any_call("stalled — no output, reconnecting")

    @pytest.mark.anyio()
    @patch("vigil.collector.ssh_connect", new_callable=AsyncMock)
    async def test_nonzero_exit_code_raises_oserror(self, mock_ssh_connect, instance, config, storage, parser):
        mock_conn = _make_mock_conn(["one line"], returncode=1, stderr_data="segfault")
        mock_ssh_connect.return_value = mock_conn

        on_line = MagicMock()
        on_status = MagicMock()
        update_line_time = MagicMock()

        with pytest.raises(OSError, match="rc=1"):
            await _stream_one_session(instance, config, storage, parser, on_line, on_status, update_line_time)

        final_status = on_status.call_args_list[-1].args[0]
        assert "rc=1" in final_status
        assert "segfault" in final_status

    @pytest.mark.anyio()
    @patch("vigil.collector.ssh_connect", new_callable=AsyncMock)
    async def test_strips_newlines_from_lines(self, mock_ssh_connect, instance, config, storage, parser):
        mock_conn = _make_mock_conn(["hello"], returncode=0)
        mock_ssh_connect.return_value = mock_conn

        on_line = MagicMock()
        on_status = MagicMock()
        update_line_time = MagicMock()

        await _stream_one_session(instance, config, storage, parser, on_line, on_status, update_line_time)

        # on_line should receive stripped line (no trailing \n)
        on_line.assert_called_once()
        assert on_line.call_args.args[0] == "hello"


# ===========================================================================
# stream_instance_logs
# ===========================================================================


class TestStreamInstanceLogs:
    @pytest.mark.anyio()
    @patch("vigil.collector._stream_one_session", new_callable=AsyncMock)
    @patch("vigil.collector.asyncio.sleep", new_callable=AsyncMock)
    async def test_backoff_on_ssh_error(self, mock_sleep, mock_session, instance, config, storage, parser):
        call_count = 0

        async def fail_then_cancel(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise asyncssh.Error(code=1, reason="connection refused")
            raise asyncio.CancelledError

        mock_session.side_effect = fail_then_cancel

        on_line = MagicMock()
        on_status = MagicMock()
        on_stall = MagicMock()

        with pytest.raises(asyncio.CancelledError):
            await stream_instance_logs(instance, config, storage, parser, on_line, on_status, on_stall)

        # Check exponential backoff: 1, 2, 4
        sleep_backoffs = [c.args[0] for c in mock_sleep.call_args_list if c.args[0] != 30]
        assert sleep_backoffs[:3] == [1, 2, 4]

        # on_status should contain "disconnected" messages
        status_calls = [c.args[0] for c in on_status.call_args_list]
        disconnect_msgs = [s for s in status_calls if "disconnected" in s]
        assert len(disconnect_msgs) == 3

    @pytest.mark.anyio()
    @patch("vigil.collector._stream_one_session", new_callable=AsyncMock)
    @patch("vigil.collector.asyncio.sleep", new_callable=AsyncMock)
    async def test_backoff_on_generic_exception(self, mock_sleep, mock_session, instance, config, storage, parser):
        call_count = 0

        async def fail_then_cancel(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("unexpected")
            raise asyncio.CancelledError

        mock_session.side_effect = fail_then_cancel

        on_line = MagicMock()
        on_status = MagicMock()
        on_stall = MagicMock()

        with pytest.raises(asyncio.CancelledError):
            await stream_instance_logs(instance, config, storage, parser, on_line, on_status, on_stall)

        status_calls = [c.args[0] for c in on_status.call_args_list]
        error_msgs = [s for s in status_calls if "error (ValueError" in s]
        assert len(error_msgs) == 2

        sleep_backoffs = [c.args[0] for c in mock_sleep.call_args_list if c.args[0] != 30]
        assert sleep_backoffs[:2] == [1, 2]

    @pytest.mark.anyio()
    @patch("vigil.collector._stream_one_session", new_callable=AsyncMock)
    @patch("vigil.collector.asyncio.sleep", new_callable=AsyncMock)
    async def test_backoff_caps_at_max(self, mock_sleep, mock_session, instance, config, storage, parser):
        config.reconnect_backoff_max = 4
        call_count = 0

        async def fail_then_cancel(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                raise OSError("nope")
            raise asyncio.CancelledError

        mock_session.side_effect = fail_then_cancel

        on_line = MagicMock()
        on_status = MagicMock()

        with pytest.raises(asyncio.CancelledError):
            await stream_instance_logs(instance, config, storage, parser, on_line, on_status)

        sleep_backoffs = [c.args[0] for c in mock_sleep.call_args_list if c.args[0] != 30]
        # 1, 2, 4, 4, 4 — capped at max_backoff=4
        assert sleep_backoffs[:5] == [1, 2, 4, 4, 4]

    @pytest.mark.anyio()
    @patch("vigil.collector._stream_one_session", new_callable=AsyncMock)
    async def test_cancellation_propagates(self, mock_session, instance, config, storage, parser):
        mock_session.side_effect = asyncio.CancelledError

        on_line = MagicMock()
        on_status = MagicMock()

        with pytest.raises(asyncio.CancelledError):
            await stream_instance_logs(instance, config, storage, parser, on_line, on_status)

    @pytest.mark.anyio()
    @patch("vigil.collector._stream_one_session", new_callable=AsyncMock)
    @patch("vigil.collector.asyncio.sleep", new_callable=AsyncMock)
    async def test_backoff_resets_after_success(self, mock_sleep, mock_session, instance, config, storage, parser):
        call_count = 0

        async def sequence(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("fail 1")  # backoff=1
            if call_count == 2:
                raise OSError("fail 2")  # backoff=2
            if call_count == 3:
                return  # success -> backoff resets
            if call_count == 4:
                raise OSError("fail 3")  # backoff should be 1 again
            raise asyncio.CancelledError

        mock_session.side_effect = sequence

        on_line = MagicMock()
        on_status = MagicMock()

        with pytest.raises(asyncio.CancelledError):
            await stream_instance_logs(instance, config, storage, parser, on_line, on_status)

        sleep_backoffs = [c.args[0] for c in mock_sleep.call_args_list if c.args[0] != 30]
        # 1, 2, <success resets>, 1
        assert sleep_backoffs == [1, 2, 1]

    @pytest.mark.anyio()
    async def test_stall_detection(self, instance, config, storage, parser):
        config.stall_threshold_minutes = 1  # 1 minute threshold

        on_line = MagicMock()
        on_status = MagicMock()
        on_stall = MagicMock()

        # Control time.monotonic to simulate a stall without real waiting.
        mono_time = [0.0]

        def fake_monotonic():
            return mono_time[0]

        # Save reference to real sleep before patching to avoid recursion.
        _real_sleep = asyncio.sleep
        stall_check_count = [0]

        async def fast_sleep(seconds, **kwargs):
            if seconds == 30:
                stall_check_count[0] += 1
                if stall_check_count[0] == 1:
                    # Advance fake time past the stall threshold
                    mono_time[0] = 120.0  # 2 minutes > 1 minute threshold
            # Yield control via the real sleep to avoid infinite recursion
            await _real_sleep(0)

        session_call_count = [0]

        async def session_with_stall_window(*args, **kwargs):
            session_call_count[0] += 1
            if session_call_count[0] == 1:
                # Yield control so _check_stall can run its loop and detect the stall
                for _ in range(10):
                    await _real_sleep(0)
                raise asyncio.TimeoutError("stalled")
            # Second call: cancel the whole thing
            raise asyncio.CancelledError

        with (
            patch("vigil.collector.time.monotonic", side_effect=fake_monotonic),
            patch("vigil.collector.asyncio.sleep", side_effect=fast_sleep),
            patch("vigil.collector._stream_one_session", side_effect=session_with_stall_window),
        ):
            with pytest.raises(asyncio.CancelledError):
                await stream_instance_logs(instance, config, storage, parser, on_line, on_status, on_stall)

        on_stall.assert_called_once()
        stall_minutes = on_stall.call_args.args[0]
        assert stall_minutes >= 1

        status_calls = [c.args[0] for c in on_status.call_args_list]
        stall_msgs = [s for s in status_calls if "STALLED" in s]
        assert len(stall_msgs) >= 1


# ===========================================================================
# LineSplitter / progress bars
# ===========================================================================


class TestLineSplitter:
    def test_splits_all_line_endings(self):
        sp = LineSplitter()
        assert sp.feed("a\nb\r\nc\rd") == [("a", False), ("b", False), ("c", True)]
        assert sp.flush() == [("d", False)]

    def test_crlf_split_across_chunks_is_one_newline(self):
        sp = LineSplitter()
        assert sp.feed("step 1\r") == []
        assert sp.feed("\nstep 2\n") == [("step 1", False), ("step 2", False)]

    def test_tqdm_redraws_are_progress(self):
        sp = LineSplitter()
        out = sp.feed("\r 10%|#  | 10/100\r 20%|## | 20/100")
        assert out == [("", True), (" 10%|#  | 10/100", True)]
        assert sp.flush() == [(" 20%|## | 20/100", False)]

    def test_trailing_cr_flushes_as_progress(self):
        sp = LineSplitter()
        assert sp.feed("50%\r") == []
        assert sp.flush() == [("50%", True)]

    def test_oversized_line_is_emitted(self):
        sp = LineSplitter()
        out = sp.feed("x" * 70000)
        assert out == [("x" * 70000, False)]


class TestProgressBars:
    @pytest.mark.anyio()
    @patch("vigil.collector.ssh_connect", new_callable=AsyncMock)
    async def test_progress_counts_as_activity_but_is_throttled(
        self, mock_ssh_connect, instance, config, storage, parser
    ):
        bar = "".join(f"\r{i}%|loss=0.{i}" for i in range(1, 50))
        mock_conn = _make_mock_conn([])
        mock_conn.create_process.return_value.stdout.read = AsyncMock(
            side_effect=["epoch 1\n", bar, "\ndone\n", ""]
        )
        mock_ssh_connect.return_value = mock_conn

        on_line = MagicMock()
        update_line_time = MagicMock()
        await _stream_one_session(instance, config, storage, parser, on_line, MagicMock(), update_line_time)

        emitted = [c.args[0] for c in on_line.call_args_list]
        # first redraw is logged, the rest within 10s are skipped; the final
        # state of the bar arrives with its newline
        assert emitted == ["epoch 1", "1%|loss=0.1", "49%|loss=0.49", "done"]
        # every redraw still resets the stall timer
        assert update_line_time.call_count == 1 + 49 + 1 + 1
