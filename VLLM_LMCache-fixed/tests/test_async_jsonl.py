from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from threading import Event, Thread
import unittest
from unittest.mock import Mock, patch

from bidaw import async_jsonl
from bidaw.async_jsonl import TimingLog


class TimingLogTests(unittest.TestCase):
    def test_flush_returns_when_writer_is_already_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(TimingLog, "_drain"):
                log = TimingLog(Path(directory, "timing.jsonl"))
            self.addCleanup(log.close)
            log._thread.join(2)
            self.assertFalse(log._thread.is_alive())
            caller = Thread(target=log.flush, daemon=True)
            caller.start()
            caller.join(2)
            self.assertFalse(caller.is_alive())

    def test_flush_timeout_returns_while_writer_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            log = TimingLog(Path(directory, "timing.jsonl"), interval=60)
            self.addCleanup(log.close)
            writing = Event()
            release = Event()
            append = log._append

            def blocked_append(records):
                writing.set()
                release.wait(5)
                append(records)

            log.write({"event": "store"})
            with patch.object(log, "_append", side_effect=blocked_append):
                caller = Thread(target=lambda: log.flush(timeout=.05), daemon=True)
                caller.start()
                try:
                    self.assertTrue(writing.wait(2))
                    caller.join(2)
                    self.assertFalse(caller.is_alive())
                finally:
                    release.set()
                    caller.join(2)

    def test_sigterm_calls_previous_handler_even_when_flush_raises(self):
        writer = Mock()
        writer.flush.side_effect = RuntimeError("writer failed")
        previous = Mock()
        with (
            patch.object(async_jsonl, "_writers", (writer,)),
            patch.object(async_jsonl, "_previous_sigterm", previous),
            self.assertLogs("bidaw.async_jsonl", level="ERROR"),
        ):
            async_jsonl._flush_on_sigterm(signal.SIGTERM, None)
        writer.flush.assert_called_once_with(timeout=5.0)
        previous.assert_called_once_with(signal.SIGTERM, None)

    def test_flush_and_close_preserve_all_records_from_parallel_producers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "timing.jsonl")
            log = TimingLog(path, interval=60)
            self.addCleanup(log.close)

            def produce(worker):
                for index in range(100):
                    log.write_fields((("worker", worker), ("index", index)))

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(produce, range(8)))
            log.flush()

            def records():
                return [json.loads(line) for line in path.read_text().splitlines()]

            self.assertCountEqual(records(), [{"worker": worker, "index": index}
                                              for worker in range(8) for index in range(100)])
            log.write({"final": 1})
            log.close()
            self.assertEqual(len(records()), 801)
            self.assertEqual(records()[-1], {"final": 1})

    def test_sigterm_and_normal_exit_flush_without_explicit_close(self):
        program = """
import os
import signal
import sys
from bidaw.async_jsonl import TimingLog
log = TimingLog(sys.argv[1], interval=60)
log.write({"event": "store", "id": 1})
if sys.argv[2] == "signal":
    os.kill(os.getpid(), signal.SIGTERM)
"""
        with tempfile.TemporaryDirectory() as directory:
            for mode, returncode in (("normal", 0), ("signal", -15)):
                with self.subTest(mode=mode):
                    path = Path(directory, mode)
                    result = subprocess.run([sys.executable, "-c", program, str(path), mode], timeout=5)
                    self.assertEqual(result.returncode, returncode)
                    self.assertEqual(json.loads(path.read_text()), {"event": "store", "id": 1})
