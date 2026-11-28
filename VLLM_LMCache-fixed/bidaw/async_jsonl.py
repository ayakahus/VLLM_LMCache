# SPDX-License-Identifier: Apache-2.0
"""Batched JSONL logging for the LMCache and vLLM probes."""
import atexit
import json
import logging
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import signal
from threading import Event, RLock, Thread, current_thread, main_thread
import time

logger = logging.getLogger(__name__)
_writers = set()
_previous_sigterm = None
_FLUSH = object()
_STOP = object()


def _flush_on_sigterm(signum, frame):
    for writer in tuple(_writers):
        try:
            writer.flush(timeout=5.0)
        except Exception:
            logger.exception("Could not flush Bidaw timing records on SIGTERM")
    if callable(_previous_sigterm):
        _previous_sigterm(signum, frame)
    elif _previous_sigterm == signal.SIG_DFL:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


class TimingLog:
    def __init__(self, path, *, interval=.15):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        self._queue = SimpleQueue()
        self._interval = interval
        self._pid = os.getpid()
        self._closed = False
        self._control = RLock()
        self._thread = Thread(target=self._drain, name="bidaw-jsonl", daemon=True)
        self._thread.start()
        atexit.register(self.close)
        self._install_signal_handler()

    def _install_signal_handler(self):
        global _previous_sigterm
        if current_thread() is main_thread():
            previous = signal.getsignal(signal.SIGTERM)
            if previous is not _flush_on_sigterm and previous != signal.SIG_IGN:
                _previous_sigterm = previous
                signal.signal(signal.SIGTERM, _flush_on_sigterm)
        _writers.add(self)

    def write(self, record):
        self.write_fields(tuple(record.items()))

    def write_fields(self, fields):
        if not self._closed:
            self._queue.put_nowait(fields)

    def flush(self, *, timeout=None):
        if self._pid != os.getpid():
            return
        with self._control:
            if self._closed or not self._thread.is_alive():
                return
            flushed = Event()
            self._queue.put_nowait((_FLUSH, flushed))
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._thread.is_alive():
            interval = .1 if deadline is None else min(.1, deadline - time.monotonic())
            if interval <= 0 or flushed.wait(interval):
                return

    def close(self):
        # Producers must be stopped before closing. A forked child has no writer thread.
        if self._pid != os.getpid():
            return
        with self._control:
            if self._closed:
                return
            self._closed = True
            self._queue.put_nowait((_STOP, None))
        self._thread.join()
        _writers.discard(self)
        atexit.unregister(self.close)
        os.close(self._fd)

    def _drain(self):
        pending = []
        deadline = None
        while True:
            timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                item = self._queue.get(timeout=timeout)
            except Empty:
                self._append(pending)
                deadline = None
                continue
            if item[0] is _FLUSH:
                self._append(pending)
                item[1].set()
                deadline = None
            elif item[0] is _STOP:
                self._append(pending)
                return
            else:
                pending.append(item)
                if deadline is None:
                    deadline = time.monotonic() + self._interval
                if len(pending) >= 512 or time.monotonic() >= deadline:
                    self._append(pending)
                    deadline = None

    def _append(self, records):
        if not records:
            return
        try:
            payload = "".join(json.dumps(dict(fields)) + "\n" for fields in records).encode()
            remaining = memoryview(payload)
            while remaining:
                written = os.write(self._fd, remaining)
                if not written:
                    raise OSError("Short JSONL append")
                remaining = remaining[written:]
        except (OSError, TypeError, ValueError):
            logger.exception("Could not append Bidaw timing records")
        finally:
            records.clear()
