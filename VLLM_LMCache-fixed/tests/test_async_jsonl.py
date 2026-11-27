from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from bidaw.async_jsonl import TimingLog


class TimingLogTests(unittest.TestCase):
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
