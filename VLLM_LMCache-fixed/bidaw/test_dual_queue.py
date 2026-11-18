#!/usr/bin/env python3
"""Run the CPU regression tests for Bidaw's dual queue."""
import sys
import unittest
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
    unittest.main(module='test_regressions', argv=[sys.argv[0], 'SchedulerTests', *sys.argv[1:]])
