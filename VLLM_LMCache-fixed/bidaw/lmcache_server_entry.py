# SPDX-License-Identifier: Apache-2.0
"""Entry point for the LMCache server with Bidaw bolted on.

Usage (from the demo directory, with the bidaw venv activated):

    python -m bidaw.lmcache_server_entry server --host 0.0.0.0 ... \
        --eviction-policy Bidaw

Because the CLI's argparse `choices` for `--eviction-policy` does not know
the name `Bidaw`, this entry rewrites that value to `LRU` (a placeholder the
patched factory ignores) and exports `BIDAW_FORCE_POLICY=Bidaw` so every
policy instance is created as Bidaw.
"""

# Standard
import os
import sys


def _rewrite_argv() -> None:
    if "--eviction-policy" in sys.argv:
        i = sys.argv.index("--eviction-policy")
        if i + 1 < len(sys.argv) and sys.argv[i + 1] == "Bidaw":
            sys.argv[i + 1] = "LRU"
            os.environ["BIDAW_FORCE_POLICY"] = "Bidaw"


def main() -> None:
    from importlib.metadata import version
    installed = version("lmcache")
    if installed != "0.5.7.post1":
        raise RuntimeError(f"Bidaw patches target LMCache 0.5.7.post1, got {installed}")
    _rewrite_argv()
    from bidaw.server_patch import apply_all

    apply_all()

    # LMCache CLI (third party)
    from lmcache.cli.main import main as lmcache_main

    lmcache_main()


if __name__ == "__main__":
    main()
