# SPDX-License-Identifier: Apache-2.0
"""Chunk-level lifecycle event probe for the FS L2 adapter.

Records one JSONL line per observed event so chunk fate can be classified
from a complete lifecycle history instead of end-state snapshots:

  submit  -- a lookup/prefetch submitted its full chunk list
             (request -> chunk correspondence)
  write   -- a chunk was physically written to SSD (file did not exist)
  skip    -- a store found the file already on disk and skipped the write
             (logical duplicate submission, no physical I/O)
  load    -- a chunk was successfully read back from SSD (backend-confirmed)
  delete  -- a chunk file was removed from SSD

Enabled by BIDAW_L2_EVENT_LOG=<path>; otherwise inert.
"""

# Standard
import json
import os
import threading
import time
from pathlib import Path

_lock = threading.Lock()
_fh = None
_path = None


def _log(ev: str, salt: str, chunk_hash: str, req: str = "",
         extra: dict | None = None) -> None:
    if _fh is None:
        return
    rec = {"t": round(time.time(), 3), "ev": ev, "salt": salt,
           "hash": chunk_hash}
    if req:
        rec["req"] = req
    if extra:
        rec.update(extra)
    with _lock:
        _fh.write(json.dumps(rec) + "\n")
        _fh.flush()


def _key_fields(key) -> tuple[str, str]:
    ident = getattr(key, "identity", None)
    ch = getattr(ident, "chunk_hash", "")
    if isinstance(ch, (bytes, bytearray)):
        ch = ch.hex()
    return (getattr(key, "cache_salt", "") or "", str(ch))


def apply_l2_event_probe() -> bool:
    """Wrap FS L2 adapter store/load/delete + StorageManager prefetch
    submission.  Returns True when the probe is active."""
    global _fh, _path
    out = os.environ.get("BIDAW_L2_EVENT_LOG", "")
    if not out:
        return False
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    _path = out
    _fh = open(out, "a", encoding="utf-8")

    # Third Party (LMCache)
    from lmcache.v1.distributed.l2_adapters import fs_l2_adapter as fs_mod
    from lmcache.v1.distributed import storage_manager as sm_mod

    # ---- FS adapter: store (write vs skip) ----
    orig_store = fs_mod.FSL2Adapter._execute_store

    async def execute_store_probed(self, keys, objects, task_id):  # noqa: ANN001
        before = {}
        for k in keys:
            try:
                p = self._key_to_file_and_tmp_path(k)[0]
                before[k] = os.path.exists(p)
            except Exception:  # noqa: BLE001
                before[k] = None
        result = await orig_store(self, keys, objects, task_id)
        for k in keys:
            salt, h = _key_fields(k)
            st = before.get(k)
            ev = "skip" if st else ("write" if st is False else "store?")
            _log(ev, salt, h)
        return result

    # ---- FS adapter: successful backend loads (per-key, from bitmap) ----
    orig_load = fs_mod.FSL2Adapter._execute_load

    async def execute_load_probed(self, keys, objects, task_id):  # noqa: ANN001
        result = await orig_load(self, keys, objects, task_id)
        bitmap = self._completed_load_tasks.get(task_id)
        if bitmap is not None:
            try:
                idx = bitmap.get_indices_list()
            except Exception:  # noqa: BLE001
                idx = []
            for i in idx:
                if 0 <= i < len(keys):
                    salt, h = _key_fields(keys[i])
                    _log("load", salt, h)
        return result

    # ---- FS adapter: deletes ----
    orig_delete = fs_mod.FSL2Adapter._execute_delete

    async def execute_delete_probed(self, keys):  # noqa: ANN001
        removed = await orig_delete(self, keys)
        for item in removed or []:
            if item is None:
                continue
            k = item[0] if isinstance(item, tuple) else item
            salt, h = _key_fields(k)
            _log("delete", salt, h)
        return removed

    # ---- request -> chunk correspondence ----
    orig_submit = sm_mod.StorageManager.submit_prefetch_task

    def submit_prefetch_task_probed(self, spec, external_request_id=""):  # noqa: ANN001
        for group in getattr(spec, "key_groups", []) or []:
            for k in getattr(group, "keys", []) or []:
                salt, h = _key_fields(k)
                _log("submit", salt, h, req=str(external_request_id))
        return orig_submit(self, spec, external_request_id)

    fs_mod.FSL2Adapter._execute_store = execute_store_probed
    fs_mod.FSL2Adapter._execute_load = execute_load_probed
    fs_mod.FSL2Adapter._execute_delete = execute_delete_probed
    sm_mod.StorageManager.submit_prefetch_task = submit_prefetch_task_probed

    _log("probe_on", "", "")
    return True
