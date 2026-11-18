# SPDX-License-Identifier: Apache-2.0
"""Monkey-patches that bolt Bidaw onto an unmodified LMCache install.

Two injection points, both applied by `bidaw.lmcache_server_entry` *before*
the server boots:

1. `CreateEvictionPolicy` — the name `Bidaw` is recognised and mapped to
   `BidawEvictionPolicy`.  The symbol is rebound in **both** namespaces that
   imported it (`...eviction_policy.factory` and
   `...storage_controllers.eviction_controller`, the latter because of its
   `from ... import` binding).  Optionally `BIDAW_FORCE_POLICY=<name>`
   overrides every policy creation (used because the CLI's argparse
   `choices` does not know the name `Bidaw`).
2. `lmcache.v1.multiprocess.http_server.app` — the module-level FastAPI app
   is still empty of routes at patch time, so `app.include_router` from
   `bidaw.api` is safe and the routes go live when uvicorn starts.
"""

# Standard
from functools import wraps
import os

# First Party
from bidaw import policy_registry


def _patch_factory() -> None:
    # Third Party (LMCache)
    import lmcache.v1.distributed.eviction_policy.factory as factory_mod
    import lmcache.v1.distributed.storage_controllers.eviction_controller as ctrl_mod

    original = factory_mod.CreateEvictionPolicy
    # Bidaw targets L1. Keep the original L2 replacement policy/geometry.
    import threading
    context = threading.local()
    original_l2_init = ctrl_mod.L2AdapterEvictionState.__init__

    def l2_init(self, *args, **kwargs):
        context.l2 = True
        try:
            original_l2_init(self, *args, **kwargs)
        finally:
            context.l2 = False

    ctrl_mod.L2AdapterEvictionState.__init__ = l2_init

    def patched(config):  # noqa: ANN001
        if getattr(context, "l2", False):
            return original(config)
        name = getattr(config, "eviction_policy", "") or ""
        forced = os.environ.get("BIDAW_FORCE_POLICY", "").strip()
        if forced:
            name = forced
        if name == "Bidaw":
            # Local import avoids a circular import at module load time.
            from bidaw.eviction_policy.bidaw_policy import BidawEvictionPolicy

            policy = BidawEvictionPolicy()
            policy_registry.register(policy)
            return policy
        return original(config)

    factory_mod.CreateEvictionPolicy = patched
    # Rebind the from-import binding used by both controllers.
    ctrl_mod.CreateEvictionPolicy = patched


def _patch_http() -> None:
    # Third Party (LMCache)
    from lmcache.v1.multiprocess import http_server as hs

    from bidaw.api import router

    if not any(getattr(r, "path", "").startswith("/bidaw") for r in hs.app.routes):
        hs.app.include_router(router)


def _patch_conversation_lookup() -> None:
    from lmcache.utils import get_size_bytes
    from lmcache.v1.multiprocess.modules.lookup import LookupModule

    original = LookupModule.lookup

    @wraps(original)  # preserve the MP handler registration metadata
    def lookup(self, key, tp_size):
        if key.cache_salt:
            ctx = self.context
            layouts = ctx.layout_desc_registry.find_group_layout_descs(
                key.model_name, key.world_size
            )
            if layouts:
                history_tokens = (key.request_configs or {}).get("lmcache.bidaw_history_tokens")
                if type(history_tokens) is not int or not 0 <= history_tokens <= key.num_tokens:
                    history_tokens = key.num_tokens
                chunk_bytes = key.world_size * sum(
                    get_size_bytes(layout.shapes, layout.dtypes) for layout in layouts.values()
                )
                policy_registry.feed_conversation(
                    key.cache_salt,
                    history_tokens * chunk_bytes // ctx.chunk_size,
                    key.request_id,
                )
        return original(self, key, tp_size)

    LookupModule.lookup = lookup


def _patch_dual_queue() -> None:
    # Bidaw Phase 1: I/O-aware dual-queue prefetch scheduling.  Controlled
    # by BIDAW_DUAL_QUEUE (default on; "0" keeps the native FIFO).
    from bidaw.dual_queue_scheduler import apply_dual_queue_patch

    apply_dual_queue_patch()


def _patch_watermark() -> None:
    """Paper section 5.6: an eviction is triggered when free host memory
    falls below 5% of capacity -> trigger_watermark = 0.95.  The native CLI
    default is 0.8; wrap ``parse_args_to_config`` (function-level import in
    ``cli/commands/server.py`` picks up the patched module binding) to stamp
    the paper value onto every eviction config.  Controlled by
    ``BIDAW_EVICTION_WATERMARK``."""
    watermark = os.environ.get("BIDAW_EVICTION_WATERMARK", "0.95")
    if watermark == "native":
        return
    try:
        wm = float(watermark)
    except ValueError as exc:
        raise ValueError("Invalid BIDAW_EVICTION_WATERMARK") from exc
    if not 0 < wm <= 1:
        raise ValueError("BIDAW_EVICTION_WATERMARK must be in (0, 1]")

    # Third Party (LMCache)
    import lmcache.v1.distributed.config as dcfg

    if getattr(dcfg.parse_args_to_config, "_bidaw_wm", False):
        return
    original = dcfg.parse_args_to_config

    def patched(args):  # noqa: ANN001
        cfg = original(args)
        for ev_cfg in [cfg.eviction_config] + [
            m.eviction for m in getattr(cfg, "l1_manager_configs", []) or []
        ]:
            if ev_cfg is not None:
                try:
                    object.__setattr__(ev_cfg, "trigger_watermark", wm)
                except Exception:  # noqa: BLE001
                    ev_cfg.trigger_watermark = wm
        return cfg

    patched._bidaw_wm = True  # type: ignore[attr-defined]
    dcfg.parse_args_to_config = patched


def _patch_l2_event_probe() -> None:
    # Diagnostic chunk-lifecycle probe (BIDAW_L2_EVENT_LOG=<jsonl path>).
    from bidaw.l2_event_probe import apply_l2_event_probe

    apply_l2_event_probe()


def apply_all() -> None:
    if os.environ.get("BIDAW_PAD_TAIL") == "1":
        from bidaw.padded_tail import apply_padded_tail_store

        apply_padded_tail_store()
    _patch_factory()
    _patch_http()
    if os.environ.get("BIDAW_FORCE_POLICY") == "Bidaw":
        _patch_conversation_lookup()
    _patch_l2_event_probe()
    _patch_dual_queue()
    _patch_watermark()
