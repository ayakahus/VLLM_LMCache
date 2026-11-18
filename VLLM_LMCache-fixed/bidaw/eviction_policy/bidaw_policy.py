# SPDX-License-Identifier: Apache-2.0
"""Bidaw eviction policy (FAST'26) for LMCache MP servers.

Implements the previous-answer-based eviction strategy of
``Bidaw: Enhancing Key-Value Caching for Interactive LLM Serving via
Bidirectional Computation-Storage Awareness`` (Hu et al., FAST'26):

* Per-user (``cache_salt``) weighted-reuse-distance (WRD) tracking, updated
  from logical lookup submissions, independently of L1 lifecycle events.
* Online lower-bound curve: the minimum observed WRD per previous-round
  answer-length bin (paper section 3.3.1, Spearman 0.94-0.98 observation).
* A background ghost cache that replays the recent access trace through
  Belady's optimal algorithm to estimate, per WRD bucket, the hit rate an
  optimal strategy would achieve (paper section 3.3.2/3.3.3).
* Eviction candidates are selected at USER granularity: the salt whose next
  access has the lowest predicted hit potential (paper eq. 2) is evicted
  first, whole histories at a time.

Units: WRD is tracked in "distinct equal-sized keys" (uniform chunk size makes bytes and
key counts monotonically equivalent); the small-bucket threshold is the L1
capacity expressed in keys (``BIDAW_L1_CAPACITY_KEYS``).
"""

# Standard
import heapq
import math
import os
import threading
import time
from collections import defaultdict, deque

from bidaw.reuse_distance import ReuseDistance

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction import EvictionPolicy
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)

logger = init_logger(__name__)

_LB_BIN_TOKENS = 32  # answer-length bin width (tokens)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class BidawEvictionPolicy(EvictionPolicy):
    """Previous-answer-based eviction with ghost-cache hit potentials."""

    def __init__(self) -> None:
        # --- configuration (env-overridable) ---
        self._capacity_keys = max(16, _env_int("BIDAW_L1_CAPACITY_KEYS", 560))
        self._ghost_window = max(256, _env_int("BIDAW_GHOST_WINDOW", 8000))
        self._ghost_interval = max(1.0, _env_float("BIDAW_GHOST_INTERVAL", 5.0))
        self._extreme_mult = max(2.0, _env_float("BIDAW_EXTREME_MULT", 6.0))
        self._promising_buckets = max(2, _env_int("BIDAW_PROMISING_BUCKETS", 8))
        self._grace_seconds = max(0.0, _env_float("BIDAW_GRACE_SECONDS", 0.0))
        self._extreme_threshold = self._capacity_keys * self._extreme_mult

        # --- shared state ---
        self._lock = threading.RLock()
        self._seq = 0
        self._reuse = ReuseDistance(_env_int("BIDAW_HISTORY_KEYS", 200_000))
        self._ghost_errors = 0
        self._ghost_last_error = None
        self._lb_samples = defaultdict(lambda: deque(maxlen=512))
        self._logical_requests = 0
        self._tracked: dict[ObjectKey, str] = {}      # key -> salt
        self._salt_last_touch: dict[str, float] = {}  # salt -> wall clock
        self._salt_hist: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=64)
        )
        self._salt_last_answer: dict[str, int] = {}
        self._lb_min: dict[int, float] = {}  # answer bin -> min WRD
        self._trace: deque = deque(maxlen=self._ghost_window)
        self._ghost = None                   # (small, [prom...], built_at)
        self._ghost_built_seq = 0
        self._stop = threading.Event()
        self._ghost_thread: threading.Thread | None = None
        self._destinations: list[EvictionDestination] = []

    # ------------------------------------------------------------------
    # EvictionPolicy interface
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "bidaw_eviction_policy"

    def register_eviction_destination(
        self, destination: EvictionDestination
    ) -> None:
        with self._lock:
            self._destinations.append(destination)

    def on_keys_created(self, keys: list[ObjectKey]) -> None:
        with self._lock:
            for key in keys:
                self._tracked[key] = key.cache_salt or ""

    def on_keys_touched(self, keys: list[ObjectKey]) -> None:
        with self._lock:
            for key in keys:
                self._tracked[key] = key.cache_salt or ""

    def on_keys_removed(self, keys: list[ObjectKey]) -> None:
        with self._lock:
            for key in keys:
                self._tracked.pop(key, None)
                # Reuse history survives L1 eviction.

    def get_eviction_actions(
        self,
        expected_ratio: float,
        key_eligible_filter=None,
        cache_salt: str | None = None,
    ) -> list[EvictionAction]:
        with self._lock:
            by_salt: dict[str, list[ObjectKey]] = defaultdict(list)
            for key, salt in self._tracked.items():
                if cache_salt is not None and salt != cache_salt:
                    continue
                if key_eligible_filter is not None:
                    try:
                        if not key_eligible_filter(key):
                            continue
                    except Exception:  # noqa: BLE001
                        continue
                by_salt[salt].append(key)
            if not by_salt:
                return []

            total = sum(len(v) for v in by_salt.values())
            if not 0 < expected_ratio <= 1:
                return []
            target = max(1, math.ceil(expected_ratio * total))
            now = time.monotonic()

            scored = []
            for salt, keys in by_salt.items():
                last_touch = self._salt_last_touch.get(salt, 0.0)
                grace_ok = (now - last_touch) >= self._grace_seconds
                potential = self._potential(salt, respect_lb=grace_ok)
                scored.append((grace_ok, potential, last_touch, salt, keys))

            # lowest potential first; non-recent (out-of-grace) salts before
            # protected ones; last-touch time as final tie-breaker.
            scored.sort(key=lambda t: (not t[0], t[1], t[2]))

            actions: list[EvictionAction] = []
            evicted = 0
            for _grace, _pot, _lt, _salt, keys in scored:
                if evicted >= target:
                    break
                actions.append(
                    EvictionAction(
                        destination=EvictionDestination.DISCARD,
                        keys=list(keys),
                    )
                )
                evicted += len(keys)
            if actions:
                logger.info(
                    "Bidaw eviction: keys=%d salts=%d target=%d -> evicted "
                    "%d keys; salt potentials: %s",
                    total,
                    len(by_salt),
                    target,
                    evicted,
                    {s: round(p, 3) for _g, p, _lt, s, _k in scored[:6]},
                )
            return actions

    # ------------------------------------------------------------------
    # Bidaw-specific API (invoked via policy_registry / HTTP)
    # ------------------------------------------------------------------
    def on_answer(self, cache_salt: str, answer_tokens: int) -> None:
        """Compute engine reported: user `cache_salt` received `answer_tokens`."""
        with self._lock:
            self._salt_last_answer[cache_salt] = max(0, int(answer_tokens))

    def status(self) -> dict:
        with self._lock:
            ghost = None
            if self._ghost is not None:
                small, prom, built = self._ghost
                ghost = {
                    "built_at_seq": self._ghost_built_seq,
                    "small_bucket_hit": round(small, 3),
                    "promising_bucket_hits": [round(p, 3) for p in prom],
                }
            return {
                "policy": "Bidaw",
                "units": "distinct_equal_sized_chunks",
                "logical_requests": self._logical_requests,
                "history_identities": len(self._reuse.last),
                "ghost_errors": self._ghost_errors,
                "ghost_last_error": self._ghost_last_error,
                "grace_seconds": self._grace_seconds,
                "tracked_keys": len(self._tracked),
                "salts": len(set(self._tracked.values())),
                # diagnostic: resident-at-end identities so post-hoc analysis
                # can split never-SSD-read chunks into A (still L1-resident)
                # vs B (evicted, never read). Kept short (bounded by L1 cap).
                "resident_identities": [
                    {"salt": k.cache_salt,
                     "hash": k.identity.chunk_hash.hex()
                     if isinstance(k.identity.chunk_hash, (bytes, bytearray))
                     else str(k.identity.chunk_hash)}
                    for k in list(self._tracked)[:2000]
                ],
                "seq": self._seq,
                "capacity_keys": self._capacity_keys,
                "last_answers": dict(list(self._salt_last_answer.items())[-8:]),
                "lb_curve": {
                    str(k): round(v, 1)
                    for k, v in sorted(self._lb_min.items())[:16]
                },
                "ghost": ghost,
            }

    # ------------------------------------------------------------------
    # Internal: access recording + WRD / lower-bound accounting
    # ------------------------------------------------------------------
    def on_request_access(self, keys: list[ObjectKey]) -> None:
        """One logical prefetch submission, including L1/L2 misses.

        Creation, reload and lock callbacks must not train the predictor.
        Deduplicate identities within the submitted request only.
        """
        with self._lock:
            self._logical_requests += 1
            now = time.monotonic()
            seen = set()
            for key in keys:
                kid = self._key_id(key)
                if kid in seen:
                    continue
                seen.add(kid)
                salt = key.cache_salt or ""
                self._seq += 1
                wrd = self._reuse.access(kid)
                if wrd is not None:
                    self._record_wrd(salt, float(wrd))
                self._salt_last_touch[salt] = now
                self._trace.append((kid, salt))
            self._ensure_ghost_thread()

    def _record_wrd(self, salt: str, wrd: float) -> None:
        self._salt_hist[salt].append(wrd)
        if salt not in self._salt_last_answer:
            return  # Unknown previous answer is not a zero-length answer.
        bin_id = min(self._salt_last_answer[salt] // _LB_BIN_TOKENS, 63)
        samples = self._lb_samples[bin_id]
        samples.append(wrd)
        if len(samples) < 8:
            return
        ordered = sorted(samples)
        q1, q3 = ordered[len(ordered) // 4], ordered[3 * len(ordered) // 4]
        lower_fence = q1 - 1.5 * (q3 - q1)
        self._lb_min[bin_id] = next(x for x in ordered if x >= lower_fence)

    @staticmethod
    def _key_id(key: ObjectKey) -> tuple:
        identity = key.identity
        return (identity.model_name, identity.kv_rank,
                identity.object_group_id, identity.cache_salt,
                identity.chunk_hash)

    def _ensure_ghost_thread(self) -> None:
        if self._ghost_thread is None or not self._ghost_thread.is_alive():
            self._stop.clear()
            self._ghost_thread = threading.Thread(
                target=self._ghost_loop, daemon=True, name="bidaw-ghost"
            )
            self._ghost_thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _ghost_loop(self) -> None:
        while not self._stop.wait(self._ghost_interval):
            try:
                self._rebuild_ghost()
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._ghost_errors += 1
                    self._ghost_last_error = repr(exc)
                logger.warning("Bidaw ghost rebuild failed", exc_info=True)

    def _rebuild_ghost(self) -> None:
        with self._lock:
            if self._seq - self._ghost_built_seq < self._ghost_window // 4:
                return  # not enough new evidence since last rebuild
            snapshot = list(self._trace)
            snapshot_seq = self._seq
        if len(snapshot) < 64:
            return

        capacity = self._capacity_keys
        # next-use index for every position within the window
        next_use = [math.inf] * len(snapshot)
        last_seen: dict[tuple, int] = {}
        for i in range(len(snapshot) - 1, -1, -1):
            kid = snapshot[i][0]
            next_use[i] = last_seen.get(kid, math.inf)
            last_seen[kid] = i

        cache: dict[tuple, int] = {}   # kid -> position pushed
        heap: list[tuple[float, int, tuple]] = []
        reuse = ReuseDistance(max_keys=len(snapshot))
        outcomes: list[tuple[float | None, int]] = []

        for i, (kid, _salt) in enumerate(snapshot):
            wrd_keys = reuse.access(kid)
            hit = kid in cache
            outcomes.append((wrd_keys, 1 if hit else 0))
            if hit:
                cache[kid] = i
                heapq.heappush(heap, (-next_use[i], i, kid))
            else:
                if len(cache) >= capacity:
                    while heap:
                        _neg, pos, k2 = heapq.heappop(heap)
                        if cache.get(k2) == pos:
                            del cache[k2]
                            break
                cache[kid] = i
                heapq.heappush(heap, (-next_use[i], i, kid))

        small: list[int] = []
        buckets: list[list[int]] = [[] for _ in range(self._promising_buckets)]
        span = self._extreme_threshold - capacity
        for wrd, hit in outcomes:
            if wrd is None:
                continue
            if wrd < capacity:
                small.append(hit)
            elif wrd < self._extreme_threshold:
                idx = min(
                    int((wrd - capacity) / max(span, 1.0) * len(buckets)),
                    len(buckets) - 1,
                )
                buckets[idx].append(hit)

        def _mean(xs: list[int], default: float) -> float:
            return sum(xs) / len(xs) if xs else default

        prom = [
            _mean(b, 0.5 * (1.0 - (i + 1) / (len(buckets) + 1)))
            for i, b in enumerate(buckets)
        ]
        with self._lock:
            self._ghost = (_mean(small, 1.0), prom, time.monotonic())
            self._ghost_built_seq = snapshot_seq
            self._ghost_last_error = None
        logger.info(
            "Bidaw ghost rebuilt: window=%d small_hit=%.2f promising=%s",
            len(snapshot),
            self._ghost[0],
            [round(p, 2) for p in self._ghost[1]],
        )

    # ------------------------------------------------------------------
    # Hit potential (paper eq. 2)
    # ------------------------------------------------------------------
    def _bucket_of(self, wrd: float) -> tuple[int, int]:
        """(kind, index): 0=small, 1=promising, 2=extreme."""
        if wrd < self._capacity_keys:
            return 0, 0
        if wrd < self._extreme_threshold:
            span = self._extreme_threshold - self._capacity_keys
            idx = min(
                int(
                    (wrd - self._capacity_keys)
                    / max(span, 1.0)
                    * self._promising_buckets
                ),
                self._promising_buckets - 1,
            )
            return 1, idx
        return 2, 0

    def _bucket_hit(self, kind: int, idx: int) -> float:
        if kind == 0:
            return 1.0
        if kind == 2:
            return 0.0
        if self._ghost is None:
            return 0.5
        _small, prom, _built = self._ghost
        if idx < len(prom):
            return prom[idx]
        return 0.0

    def _potential(self, salt: str, respect_lb: bool) -> float:
        """Expected hit rate of this salt's next access (paper eq. 2)."""
        hist = self._salt_hist.get(salt)
        if not hist:
            return 0.5  # no evidence: neutral; recency breaks ties

        answer_len = self._salt_last_answer.get(salt, 0)
        bin_id = min(answer_len // _LB_BIN_TOKENS, 63)
        lb = self._lb_min.get(bin_id) if respect_lb else None

        small = prom = extreme = 0
        total = 0
        prom_mass = [0.0] * self._promising_buckets
        for wrd in hist:
            if lb is not None and wrd < lb:
                continue  # pruned: next access cannot be this close
            kind, idx = self._bucket_of(wrd)
            if kind == 0:
                small += 1
            elif kind == 1:
                prom += 1
                prom_mass[idx] += 1.0
            else:
                extreme += 1
            total += 1

        if total == 0:
            if lb is None:
                return 0.5
            kind, idx = self._bucket_of(max(lb, 0.0))
            return self._bucket_hit(kind, idx)

        ghost_prom = (
            self._ghost[1]
            if self._ghost is not None
            else [0.5] * self._promising_buckets
        )
        potential = small / total * 1.0 + extreme / total * 0.0
        for i in range(self._promising_buckets):
            potential += prom_mass[i] / total * ghost_prom[i]
        return potential
