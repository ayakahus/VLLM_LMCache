# SPDX-License-Identifier: Apache-2.0
"""Bidaw eviction policy (FAST'26) for LMCache MP servers.

Implements the previous-answer-based eviction strategy of
``Bidaw: Enhancing Key-Value Caching for Interactive LLM Serving via
Bidirectional Computation-Storage Awareness`` (Hu et al., FAST'26):

* Per-user (``cache_salt``) weighted-reuse-distance (WRD) tracking, updated
  from logical lookup submissions, independently of L1 lifecycle events.
* Online lower-bound curve fitted to the minimum inlier WRD in each
  previous-answer length bin (paper section 3.3.1).
* A byte-budgeted ghost cache that replays recent accesses with known future
  uses in that window (paper sections 3.3.2 and 3.3.3).
* Eviction candidates with a salt are selected at user granularity by
  predicted hit potential (paper eq. 2). Unsalted keys are individual
  candidates with neutral potential and LRU ordering.

WRD and ghost capacity are measured in bytes.
"""

# Standard
import heapq
import math
import os
import threading
import time
from bisect import bisect_left
from collections import OrderedDict, defaultdict, deque

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

_ANSWER_BIN_TOKENS = 32

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
        self._chunk_bytes = max(1, _env_int("BIDAW_CHUNK_BYTES", 7340032))
        self._capacity_bytes = self._capacity_keys * self._chunk_bytes
        self._ghost_window = max(256, _env_int("BIDAW_GHOST_WINDOW", 8000))
        self._ghost_interval = max(1.0, _env_float("BIDAW_GHOST_INTERVAL", 5.0))
        self._extreme_mult = max(2.0, _env_float("BIDAW_EXTREME_MULT", 6.0))
        self._promising_buckets = max(2, _env_int("BIDAW_PROMISING_BUCKETS", 8))
        self._grace_seconds = max(0.0, _env_float("BIDAW_GRACE_SECONDS", 0.0))
        self._extreme_threshold = self._capacity_bytes * self._extreme_mult

        # --- shared state ---
        self._lock = threading.RLock()
        self._seq = 0
        self._reuse = ReuseDistance(_env_int("BIDAW_HISTORY_KEYS", 200_000))
        self._ghost_errors = 0
        self._ghost_last_error = None
        self._lb_samples = defaultdict(lambda: deque(maxlen=512))
        self._logical_requests = 0
        self._recent_lookups: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._tracked: dict[ObjectKey, float] = {}  # key -> last resident touch
        self._salt_last_touch: dict[str, float] = {}  # salt -> monotonic time
        self._salt_hist: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=64)
        )
        self._salt_last_answer: dict[str, int] = {}
        self._lb_min: dict[int, int] = {}  # answer bin center -> min inlier WRD
        self._lb_curve: tuple[tuple[int, int], ...] = ()
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
            now = time.monotonic()
            for key in keys:
                self._tracked[key] = now

    def on_keys_touched(self, keys: list[ObjectKey]) -> None:
        self.on_keys_created(keys)

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
        if not 0 < expected_ratio <= 1:
            return []
        with self._lock:
            by_salt: dict[str, list[ObjectKey]] = defaultdict(list)
            scored: list[tuple[bool, float, float, list[ObjectKey]]] = []
            for key, last_touch in self._tracked.items():
                salt = key.cache_salt or ""
                if cache_salt is not None and salt != cache_salt:
                    continue
                if key_eligible_filter is not None:
                    try:
                        if not key_eligible_filter(key):
                            continue
                    except Exception:  # noqa: BLE001
                        continue
                if salt:
                    by_salt[salt].append(key)
                else:
                    scored.append((False, 0.5, last_touch, [key]))
            total = len(scored) + sum(len(keys) for keys in by_salt.values())
            if not total:
                return []

            target = max(1, math.ceil(expected_ratio * total))
            now = time.monotonic()

            salt_potentials = {}
            for salt, keys in by_salt.items():
                last_touch = self._salt_last_touch.get(salt, 0.0)
                in_grace = (now - last_touch) < self._grace_seconds
                potential = self._potential(salt, respect_lb=not in_grace)
                scored.append((in_grace, potential, last_touch, keys))
                if len(salt_potentials) < 6:
                    salt_potentials[salt] = round(potential, 3)

            # Prefer out-of-grace candidates, then low potential and old touches.
            scored.sort(key=lambda candidate: candidate[:3])

            selected: list[ObjectKey] = []
            for _grace, _potential, _last_touch, keys in scored:
                # Touch order does not identify prefix leaves. Evicting a
                # partial session could leave an unreachable suffix.
                selected.extend(keys)
                if len(selected) >= target:
                    break
            logger.info(
                "Bidaw eviction: keys=%d salts=%d target=%d selected=%d; "
                "salt potentials: %s",
                total,
                len(by_salt),
                target,
                len(selected),
                salt_potentials,
            )
            return [EvictionAction(destination=EvictionDestination.DISCARD, keys=selected)]

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
                "units": "bytes",
                "logical_requests": self._logical_requests,
                "history_identities": len(self._reuse.last),
                "ghost_errors": self._ghost_errors,
                "ghost_last_error": self._ghost_last_error,
                "grace_seconds": self._grace_seconds,
                "tracked_keys": len(self._tracked),
                "salts": len({key.cache_salt or "" for key in self._tracked}),
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
                "capacity_bytes": self._capacity_bytes,
                "last_answers": dict(list(self._salt_last_answer.items())[-8:]),
                "lb_bin_tokens": _ANSWER_BIN_TOKENS,
                "lb_curve": {
                    str(length): bound for length, bound in self._lb_curve[:16]
                },
                "ghost": ghost,
            }

    # ------------------------------------------------------------------
    # Internal: access recording + WRD / lower-bound accounting
    # ------------------------------------------------------------------
    def on_conversation_access(
        self, salt: str, history_bytes: int, request_id: str | None = None
    ) -> None:
        """Observe one user round, including rounds with no cacheable chunk."""
        with self._lock:
            if request_id is not None:
                identity = salt, request_id
                if identity in self._recent_lookups:
                    return
                self._recent_lookups[identity] = None
                if len(self._recent_lookups) > self._ghost_window:
                    self._recent_lookups.popitem(last=False)
            self._logical_requests += 1
            wrd = self._observe(("conversation", salt), history_bytes)
            if wrd is not None:
                self._record_wrd(salt, wrd)
            self._salt_last_touch[salt] = time.monotonic()
            self._ensure_ghost_thread()

    def on_request_access(self, objects: dict[ObjectKey, int]) -> None:
        """Account for shared-prefix traffic without assigning it a user."""
        if not objects:
            return
        with self._lock:
            self._logical_requests += 1
            for key, size_bytes in objects.items():
                self._observe(self._key_id(key), size_bytes)
            self._ensure_ghost_thread()

    def _observe(self, identity: tuple, size_bytes: int) -> int | None:
        self._seq += 1
        self._trace.append((identity, size_bytes))
        return self._reuse.access(identity, size_bytes)

    def _record_wrd(self, salt: str, wrd: int) -> None:
        self._salt_hist[salt].append(wrd)
        answer_length = self._salt_last_answer.get(salt)
        if answer_length is None:
            return
        bin_start = answer_length // _ANSWER_BIN_TOKENS * _ANSWER_BIN_TOKENS
        center = bin_start + _ANSWER_BIN_TOKENS // 2
        samples = self._lb_samples[center]
        samples.append(wrd)
        if len(samples) < 8:
            return
        ordered = sorted(samples)
        q1, q3 = ordered[len(ordered) // 4], ordered[3 * len(ordered) // 4]
        lower_fence = q1 - 1.5 * (q3 - q1)
        bound = next(x for x in ordered if x >= lower_fence)
        if self._lb_min.get(center) != bound:
            self._lb_min[center] = bound
            self._fit_lb_curve()

    def _fit_lb_curve(self) -> None:
        # Right-to-left minima give the greatest nondecreasing curve whose
        # knots never exceed the observed minimum inlier in their bin.
        curve = []
        for length, bound in sorted(self._lb_min.items(), reverse=True):
            if curve:
                bound = min(bound, curve[-1][1])
            curve.append((length, bound))
        self._lb_curve = tuple(reversed(curve))

    def _answer_lower_bound(self, answer_length: int) -> int:
        curve = self._lb_curve
        if not curve:
            return 0
        index = bisect_left(curve, answer_length, key=lambda knot: knot[0])
        if index == 0:
            return curve[0][1]
        if index == len(curve):
            return curve[-1][1]
        left_length, left_bound = curve[index - 1]
        right_length, right_bound = curve[index]
        return left_bound + (
            (answer_length - left_length) * (right_bound - left_bound)
            // (right_length - left_length)
        )

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
            if self._seq == self._ghost_built_seq or len(self._trace) < 64:
                return
            snapshot = list(self._trace)
            snapshot_seq = self._seq

        capacity = self._capacity_bytes
        # next-use index for every position within the window
        next_use = [math.inf] * len(snapshot)
        last_seen: dict[tuple, int] = {}
        for i in range(len(snapshot) - 1, -1, -1):
            kid = snapshot[i][0]
            next_use[i] = last_seen.get(kid, math.inf)
            last_seen[kid] = i

        cache: dict[tuple, tuple[int, int]] = {}  # identity -> (position, bytes)
        used_bytes = 0
        heap: list[tuple[float, int, tuple]] = []
        reuse = ReuseDistance(max_keys=len(snapshot))
        small: list[int] = []
        buckets: list[list[int]] = [[] for _ in range(self._promising_buckets)]

        for i, (kid, size_bytes) in enumerate(snapshot):
            wrd = reuse.access(kid, size_bytes)
            previous = cache.pop(kid, None)
            if previous is not None:
                used_bytes -= previous[1]
            if wrd is not None and size_bytes:
                kind, idx = self._bucket_of(wrd)
                # The ghost predicts user residency. It budgets each user's
                # latest known size as their conversation grows.
                hit = int(previous is not None and size_bytes <= capacity)
                if kind == 0:
                    small.append(hit)
                elif kind == 1:
                    buckets[idx].append(hit)
            if not 0 < size_bytes <= capacity:
                continue
            while used_bytes + size_bytes > capacity:
                _neg, pos, victim = heapq.heappop(heap)
                resident = cache.get(victim)
                if resident is not None and resident[0] == pos:
                    used_bytes -= cache.pop(victim)[1]
            cache[kid] = i, size_bytes
            used_bytes += size_bytes
            heapq.heappush(heap, (-next_use[i], i, kid))

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
        if wrd < self._capacity_bytes:
            return 0, 0
        if wrd < self._extreme_threshold:
            span = self._extreme_threshold - self._capacity_bytes
            idx = min(
                int(
                    (wrd - self._capacity_bytes)
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
        current = self._reuse.current_distance(
            ("conversation", salt), previously_seen=salt in self._salt_last_touch
        )
        answer_len = self._salt_last_answer.get(salt) if respect_lb else None
        answer_lb = self._answer_lower_bound(answer_len) if answer_len is not None else 0
        first_bucket = self._bucket_of(max(current or 0, answer_lb))
        if first_bucket[0] == 2:
            return 0.0

        hist = self._salt_hist.get(salt)
        if not hist:
            return 0.5 if first_bucket == (0, 0) else self._bucket_hit(*first_bucket)

        total = 0
        hits = 0.0
        for wrd in hist:
            bucket = self._bucket_of(wrd)
            if bucket < first_bucket:
                continue
            hits += self._bucket_hit(*bucket)
            total += 1

        return hits / total if total else self._bucket_hit(*first_bucket)
