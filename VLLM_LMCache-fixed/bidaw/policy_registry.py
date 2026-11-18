# SPDX-License-Identifier: Apache-2.0
"""Thread-safe registry bridging the HTTP API to live Bidaw policy instances.

The LMCache eviction factory creates one policy instance per cache tier
(one for L1, one per L2 adapter).  The HTTP `/bidaw/answer` endpoint needs a
handle to those instances to feed per-user answer lengths, so every created
BidawEvictionPolicy registers itself here.
"""

# Standard
import threading
from typing import Any

_lock = threading.Lock()
_policies: list[Any] = []


def register(policy: Any) -> None:
    with _lock:
        if id(policy) not in {id(p) for p in _policies}:
            _policies.append(policy)


def unregister_all() -> None:
    with _lock:
        _policies.clear()


def feed_answer(cache_salt: str, answer_tokens: int) -> int:
    """Report 'user `cache_salt` just received an `answer_tokens`-long reply'."""
    with _lock:
        targets = list(_policies)
    fed = 0
    for policy in targets:
        try:
            policy.on_answer(cache_salt, answer_tokens)
            fed += 1
        except Exception:  # noqa: BLE001
            pass
    return fed


def snapshot_status() -> dict[str, Any]:
    with _lock:
        targets = list(_policies)
    out = []
    for policy in targets:
        try:
            out.append(policy.status())
        except Exception as exc:  # noqa: BLE001
            out.append({"error": str(exc)})
    return {"policies": out}


def feed_access(keys: list[Any]) -> int:
    """Observe a logical lookup, including objects absent from L1/L2."""
    with _lock:
        targets = list(_policies)
    for policy in targets:
        policy.on_request_access(keys)
    return len(targets)
