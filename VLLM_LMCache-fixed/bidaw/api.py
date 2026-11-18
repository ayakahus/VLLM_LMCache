# SPDX-License-Identifier: Apache-2.0
"""HTTP API injected into the LMCache server: `/bidaw/*`.

`POST /bidaw/answer` — compute-engine side reports that a user
(`cache_salt`) just received an `answer_tokens`-long reply.  This is the
Bidaw storage-awareness input: the eviction manager predicts the weighted
reuse distance of the user's next access from it (paper section 3.3.1).

`GET /bidaw/status` — inspect live policy state (WRD histograms, ghost
cache bucket hit rates, lower-bound curve).
"""

# Standard
from typing import Any, Literal

# Third Party
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# First Party
from bidaw import policy_registry
from bidaw.l2_event_probe import record_round

router = APIRouter(tags=["bidaw"])


class AnswerReport(BaseModel):
    cache_salt: str
    answer_tokens: int = Field(ge=0)


@router.post("/bidaw/answer")
def report_answer(body: AnswerReport) -> dict[str, Any]:
    fed = policy_registry.feed_answer(body.cache_salt, body.answer_tokens)
    return {"status": "ok", "policies_updated": fed}


class FateRound(BaseModel):
    cache_salt: str = Field(min_length=1)
    round_index: int = Field(ge=0)
    phase: Literal["start", "finish", "session_end"]


@router.post("/bidaw/fate/round")
def report_fate_round(body: FateRound) -> dict[str, bool]:
    if not record_round(body.cache_salt, body.round_index, body.phase):
        raise HTTPException(status_code=409, detail="L2 event probe is disabled")
    return {"recorded": True}


@router.get("/bidaw/status")
def bidaw_status() -> dict[str, Any]:
    status = policy_registry.snapshot_status()
    # Scheduler section appears only when the dual-queue patch is enabled.
    try:
        from bidaw.dual_queue_scheduler import scheduler_snapshot

        status["scheduler"] = scheduler_snapshot()
    except Exception:  # noqa: BLE001
        pass
    return status
