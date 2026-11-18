#!/usr/bin/env python3
"""Classify physical FS L2 writes using explicit replay and cache events."""
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LABELS = {
    "A": "All observed later lookups hit DRAM",
    "B": "Final-round write evicted unread",
    "C": "Bidaw eviction and preparing-queue readback",
    "U": "Unresolved",
}


@dataclass
class Write:
    size: int
    salt: str
    round_index: int | None
    evicted_at: int = -1
    eviction_policy: str | None = None
    read_attempt: bool = False
    pending_reads: dict[str, int] = field(default_factory=dict)
    load_events: int = 0
    load_requests: set[str] = field(default_factory=set)
    qualifying_requests: set[str] = field(default_factory=set)
    ssd_evicted: bool = False
    closed: bool = False
    lookups: set = field(default_factory=set)
    unknown_lookup: bool = False


def latest_run(path):
    """Return the marker and events for the most recent probe run."""
    with path.open(encoding="utf-8") as stream:
        marker = None
        while line := stream.readline():
            if '"probe_on"' in line:
                record = json.loads(line)
                if record.get("ev") == "probe_on":
                    marker = record
                    start = stream.tell()

        if marker is None or marker.get("version") != 3:
            raise ValueError("A version 3 probe log is required. Record a new run with the updated probe.")

        stream.seek(start)
        events = []
        for line in stream:
            if line.strip():
                event = json.loads(line)
                if event.get("run") == marker["run"]:
                    events.append(event)
        return marker, events


def object_key(event):
    return tuple(event[field] for field in ("model", "rank", "group", "salt", "hash"))


def replay_rounds(events):
    """Associate events with replay rounds and validate session-end evidence."""
    rounds = {}
    starts = {}
    open_rounds = set()
    ended = {}
    invalid = set()
    finished = set()
    round_at = {}
    for index, event in enumerate(events):
        kind = event["ev"]
        if kind not in {"round_start", "round_finish", "round_session_end", "write", "lookup"}:
            continue
        salt = event["salt"]
        current_round = rounds.get(salt)
        if kind == "round_start":
            number = event["round"]
            if salt in ended or salt in open_rounds or (
                current_round is not None and number <= current_round
            ):
                invalid.add(salt)
            rounds[salt] = number
            starts[salt] = index
            open_rounds.add(salt)
        elif kind == "round_finish":
            if salt not in open_rounds or current_round != event["round"]:
                invalid.add(salt)
            else:
                open_rounds.remove(salt)
                finished.add((salt, current_round))
        elif kind == "round_session_end":
            if salt in open_rounds or current_round != event["round"] or (
                salt, current_round
            ) not in finished:
                invalid.add(salt)
            else:
                ended[salt] = current_round
        elif current_round is not None and (kind == "write" or salt in open_rounds):
            round_at[index] = (current_round, starts[salt])
    return round_at, ended, invalid


def classify(events):
    # Submission events may be written after a fast-lane result.
    events = sorted(events, key=lambda event: event["t_ns"])
    round_at, ended, invalid = replay_rounds(events)
    l1_hits = set()
    l2_hits = {}
    preparing = {}
    for index, event in enumerate(events):
        kind = event["ev"]
        if kind == "preparing":
            preparing[event["request"]] = index
        elif kind == "l1_hit":
            l1_hits.add((object_key(event), event["request"]))
        elif kind == "l2_hit":
            l2_hits[(object_key(event), event["request"])] = index

    classes = dict.fromkeys(LABELS, 0)
    byte_counts = dict.fromkeys(LABELS, 0)
    active = defaultdict(dict)
    writes = []
    seen = set()
    lookups = {}
    rewrites = loads = deletes = unattributed_loads = padded_tail_writes = 0

    for index, event in enumerate(events):
        kind = event["ev"]
        if kind not in {"write", "lookup", "l1_evict", "load_attempt", "load", "delete"}:
            continue
        key = object_key(event)
        copies = active[key] if kind == "write" else active.get(key, {})
        if kind == "lookup":
            request_key = (key, event["request"])
            context = round_at.get(index)
            if context is not None and copies:
                lookups.setdefault(request_key, (index, context[1]))
            for write in copies.values():
                if context is None or write.round_index is None:
                    write.unknown_lookup = True
                elif context[0] > write.round_index:
                    write.lookups.add(request_key)
        elif kind == "l1_evict":
            for write in copies.values():
                write.evicted_at = index
                write.eviction_policy = event["policy"]
        elif kind == "write":
            adapter = event["adapter"]
            padded_tail_writes += "tail_valid_tokens" in event
            location = (key, adapter)
            rewrites += location in seen
            seen.add(location)
            previous = copies.get(adapter)
            if previous is not None:
                previous.closed = True
                previous.pending_reads.clear()
            context = round_at.get(index)
            write = Write(event["bytes"], event["salt"], context[0] if context else None)
            copies[adapter] = write
            writes.append(write)
        elif kind in {"load_attempt", "load"}:
            for write in copies.values():
                write.read_attempt = True
            write = copies.get(event["adapter"])
            request = event.get("request")
            # Early version 3 logs have only the prefetch request ID.
            task = event.get("io_task") or request
            if kind == "load_attempt":
                if write is not None and task is not None:
                    write.pending_reads[task] = index
                continue
            loads += 1
            attempt = write.pending_reads.pop(task, None) if write is not None else None
            if attempt is None:
                unattributed_loads += 1
                continue
            write.load_events += 1
            if request is not None:
                write.load_requests.add(request)
            request_key = (key, request)
            if request_key in write.lookups and write.eviction_policy == "bidaw_eviction_policy":
                submitted, round_start = lookups[request_key]
                if (0 <= write.evicted_at < round_start <= submitted
                    < preparing.get(request, -1) < attempt < index
                    < l2_hits.get(request_key, -1)):
                    write.qualifying_requests.add(request)
        elif kind == "delete":
            deletes += 1
            write = copies.pop(event["adapter"], None)
            if write is not None:
                write.closed = True
                write.pending_reads.clear()
                write.ssd_evicted = event["eviction"]
                if not copies:
                    del active[key]

    effective_pairs = 0
    for write in writes:
        session_ended = write.salt in ended
        if write.salt in invalid or write.round_index is None:
            category = "U"
        elif write.qualifying_requests:
            category = "C"
            effective_pairs += len(write.qualifying_requests)
        elif (not write.read_attempt and not write.unknown_lookup
              and write.lookups and write.lookups <= l1_hits
              and (write.closed or session_ended)):
            category = "A"
        elif (not write.read_attempt and not write.unknown_lookup
              and write.ssd_evicted and write.evicted_at >= 0
              and session_ended and ended[write.salt] == write.round_index):
            category = "B"
        else:
            category = "U"
        classes[category] += 1
        byte_counts[category] += write.size

    return {
        "classes": classes,
        "bytes": byte_counts,
        "physical_writes": len(writes),
        "padded_tail_writes": padded_tail_writes,
        "unique": len(seen),
        "rewrite_events": rewrites,
        "load_events": loads,
        "unattributed_load_events": unattributed_loads,
        "attributed_load_events": loads - unattributed_loads,
        "attributed_loads_per_write": (loads - unattributed_loads) / len(writes) if writes else None,
        "writes_read_back": sum(write.load_events > 0 for write in writes),
        "readback_prefetch_write_pairs": sum(len(write.load_requests) for write in writes),
        "qualifying_prefetch_write_pairs": effective_pairs,
        "completed_replay_sessions": len(ended.keys() - invalid),
        "delete_events": deletes,
    }


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    marker, events = latest_run(LOG_DIR / "l2_events.jsonl")
    report = {
        "report_version": 2,
        "probe_version": marker["version"],
        "run_id": marker["run"],
        **classify(events),
    }
    total = report["physical_writes"]
    classes = report["classes"]
    per_write = report["attributed_loads_per_write"]
    intensity = f"{per_write:.2f}" if per_write is not None else "n/a"
    lines = [
        f"Probe v{report['probe_version']}; report v{report['report_version']}; run {report['run_id']}",
        f"Confirmed physical writes: {total}",
        f"Padded-tail writes (included above): {report['padded_tail_writes']}",
        f"Unique L2 locations written: {report['unique']}",
        f"Physical rewrites: {report['rewrite_events']}",
        f"Successful backend load events (all): {report['load_events']}",
        f"Successful loads attributed to this run's writes: {report['attributed_load_events']}",
        f"Successful loads without a matched write lifecycle: {report['unattributed_load_events']}",
        f"Physical writes read back at least once: {report['writes_read_back']}",
        f"Attributed backend loads per physical write: {intensity}",
        f"Distinct prefetch/write pairs with a readback: {report['readback_prefetch_write_pairs']}",
        f"Distinct prefetch/write pairs meeting C: {report['qualifying_prefetch_write_pairs']}",
        f"Completed replay sessions: {report['completed_replay_sessions']}",
        f"Delete events: {report['delete_events']}",
        "",
    ]
    for key, label in LABELS.items():
        percent = classes[key] / total if total else 0
        lines.append(f"{key} {label}: {classes[key]} writes ({percent:.1%}), "
                     f"{report['bytes'][key] / 2**30:.2f} GiB")

    output = LOG_DIR / "ssd_chunk_fate"
    output.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    if total:
        ax.pie(
            list(classes.values()),
            labels=[f"{key} {LABELS[key]}\n{classes[key]} writes" for key in LABELS],
            colors=["#6aa9e0", "#8fd18f", "#e0b050", "#b9b9b9"],
            startangle=90,
            autopct="%1.1f%%",
            pctdistance=0.72,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
    else:
        ax.text(0.5, 0.5, "No confirmed writes in this run", ha="center", va="center")
        ax.axis("off")
    ax.set_title(f"Bidaw L2 write fates ({total} confirmed writes)")
    fig.savefig(LOG_DIR / "ssd_chunk_fate_pie.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
