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
    bidaw_evicted_at: int = -1
    read_attempt: bool = False
    effective: bool = False
    ssd_evicted: bool = False
    closed: bool = False
    lookups: set = field(default_factory=set)
    unknown_lookup: bool = False


def latest_run(path):
    """Yield events from the most recent probe marker in an append-only log."""
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
        for line in stream:
            if line.strip():
                event = json.loads(line)
                if event.get("run") == marker["run"]:
                    yield event


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
    rewrites = loads = deletes = unattributed_loads = 0

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
                lookups.setdefault(request_key, (index, *context))
            for write in copies.values():
                if context is None or write.round_index is None:
                    write.unknown_lookup = True
                elif context[0] > write.round_index:
                    write.lookups.add(request_key)
        elif kind == "l1_evict":
            for write in copies.values():
                write.evicted_at = index
                if event["policy"] == "bidaw_eviction_policy":
                    write.bidaw_evicted_at = index
        elif kind == "write":
            adapter = event["adapter"]
            location = (key, adapter)
            rewrites += location in seen
            seen.add(location)
            previous = copies.get(adapter)
            if previous is not None:
                previous.closed = True
            context = round_at.get(index)
            write = Write(event["bytes"], event["salt"], context[0] if context else None)
            copies[adapter] = write
            writes.append(write)
        elif kind in {"load_attempt", "load"}:
            for write in copies.values():
                write.read_attempt = True
            if kind == "load":
                loads += 1
                write = copies.get(event["adapter"])
                if write is None:
                    unattributed_loads += 1
                    continue
                request = event.get("request")
                request_key = (key, request)
                lookup = lookups.get(request_key)
                if lookup and write.round_index is not None:
                    submitted, number, round_start = lookup
                    if (number > write.round_index
                        and 0 <= write.bidaw_evicted_at < round_start <= submitted
                        < preparing.get(request, -1) < index
                        < l2_hits.get(request_key, -1)):
                        write.effective = True
        elif kind == "delete":
            deletes += 1
            write = copies.pop(event["adapter"], None)
            if write is not None:
                write.closed = True
                write.ssd_evicted = event["eviction"]
                if not copies:
                    del active[key]

    for write in writes:
        session_ended = write.salt in ended
        if write.salt in invalid or write.round_index is None:
            category = "U"
        elif write.effective:
            category = "C"
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
        "unique": len(seen),
        "rewrite_events": rewrites,
        "load_events": loads,
        "unattributed_load_events": unattributed_loads,
        "delete_events": deletes,
    }


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = classify(latest_run(LOG_DIR / "l2_events.jsonl"))
    total = report["physical_writes"]
    classes = report["classes"]
    lines = [
        f"Confirmed physical writes: {total}",
        f"Unique L2 locations written: {report['unique']}",
        f"Physical rewrites: {report['rewrite_events']}",
        f"Backend load events: {report['load_events']}",
        f"Loads with no matching write in this run: {report['unattributed_load_events']}",
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
