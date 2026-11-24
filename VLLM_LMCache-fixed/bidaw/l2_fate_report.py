#!/usr/bin/env python3
"""Classify physical FS L2 writes using explicit replay and cache events."""
import json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LABELS = {
    "A": "All observed later lookups hit DRAM",
    "B1": "No later reuse in the completed observation window",
    "B2": "SSD copy removed before the first DRAM miss",
    "B3": "SSD copy present at the first DRAM miss without an effective readback",
    "C": "Confirmed SSD readback followed by a matching L2 hit",
    "U": "Incomplete evidence or observation window",
}
REPORT_VERSION = 8

@dataclass
class Write:
    size: int
    session: str
    round_index: int | None
    key: tuple
    written_at: int
    shared_cache: bool = False
    pending_reads: dict[str, str | None] = field(default_factory=dict)
    load_events: int = 0
    load_requests: set[str] = field(default_factory=set)
    qualifying_requests: set[str] = field(default_factory=set)
    deleted_at: int | None = None
    ambiguous_removal_at: int | None = None


@dataclass(frozen=True)
class ReplayRound:
    session: str
    number: int
    shared_cache: bool


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

        if marker is None or marker.get("version") not in (3, 4):
            raise ValueError("A version 3 or 4 probe log is required. Record a new run with the updated probe.")

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
    last_round = {}
    open_rounds = {}
    ended = {}
    invalid = set()
    finished_at = {}
    round_at = {}
    for index, event in enumerate(events):
        kind = event["ev"]
        if kind not in {"round_start", "round_finish", "round_session_end", "write", "lookup"}:
            continue
        salt = event["salt"]
        round_id = event.get("round_id")
        if not round_id:
            continue
        if kind == "round_start":
            number = event["round"]
            shared_cache = event.get("shared_cache", False)
            existing = rounds.get(round_id)
            if existing is not None:
                invalid.update((salt, existing.session))
            if salt in ended or salt in open_rounds or (
                salt in last_round and number <= rounds[last_round[salt]].number
            ):
                invalid.add(salt)
            rounds[round_id] = ReplayRound(salt, number, shared_cache)
            last_round[salt] = open_rounds[salt] = round_id
            continue

        scope = rounds.get(round_id)
        if kind in {"write", "lookup"}:
            if scope is None:
                continue
            matching_key = not salt if scope.shared_cache else scope.session == salt
            if matching_key and (kind == "write" or open_rounds.get(scope.session) == round_id):
                round_at[index] = (round_id, scope)
            continue
        if (scope is None or scope.session != salt or scope.number != event["round"]
                or scope.shared_cache != event.get("shared_cache", False)):
            invalid.add(salt)
            continue
        if kind == "round_finish":
            if open_rounds.get(salt) != round_id:
                invalid.add(salt)
            else:
                del open_rounds[salt]
                finished_at[round_id] = index
        elif salt in open_rounds or last_round.get(salt) != round_id or round_id not in finished_at:
            invalid.add(salt)
        else:
            ended[salt] = index
    return round_at, ended, invalid, finished_at


def observed_after(records, written_at, stop):
    start = bisect_right(records, written_at, key=lambda record: record[0])
    end = bisect_left(records, stop, lo=start, key=lambda record: record[0])
    return records[start:end]


def write_category(write, lookups, reads, l1_hits, unconfirmed_hits, events,
                   ended, invalid, finished_at, shared_ends):
    if write.qualifying_requests:
        return "C"
    if write.round_index is None or write.session in invalid:
        return "U"
    if write.shared_cache:
        end = shared_ends[0] if len(shared_ends) == 1 else None
    else:
        end = ended.get(write.session)
    stop = len(events) if end is None else end
    post_lookups = {}
    scoped_requests = set()
    for index, request, context in observed_after(lookups, write.written_at, stop):
        if context is None or context[1].session in invalid:
            return "U"
        round_id, scope = context
        scoped_requests.add(request)
        if scope.session != write.session or scope.number > write.round_index:
            post_lookups.setdefault(request, (index, round_id))

    if any(request not in scoped_requests or unmatched
           for _, request, unmatched in observed_after(reads, write.written_at, stop)):
        return "U"
    if any(request in unconfirmed_hits for request in post_lookups):
        return "U"
    misses = post_lookups.keys() - l1_hits
    if post_lookups and not misses:
        return "A"
    if end is None:
        return "U"
    if not post_lookups:
        return "B1"
    first_miss, round_id = min(post_lookups[request] for request in misses)
    if not first_miss < finished_at.get(round_id, -1) < end:
        return "U"
    miss_time = events[first_miss]["t_ns"]
    if write.deleted_at is not None and write.deleted_at < miss_time:
        return "B2"
    if write.ambiguous_removal_at is not None and write.ambiguous_removal_at < miss_time:
        return "U"
    return "B3"


def classify(events):
    """Classify each physical write, matching SSD reads to the copy that served them."""
    # Submission events may be written after a fast-lane result.
    events = sorted(events, key=lambda event: event["t_ns"])
    round_at, ended, invalid, finished_at = replay_rounds(events)
    classes = dict.fromkeys(LABELS, 0)
    byte_counts = dict.fromkeys(LABELS, 0)
    active = defaultdict(dict)
    writes = []
    seen = set()
    lookups = defaultdict(list)
    reads = defaultdict(list)
    l1_hits = defaultdict(set)
    unconfirmed_hits = defaultdict(set)
    confirmed_sources = {}
    shared_ends = []
    rewrites = loads = deletes = unattributed_loads = padded_tail_writes = 0
    legacy_touch_events = 0

    for index, event in enumerate(events):
        kind = event["ev"]
        if kind == "shared_replay_end":
            shared_ends.append(index)
        if kind not in {"write", "lookup", "l1_hit", "l2_hit", "load_attempt", "load", "delete", "ssd_eviction"}:
            continue
        key = object_key(event)
        if kind == "lookup":
            lookups[key].append((index, event["request"], round_at.get(index)))
            continue
        if kind == "l1_hit":
            l1_hits[key].add(event["request"])
            continue
        if kind == "l2_hit":
            request = event["request"]
            source = confirmed_sources.pop((key, request), None)
            if source is None:
                unconfirmed_hits[key].add(request)
            else:
                source.qualifying_requests.add(request)
                unconfirmed_hits[key].discard(request)
            continue

        adapter = event.get("adapter")
        copies = active[key] if kind == "write" else active.get(key, {})
        if kind in {"delete", "ssd_eviction"}:
            deletes += 1
        if kind == "ssd_eviction" and adapter is None:
            # Earlier bounded FS logs omit the adapter. The benchmark has one.
            if len(copies) != 1:
                for copy in copies.values():
                    if copy.ambiguous_removal_at is None:
                        copy.ambiguous_removal_at = event["t_ns"]
                continue
            adapter = next(iter(copies))
        write = copies.get(adapter)
        if kind == "write":
            padded = "tail_valid_tokens" in event
            padded_tail_writes += padded
            location = (key, adapter)
            rewrites += location in seen
            seen.add(location)
            if write is not None:
                write.deleted_at = event["t_ns"]
                write.pending_reads.clear()
            context = round_at.get(index)
            scope = context[1] if context else None
            write = Write(event["bytes"], scope.session if scope else event["salt"],
                          scope.number if scope else None, key, index,
                          shared_cache=scope.shared_cache if scope else False)
            copies[adapter] = write
            writes.append(write)
        elif kind in {"delete", "ssd_eviction"}:
            if write is not None:
                write.deleted_at = event["t_ns"]
                write.pending_reads.clear()
                del copies[adapter]
                if not copies:
                    del active[key]
        else:
            request = event.get("request")
            # Early version 3 logs have only the prefetch request ID.
            task = event.get("io_task") or request
            if kind == "load_attempt":
                reads[key].append((index, request, False))
                if write is not None and task is not None:
                    write.pending_reads[task] = request
                continue
            if task is None:
                # Older probes logged recency notifications as loads, with neither ID.
                legacy_touch_events += 1
                continue
            loads += 1
            matched = (write is not None and task in write.pending_reads
                       and request == write.pending_reads.pop(task))
            reads[key].append((index, request, not matched))
            if request is not None:
                source = write if matched else None
                request_key = (key, request)
                # A result identifies the request and object, but not the adapter.
                if confirmed_sources.setdefault(request_key, source) is not source:
                    confirmed_sources[request_key] = None
            if not matched:
                unattributed_loads += 1
                continue
            write.load_events += 1
            if request is not None:
                write.load_requests.add(request)

    effective_pairs = 0
    for write in writes:
        key = write.key
        category = write_category(write, lookups[key], reads[key], l1_hits[key],
                                  unconfirmed_hits[key], events, ended, invalid,
                                  finished_at, shared_ends)
        effective_pairs += len(write.qualifying_requests)
        classes[category] += 1
        byte_counts[category] += write.size

    return {
        "classes": classes,
        "bytes": byte_counts,
        "physical_writes": len(writes),
        "padded_tail_writes": padded_tail_writes,
        "writes_without_round_evidence": sum(
            write.round_index is None or write.session in invalid for write in writes
        ),
        "unique": len(seen),
        "rewrite_events": rewrites,
        "load_events": loads,
        "legacy_touch_events": legacy_touch_events,
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
    import argparse

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=LOG_DIR / "l2_events.jsonl",
                        help="Probe event log to classify (default: logs/l2_events.jsonl)")
    parser.add_argument("--out", type=Path, default=LOG_DIR / "ssd_chunk_fate",
                        help="Output prefix; writes <out>.txt, <out>.json, <out>_pie.png")
    args = parser.parse_args()

    marker, events = latest_run(args.log)
    report = {
        "report_version": REPORT_VERSION,
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
        f"Legacy recency notifications excluded from loads: {report['legacy_touch_events']}",
        f"Successful loads attributed to this run's writes: {report['attributed_load_events']}",
        f"Successful loads without a matched write lifecycle: {report['unattributed_load_events']}",
        f"Physical writes read back at least once: {report['writes_read_back']}",
        f"Attributed backend loads per physical write: {intensity}",
        f"Distinct prefetch/write pairs with a readback: {report['readback_prefetch_write_pairs']}",
        f"Distinct prefetch/write pairs meeting C: {report['qualifying_prefetch_write_pairs']}",
        f"Completed replay sessions: {report['completed_replay_sessions']}",
        f"Writes without verified replay round: {report['writes_without_round_evidence']}",
        f"Delete events: {report['delete_events']}",
        "",
    ]
    if marker["version"] < 4:
        lines.append("Probe v3 lacks request-level round IDs. Replay with a v4 probe for A and B1–B3.")
    for key, label in LABELS.items():
        percent = classes[key] / total if total else 0
        lines.append(f"{key} {label}: {classes[key]} writes ({percent:.1%}), "
                     f"{report['bytes'][key] / 2**30:.2f} GiB")

    output = args.out
    output.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    if total:
        ax.pie(
            list(classes.values()),
            labels=[f"{key} {LABELS[key]}\n{classes[key]} writes" for key in LABELS],
            colors=["#6aa9e0", "#8fd18f", "#a76bba", "#d47f74", "#e0b050", "#b9b9b9"],
            startangle=90,
            autopct="%1.1f%%",
            pctdistance=0.72,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
    else:
        ax.text(0.5, 0.5, "No confirmed writes in this run", ha="center", va="center")
        ax.axis("off")
    ax.set_title(f"L2 write fates ({total} confirmed writes)")
    fig.savefig(args.out.parent / f"{args.out.name}_pie.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
