# SPDX-License-Identifier: Apache-2.0
"""Summarize request TTFT and asynchronous cache work across benchmark repeats."""
import argparse
import json
import math
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean

from bidaw.gandalf_report import ARMS
from bidaw.l2_fate_report import latest_run, object_key
from bidaw.trace_replay import percentile

logger = logging.getLogger(__name__)

PROMETHEUS = {
    "server_ttft": "vllm:time_to_first_token_seconds",
    "gpu_prefill": "vllm:request_prefill_time_seconds",
    "lookup": "vllm:lmcache_mp_lookup_duration_seconds",
    "vllm_queue": "vllm:request_queue_time_seconds",
}
PATH_METRICS = {
    "client_ttft": "Client TTFT",
    "server_ttft": "Server TTFT",
    "gpu_prefill": "vLLM prefill",
    "lookup": "LMCache lookup",
    "ssd_read": "SSD → DRAM object read",
    "ssd_per_request": "SSD read time per request",
    "prefetch": "LMCache prefetch lookup → hit",
    "retrieve": "DRAM → GPU RETRIEVE",
    "gpu_queue": "GPU scheduling queue",
    "gpu_queue_estimate": "GPU scheduling queue (estimated mean)",
    "vllm_queue": "vLLM queue (raw)",
}
BACKGROUND_METRICS = {
    "store": "HBM → DRAM STORE",
    "reserve": "DRAM reservation",
    "pin": "HBM STORE submit → worker acknowledgement",
    "evict": "DRAM eviction execution",
    "selection": "Eviction candidate selection",
    "ssd_write": "Migration SSD write",
}
QUANTILES = {"p50": .50, "p95": .95, "p99": .99}
SAMPLE_RE = re.compile(r'^([^\s{]+)(?:\{([^}]*)\})?\s+(\S+)')
LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


@dataclass
class Histogram:
    count: int = 0
    total: float = 0.0
    buckets: dict = field(default_factory=dict)

    def merge(self, other):
        if self.buckets and self.buckets.keys() != other.buckets.keys():
            return False
        self.count += other.count
        self.total += other.total
        for bound, count in other.buckets.items():
            self.buckets[bound] = self.buckets.get(bound, 0) + count
        return True

    def quantile(self, fraction):
        target = fraction * self.count
        previous_bound = previous_count = 0
        for bound, count in sorted(self.buckets.items()):
            if not math.isfinite(bound):
                return None
            if count >= target and count > previous_count:
                return previous_bound + (bound - previous_bound) * (
                    (target - previous_count) / (count - previous_count)
                )
            previous_bound, previous_count = bound, count
        return None

    def summary(self):
        if not self.count:
            return None
        return {"n": self.count, "mean": self.total / self.count * 1000,
                **{key: seconds * 1000 if (seconds := self.quantile(p)) is not None else None
                   for key, p in QUANTILES.items()}, "source": "prometheus_histogram"}


def read_histograms(path):
    rows = defaultdict(Histogram)
    if not path.is_file():
        return rows
    metrics = {metric + suffix: (key, suffix) for key, metric in PROMETHEUS.items()
               for suffix in ("_bucket", "_sum", "_count")}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            match = SAMPLE_RE.match(line)
            if match is None or match[1] not in metrics:
                continue
            metric, suffix = metrics[match[1]]
            labels = dict(LABEL_RE.findall(match[2] or ""))
            bound = labels.pop("le", None)
            row = rows[metric, tuple(sorted(labels.items()))]
            value = float(match[3])
            if suffix == "_sum":
                row.total = value
            elif suffix == "_count":
                row.count = int(value)
            elif bound is not None:
                row.buckets[float(bound)] = int(value)
    return rows


def histogram_deltas(folder):
    before_path = folder / "vllm-metrics-before.txt"
    after_path = folder / "vllm-metrics-after.txt"
    if not before_path.is_file() or not after_path.is_file():
        return {}
    before = read_histograms(before_path)
    after = read_histograms(after_path)
    deltas = defaultdict(Histogram)
    for series, current in after.items():
        previous = before.get(series, Histogram())
        if previous.buckets and previous.buckets.keys() != current.buckets.keys():
            continue
        count = current.count - previous.count
        total = current.total - previous.total
        buckets = {bound: n - previous.buckets.get(bound, 0)
                   for bound, n in current.buckets.items()}
        ordered = [buckets[b] for b in sorted(buckets)]
        if (count <= 0 or total < 0 or not math.isfinite(total)
                or any(current < previous for previous, current in zip([0, *ordered], ordered))
                or (ordered and ordered[-1] != count)):
            continue
        deltas[series[0]].merge(Histogram(count, total, buckets))
    return deltas


def stats(values, source="events"):
    if not values:
        return None
    return {"n": len(values), "mean": fmean(values),
            **{key: percentile(values, p) for key, p in QUANTILES.items()}, "source": source}


def duration(start, end):
    if start is None or end is None or start > end:
        return None
    return (end - start) / 1e6


def union_ms(intervals):
    total = 0
    end = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total / 1e6


def add_value(samples, name, value):
    if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
        samples[name].append(value)


def record_ssd_and_prefetch(events, measured, samples, request_count, *, has_io_ids):
    attempts = {}
    lookup_start = {}
    lookup_round = {}
    hit_end = {}
    intervals = defaultdict(list)
    for event in events:
        kind = event["ev"]
        if kind not in {"lookup", "l1_hit", "l2_hit", "load_attempt", "load"}:
            continue
        timestamp = event["t_ns"]
        if kind == "lookup" and measured(event):
            request = event["request"]
            lookup_start.setdefault(request, timestamp)
            if event.get("round_id"):
                lookup_round[request] = event["round_id"]
        elif kind in {"l1_hit", "l2_hit"}:
            request = event["request"]
            if request in lookup_start:
                hit_end[request] = timestamp
        elif kind in {"load_attempt", "load"} and event.get("io_task"):
            key = (event["io_task"], event["adapter"], object_key(event))
            if kind == "load_attempt" and measured(event):
                attempts[key] = event
            elif kind == "load":
                attempt = attempts.pop(key, None)
                if attempt is not None:
                    started = attempt["t_ns"]
                    add_value(samples, "ssd_read", duration(started, timestamp))
                    request = attempt.get("request")
                    if request and started <= timestamp:
                        scope = attempt.get("round_id") or lookup_round.get(request) or request
                        intervals[scope].append((started, timestamp))
    for request, end in hit_end.items():
        add_value(samples, "prefetch", duration(lookup_start[request], end))
    # Older v4 probes still carry exact task IDs. An unpaired legacy load
    # leaves the amortized metric unknown, since it may include actual reads.
    legacy_reads = any(e["ev"] == "load" and not e.get("io_task") for e in events)
    if has_io_ids and not legacy_reads and len(intervals) <= request_count:
        samples["ssd_per_request"].extend(union_ms(spans) for spans in intervals.values())
        samples["ssd_per_request"].extend([0.0] * (request_count - len(intervals)))


def pure_gpu_queue(event):
    total = duration(event["t_arrive"], event.get("t_running"))
    if total is None:
        return None
    for start, end in (("t_lookup_start", "t_lookup_done"), ("t_kv_load_start", "t_kv_load_done")):
        if event.get(start) is not None:
            wait = duration(event[start], event.get(end))
            if wait is None:
                return None
            total -= wait
    return max(0.0, total)


def collect_arm(folder):
    replay = json.loads((folder / "replay.json").read_text())
    records = [record for record in replay["records"]
               if record.get("phase") == "measure" and "ttft_ms" in record and "error" not in record]
    samples = defaultdict(list)
    samples["client_ttft"] = [record["ttft_ms"] for record in records]
    log = folder / "l2_events.jsonl"
    marker, events = {}, []
    if log.is_file():
        try:
            marker, events = latest_run(log)
        except ValueError as exc:
            logger.warning("Skipping unsupported L2 event log %s: %s", log, exc)
    events.sort(key=lambda e: e["t_ns"])
    measured_rounds = {(str(record["user"]), record["round"]) for record in records}
    salt_prefix = replay.get("args", {}).get("salt_prefix", "trace") + "-"
    round_ids = {e["round_id"] for e in events if e["ev"] == "round_start"
                 and (e["salt"].removeprefix(salt_prefix), e["round"]) in measured_rounds}
    has_warmup = any(record.get("phase") == "warmup" for record in replay["records"])

    def measured(event):
        return not has_warmup or event.get("round_id") in round_ids

    if marker:
        record_ssd_and_prefetch(events, measured, samples, len(records),
                                has_io_ids=marker.get("version") == 4)
    counters = CounterForArm()
    probe_enabled = any(e["ev"] == "timing_probe_on" for e in events)
    if probe_enabled:
        counters.allocation_failures = 0
        counters.eviction_cycles = 0
    for event in events:
        kind = event["ev"]
        if kind == "l1_evict_cycle":
            counters.eviction_cycles = (counters.eviction_cycles or 0) + 1
            add_value(samples, "evict", event.get("duration_ms"))
            add_value(samples, "selection", event.get("selection_ms"))
            add_value(samples, "ssd_write", event.get("ssd_write_ms"))
        if not measured(event):
            continue
        if kind == "retrieve_done":
            add_value(samples, "retrieve", event.get("duration_ms"))
        elif kind == "store_done":
            counters.store_samples += 1
            counters.allocation_failures = (counters.allocation_failures or 0) + bool(event["alloc_failed"])
            add_value(samples, "store", event.get("duration_ms"))
            add_value(samples, "reserve", event.get("reserve_ms"))

    timing = folder / "vllm_timing.jsonl"
    if timing.is_file():
        with timing.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                event = json.loads(line)
                if not measured(event):
                    continue
                if event["ev"] == "request_timing":
                    add_value(samples, "gpu_queue", pure_gpu_queue(event))
                elif event["ev"] == "hbm_store_done" and event.get("worker_healthy", True):
                    add_value(samples, "pin", event.get("hbm_store_pin_ms"))
    histograms = histogram_deltas(folder)
    external_hits = replay.get("metrics_delta", {}).get("lmcache_mp_lookup_hit_tokens_total")
    retrieve_available = bool(samples["retrieve"]) or external_hits == 0
    if (not samples["gpu_queue"] and retrieve_available and records
            and all(name in histograms for name in ("vllm_queue", "lookup"))):
        queue_mean = histograms["vllm_queue"].summary()["mean"]
        lookup_amortized = 1000 * histograms["lookup"].total / len(records)
        retrieve_amortized = sum(samples["retrieve"]) / len(records)
        estimate = max(0.0, queue_mean - lookup_amortized - retrieve_amortized)
        counters.queue_estimate = (len(records), estimate)
    return samples, histograms, counters


@dataclass
class CounterForArm:
    eviction_cycles: int | None = None
    allocation_failures: int | None = None
    store_samples: int = 0
    queue_estimate: tuple[int, float] | None = None


def summarize(run_root):
    results = {}
    repeats = sorted(run_root.glob("repeat-*"))
    if not repeats:
        raise ValueError(f"No repeat-* directories in {run_root}")
    for arm in ARMS:
        samples = defaultdict(list)
        histograms = defaultdict(Histogram)
        incompatible = set()
        counters = CounterForArm()
        estimates = []
        for repeat in repeats:
            folder = repeat / arm
            if not (folder / "replay.json").is_file():
                continue
            observed, delta, counts = collect_arm(folder)
            for name, values in observed.items():
                samples[name].extend(values)
            for name, value in delta.items():
                if not histograms[name].merge(value):
                    incompatible.add(name)
            for name in ("allocation_failures", "eviction_cycles"):
                value = getattr(counts, name)
                if value is not None:
                    setattr(counters, name, (getattr(counters, name) or 0) + value)
            counters.store_samples += counts.store_samples
            if counts.queue_estimate is not None:
                estimates.append(counts.queue_estimate)
        metrics = {name: stats(samples[name], "replay" if name == "client_ttft" else "events")
                   for name in PATH_METRICS | BACKGROUND_METRICS}
        for name, histogram in histograms.items():
            if name not in incompatible:
                metrics[name] = histogram.summary()
        if estimates:
            n = sum(n for n, _ in estimates)
            metrics["gpu_queue_estimate"] = {
                "n": n, "mean": sum(n * value for n, value in estimates) / n,
                "p50": None, "p95": None, "p99": None, "source": "queue_minus_lookup_and_retrieve",
            }
        results[arm] = {"metrics_ms": metrics, "eviction_cycles": counters.eviction_cycles,
                        "store_attempts_observed": counters.store_samples if counters.allocation_failures is not None else None,
                        "allocation_failures": counters.allocation_failures,
                        "incompatible_histograms": sorted(incompatible)}
    return results


def display(value):
    return "n/a" if value is None else f"{value:.3f}"


def render(results):
    lines = []
    for title, group, columns in (
        ("TTFT critical path (ms)", PATH_METRICS, ("mean", "p50", "p95", "p99")),
        ("Background writes and eviction (ms)", BACKGROUND_METRICS, ("mean", "p95", "p99")),
    ):
        lines.extend([f"## {title}", "", "| Arm | Metric | Samples | " + " | ".join(columns) + " |",
                      "| --- | --- | ---: | " + " | ".join(["---:"] * len(columns)) + " |"])
        for arm, result in results.items():
            for name, label in group.items():
                row = result["metrics_ms"][name] or {}
                values = " | ".join(display(row.get(col)) for col in columns)
                lines.append(f"| {arm} | {label} | {row.get('n', 'n/a')} | {values} |")
        lines.append("")
    lines.extend(["| Arm | Eviction cycles | STORE attempts observed | Allocation failures |",
                  "| --- | ---: | ---: | ---: |"])
    for arm, result in results.items():
        values = [result[key] if result[key] is not None else "n/a"
                  for key in ("eviction_cycles", "store_attempts_observed", "allocation_failures")]
        lines.append(f"| {arm} | " + " | ".join(map(str, values)) + " |")
    lines.extend(["", "Prometheus percentiles are bucket estimates for the full before/after snapshot window.",
                  "SSD per-request time includes zeros for requests with no reported read.",
                  "Overlapping object reads in one request are counted once in the amortized metric.",
                  "Timings overlap and use different sample populations. The chart and tables do not sum components."])
    return "\n".join(lines) + "\n"


def plot(results, output):
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        plot_with_pillow(results, output)
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 13), layout="constrained")
    colors = ("#326a96", "#42806b", "#b87b30", "#a44f62")
    for ax, group, title in zip(axes, (PATH_METRICS, BACKGROUND_METRICS),
                                ("TTFT and request stages", "Background write and eviction"), strict=True):
        indexes = list(range(len(group)))
        for number, (arm, result) in enumerate(results.items()):
            positions = [i + (number - 1.5) * .18 for i in indexes]
            means = [(result["metrics_ms"][name] or {}).get("mean") for name in group]
            points = [(value, position) for position, value in zip(positions, means, strict=True)
                      if value is not None]
            xs, ys = zip(*points) if points else ([], [])
            ax.scatter(xs, ys, color=colors[number], label=arm)
        ax.set_yticks(indexes, group.values())
        for index, name in enumerate(group):
            if all(result["metrics_ms"][name] is None for result in results.values()):
                ax.text(.02, index, "n/a", transform=ax.get_yaxis_transform(), color="#606b75")
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.set_xlabel("Mean latency (ms)")
        ax.grid(axis="x", alpha=.25)
        ax.legend(ncol=4, loc="lower right")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_with_pillow(results, output):
    """Render the same means in the pinned serving environment."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1300, 1090
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)

    def font(size):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default(size=size)

    text_font, title_font = font(18), font(26)
    colors = ("#326a96", "#42806b", "#b87b30", "#a44f62")
    draw.text((42, 24), "Mean latency by policy (ms)", fill="#182930", font=title_font)
    for i, (arm, color) in enumerate(zip(results, colors, strict=True)):
        x = 44 + 270 * i
        draw.ellipse((x, 75, x + 12, 87), fill=color)
        draw.text((x + 24, 70), arm, fill="#182930", font=text_font)
    for top, group, title in (
        (120, PATH_METRICS, "TTFT and request stages"),
        (716, BACKGROUND_METRICS, "Background write and eviction"),
    ):
        left, right = 498, width - 75
        bottom = top + 46 * len(group) + 54
        means = [[(result["metrics_ms"][metric] or {}).get("mean") for result in results.values()]
                 for metric in group]
        maximum = max((value for row in means for value in row if value is not None), default=1) or 1
        draw.text((42, top), title, fill="#182930", font=title_font)
        for step in range(5):
            x = left + step * (right - left) / 4
            draw.line((x, top + 40, x, bottom), fill="#e1e7eb")
            draw.text((x, bottom + 4), f"{maximum * step / 4:.3g}", anchor="mt",
                      fill="#46545a", font=text_font)
        for index, (label, values) in enumerate(zip(group.values(), means, strict=True)):
            y = top + 59 + index * 46
            draw.text((42, y), label, anchor="lm", fill="#182930", font=text_font)
            if all(value is None for value in values):
                draw.text((left + 12, y), "n/a", anchor="lm", fill="#616b70", font=text_font)
            for i, (value, color) in enumerate(zip(values, colors, strict=True)):
                if value is not None:
                    x = left + value * (right - left) / maximum
                    point_y = y + (i - 1.5) * 8
                    draw.ellipse((x - 4, point_y - 4, x + 4, point_y + 4), fill=color)
    image.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, help="Directory containing repeat-*/<arm>/")
    args = parser.parse_args()
    results = summarize(args.run_root)
    root = args.run_root
    report = render(results)
    (root / "ttft_breakdown.txt").write_text(report, encoding="utf-8")
    (root / "ttft_breakdown.json").write_text(
        json.dumps({"report_version": 1, "arms": results}, indent=2) + "\n", encoding="utf-8")
    plot(results, root / "ttft_breakdown.png")
    print(report, end="")


if __name__ == "__main__":
    main()
