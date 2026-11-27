# SPDX-License-Identifier: Apache-2.0
"""Compare SSD write outcomes using the same rules for every arm."""
import argparse
import json
from pathlib import Path

from bidaw.gandalf_report import benchmark_arms
from bidaw.l2_fate_report import LABELS, classify, latest_run

DEFINITIONS = LABELS
ARM_LABELS = {
    "lru": ("LRU baseline", "FCFS · write-through"),
    "lru_writeback": ("LRU write-back", "FCFS · write on L1 eviction"),
    "bidaw": ("Bidaw", "Dual queue · write-through"),
    "gandalf": ("Gandalf", "FCFS · selective write-back"),
    "l1_only": ("L1-only", "FCFS · DRAM only"),
}
SERIES = {
    "A": ("#477eb2", "All later lookups hit DRAM"),
    "B1": ("#4c8c71", "No later reuse"),
    "B2": ("#8556a0", "SSD eviction before DRAM miss"),
    "B3": ("#aa5656", "Present at miss, no effective readback"),
    "C": ("#d48b2b", "SSD readback + L2 hit"),
    "U": ("#8994a3", "Unresolved"),
}


def summarize(path, *, l1_only=False):
    marker, events = latest_run(path) if path.is_file() or not l1_only else ({}, [])
    counts = classify(events)
    total_bytes = sum(counts["bytes"].values())
    return {
        "log": str(path), "run_id": marker.get("run"), "probe_version": marker.get("version"),
        **counts, "physical_write_bytes": total_bytes,
        "write_share": {
            category: counts["classes"][category] / counts["physical_writes"]
            if counts["physical_writes"] else None
            for category in DEFINITIONS
        },
        "byte_share": {
            category: counts["bytes"][category] / total_bytes if total_bytes else None
            for category in DEFINITIONS
        },
    }


def load_suite(root):
    results = {}
    for repeat in sorted(root.glob("repeat-*")):
        for arm in benchmark_arms(repeat):
            folder = repeat / arm
            stats = summarize(folder / "l2_events.jsonl", l1_only=arm == "l1_only")
            status_path = folder / "status-after.json"
            if status_path.is_file():
                tier = json.loads(status_path.read_text())["tier_policy"]
                if (stats["physical_writes"], stats["physical_write_bytes"]) != (
                    tier.get("writes", 0), tier.get("write_bytes", 0)
                ):
                    raise ValueError(f"{folder}: FS write events do not match the final adapter counters")
            results[f"{repeat.name}/{arm}"] = stats
    if not results:
        raise ValueError(f"No benchmark repeats found in {root}")
    return results


def render(results):
    lines = ["SSD write fates", "", *[f"{key}  {label}" for key, label in DEFINITIONS.items()]]
    headers = " | ".join(DEFINITIONS)
    alignment = " | ".join(["---"] + ["---:"] * (len(DEFINITIONS) + 1))
    for key, share_key, total_key, heading in (
        ("classes", "write_share", "physical_writes", "Writes"),
        ("bytes", "byte_share", "physical_write_bytes", "Write bytes")
    ):
        lines.extend(["", f"| Run | {heading} | {headers} |", f"| {alignment} |"])
        for name, stats in results.items():
            values = []
            for category in DEFINITIONS:
                share = stats[share_key][category]
                percentage = f" ({share:.1%})" if share is not None else ""
                values.append(f"{stats[key][category]:,}{percentage}")
            lines.append(f"| {name} | {stats[total_key]:,} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def pool_arm_counts(results):
    present = {name.rsplit("/", 1)[-1] for name in results}
    pooled = {arm: dict.fromkeys(DEFINITIONS, 0) for arm in ARM_LABELS if arm in present}
    for name, stats in results.items():
        arm = name.rsplit("/", 1)[-1]
        for category, count in stats["classes"].items():
            pooled[arm][category] += count
    return pooled


def plot_suite(results, output):
    from PIL import Image, ImageDraw, ImageFont

    def font(size, bold=False):
        try:
            name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
            return ImageFont.truetype(name, size)
        except OSError:
            return ImageFont.load_default(size=size)

    pooled = pool_arm_counts(results)
    grid_bottom = 178 + 80 * len(pooled)
    legend_y = grid_bottom + 61
    chart = Image.new("RGB", (1500, legend_y + 109), "#ffffff")
    draw = ImageDraw.Draw(chart)
    title, heading, body, small = font(36, True), font(21, True), font(18), font(16)
    ink, muted = "#172635", "#495968"
    x0, x1 = 386, 964
    columns = (*range(1004, 1400, 67), 1442)

    draw.text((42, 32), "SSD write outcomes", font=title, fill=ink)
    repeats = len({name.split("/", 1)[0] for name in results})
    draw.text((44, 86), f"Share of all confirmed SSD writes · {repeats} "
              f"{'repeat' if repeats == 1 else 'repeats'} per configuration",
              font=body, fill=muted)
    draw.text((44, 145), "Experimental configuration", font=heading, fill=ink)
    draw.text((x0, 145), "Distribution by write count", font=heading, fill=ink)
    for x, name in zip(columns, (*SERIES, "Writes"), strict=True):
        draw.text((x, 145), name, font=heading, fill=ink, anchor="mt")
    for percent in (0, 25, 50, 75, 100):
        x = x0 + round((x1 - x0) * percent / 100)
        draw.line((x, 204, x, grid_bottom), fill="#e0e5ea")
        draw.text((x, 175), f"{percent}%", font=small, fill=muted, anchor="mt")

    for row, (arm, counts) in enumerate(pooled.items()):
        y = 226 + row * 80
        name, config = ARM_LABELS[arm]
        draw.text((44, y - 1), name, font=heading, fill=ink)
        draw.text((44, y + 27), config, font=body, fill=muted)
        draw.rectangle((x0, y, x1 - 1, y + 32), fill="#edf0f3")
        total = sum(counts.values())
        cumulative = 0
        left = x0
        for x, (category, (color, _)) in zip(columns[:-1], SERIES.items(), strict=True):
            count = counts[category]
            cumulative += count
            if total:
                right = x0 + round((x1 - x0) * cumulative / total)
                if right > left:
                    draw.rectangle((left, y, right - 1, y + 32), fill=color)
                left = right
            draw.text((x, y + 16), f"{count / total:.1%}" if total else "—",
                      font=body, fill=ink, anchor="mm")
        draw.text((columns[-1], y + 16), f"{total:,}", font=heading, fill=ink, anchor="mm")
        if not total:
            draw.text(((x0 + x1) // 2, y + 16), "No SSD writes", font=small, fill=muted, anchor="mm")

    draw.line((44, legend_y, 1450, legend_y), fill="#e0e5ea")
    for index, (category, (color, label)) in enumerate(SERIES.items()):
        x = 44 + (index % 3) * 480
        y = legend_y + 23 + (index // 3) * 42
        draw.rounded_rectangle((x, y + 3, x + 21, y + 24), radius=3, fill=color)
        draw.text((x + 30, y), f"{category}  {label}", font=body, fill=ink)
    chart.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path,
                        help="Benchmark run directory or one L2 event JSONL file")
    parser.add_argument("--out", type=Path, help="Output prefix for reports and the four-arm chart")
    args = parser.parse_args()
    source = args.source
    suite = source.is_dir()
    try:
        if suite:
            results = load_suite(source)
            output = args.out or source / "ssd_write_fates"
        else:
            results = {source.stem: summarize(source)}
            output = args.out or source.with_name(f"{source.stem}_write_fates")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {"report_version": 2, "definitions": DEFINITIONS, "runs": results}
    Path(f"{output}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = render(results)
    Path(f"{output}.txt").write_text(report, encoding="utf-8")
    if suite:
        plot_suite(results, Path(f"{output}.png"))
    print(report, end="")


if __name__ == "__main__":
    main()
