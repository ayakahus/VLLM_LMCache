# SPDX-License-Identifier: Apache-2.0
"""Compare A/B/C/U write outcomes using the same C rule for every arm."""
import argparse
import json
from pathlib import Path

from bidaw.gandalf_report import ARMS
from bidaw.l2_fate_report import LABELS, classify, latest_run

DEFINITIONS = {**LABELS, "C": "Confirmed SSD readback followed by a matching L2 hit"}
ARM_LABELS = {
    "lru": ("LRU baseline", "FCFS · write-through"),
    "lru_writeback": ("LRU write-back", "FCFS · write on L1 eviction"),
    "bidaw": ("Bidaw", "Dual queue · write-through"),
    "gandalf": ("Gandalf", "FCFS · selective write-back"),
}
SERIES = {
    "A": ("#477eb2", "All later lookups hit DRAM"),
    "B": ("#4c8c71", "No later reuse request"),
    "C": ("#d48b2b", "SSD readback + L2 hit"),
    "U": ("#8994a3", "Unresolved"),
}


def summarize(path):
    marker, events = latest_run(path)
    counts = classify(events, confirmed_readback=True)
    total_bytes = sum(counts["bytes"].values())
    return {
        "log": str(path), "run_id": marker["run"], "probe_version": marker["version"],
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
        for arm in ARMS:
            folder = repeat / arm
            stats = summarize(folder / "l2_events.jsonl")
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
    for key, share_key, total_key, heading in (
        ("classes", "write_share", "physical_writes", "Writes"),
        ("bytes", "byte_share", "physical_write_bytes", "Write bytes")
    ):
        lines.extend(["", f"| Run | {heading} | A | B | C | U |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for name, stats in results.items():
            values = []
            for category in DEFINITIONS:
                share = stats[share_key][category]
                percentage = f" ({share:.1%})" if share is not None else ""
                values.append(f"{stats[key][category]:,}{percentage}")
            lines.append(f"| {name} | {stats[total_key]:,} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def pool_arm_counts(results):
    pooled = {arm: dict.fromkeys(DEFINITIONS, 0) for arm in ARMS}
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

    chart = Image.new("RGB", (1500, 630), "#ffffff")
    draw = ImageDraw.Draw(chart)
    title, heading, body, small = font(36, True), font(21, True), font(18), font(16)
    ink, muted = "#172635", "#495968"
    x0, x1 = 386, 964
    columns = (1018, 1110, 1202, 1294, 1413)

    draw.text((42, 32), "SSD write outcomes", font=title, fill=ink)
    repeats = len(results) // len(ARMS)
    draw.text((44, 86), f"Share of all confirmed SSD writes · {repeats} "
              f"{'repeat' if repeats == 1 else 'repeats'} per configuration",
              font=body, fill=muted)
    draw.text((44, 145), "Experimental configuration", font=heading, fill=ink)
    draw.text((x0, 145), "Distribution by write count", font=heading, fill=ink)
    for x, name in zip(columns, (*SERIES, "Writes"), strict=True):
        draw.text((x, 145), name, font=heading, fill=ink, anchor="mt")
    for percent in (0, 25, 50, 75, 100):
        x = x0 + round((x1 - x0) * percent / 100)
        draw.line((x, 204, x, 498), fill="#e0e5ea")
        draw.text((x, 175), f"{percent}%", font=small, fill=muted, anchor="mt")

    for row, (arm, counts) in enumerate(pool_arm_counts(results).items()):
        y = 226 + row * 80
        name, config = ARM_LABELS[arm]
        draw.text((44, y - 1), name, font=heading, fill=ink)
        draw.text((44, y + 27), config, font=body, fill=muted)
        draw.rectangle((x0, y, x1 - 1, y + 32), fill="#edf0f3")
        total = sum(counts.values())
        cumulative = 0
        left = x0
        for x, (category, (color, _)) in zip(columns[:4], SERIES.items(), strict=True):
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

    draw.line((44, 559, 1450, 559), fill="#e0e5ea")
    for x, (category, (color, label)) in zip((44, 416, 784, 1152), SERIES.items(), strict=True):
        draw.rounded_rectangle((x, 585, x + 21, 606), radius=3, fill=color)
        draw.text((x + 30, 582), f"{category}  {label}", font=body, fill=ink)
    chart.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path,
                        help="Four-arm benchmark run directory or one L2 event JSONL file")
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
    summary = {"report_version": 1, "definitions": DEFINITIONS, "runs": results}
    Path(f"{output}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = render(results)
    Path(f"{output}.txt").write_text(report, encoding="utf-8")
    if suite:
        plot_suite(results, Path(f"{output}.png"))
    print(report, end="")


if __name__ == "__main__":
    main()
