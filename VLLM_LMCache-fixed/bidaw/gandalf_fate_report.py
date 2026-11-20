# SPDX-License-Identifier: Apache-2.0
"""Compare A/B/C/U write outcomes using the same C rule for every arm."""
import argparse
import json
from pathlib import Path

from bidaw.gandalf_report import ARMS
from bidaw.l2_fate_report import LABELS, classify, latest_run

DEFINITIONS = {**LABELS, "C": "Confirmed SSD readback followed by a matching L2 hit"}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path,
                        help="Four-arm benchmark run directory or one L2 event JSONL file")
    parser.add_argument("--out", type=Path, help="Output prefix for the .txt and .json reports")
    args = parser.parse_args()
    source = args.source
    try:
        if source.is_dir():
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
    print(report, end="")


if __name__ == "__main__":
    main()
