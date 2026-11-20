#!/usr/bin/env python3
"""Per-request TTFT/T2T statistics for the four-arm benchmark, with a chart.

T2T (time per output token) follows the standard TPOT definition
``(total_ms - ttft_ms) / (actual_response_tokens - 1)``; percentiles use the
repository's nearest-rank rule (``bidaw.trace_replay.percentile``).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bidaw.trace_replay import percentile

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "logs/gandalf_benchmarks/run.ofRUSFwC"


def default_root():
    """Latest benchmark run under logs/gandalf_benchmarks, else the first run."""
    parent = DEFAULT_ROOT.parent
    runs = list(parent.glob("run.*")) if parent.is_dir() else []
    return max(runs, key=lambda path: path.stat().st_mtime) if runs else DEFAULT_ROOT

ARMS = [("lru", "LRU baseline"), ("lru_writeback", "LRU write-back"),
        ("bidaw", "Bidaw"), ("gandalf", "Gandalf")]
COLORS = {"lru": "#6aa9e0", "lru_writeback": "#8fd18f",
          "bidaw": "#e0b050", "gandalf": "#b9776a"}


def stats(values):
    return {k: percentile(values, p) for k, p in (("p50", .5), ("p95", .95), ("p99", .99))}


def summarize(root):
    rows = {}
    for arm, label in ARMS:
        records = json.loads((root / f"repeat-1/{arm}/replay.json").read_text())["records"]
        recs = [r for r in records if r.get("phase") == "measure" and "ttft_ms" in r]
        mismatch = sum(1 for r in recs
                       if r.get("actual_response_tokens") != r.get("rlen"))
        bad = sum(1 for r in recs if r["ttft_ms"] > r["total_ms"] + 1e-6)
        ttft = [r["ttft_ms"] for r in recs]
        total = [r["total_ms"] for r in recs]
        t2t = [(r["total_ms"] - r["ttft_ms"])
               / max(1, (r.get("actual_response_tokens") or r["rlen"]) - 1)
               for r in recs]
        rows[arm] = dict(
            label=label, n=len(recs), token_length_mismatch=mismatch, ttft_gt_total=bad,
            ttft_mean=sum(ttft) / len(ttft), ttft=stats(ttft),
            t2t_mean=sum(t2t) / len(t2t), t2t=stats(t2t),
            total_mean=sum(total) / len(total), total=stats(total),
        )
    return rows


def render(rows):
    lines = ["四臂延迟指标 (measured phase; T2T = (total-ttft)/(actual_tokens-1))", "",
             "| Arm | n | TTFT mean | TTFT p99 | T2T mean | T2T p99 | E2E mean | E2E p99 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, _ in ARMS:
        r = rows[arm]
        lines.append("| {label} | {n} | {tm:.2f} | {tp:.2f} | {t2m:.2f} | {t2p:.2f} | {em:.2f} | {ep:.2f} |".format(
            label=r["label"], n=r["n"], tm=r["ttft_mean"], tp=r["ttft"]["p99"],
            t2m=r["t2t_mean"], t2p=r["t2t"]["p99"], em=r["total_mean"], ep=r["total"]["p99"]))
    lines += ["", "| Arm | TTFT p50 | TTFT p95 | T2T p50 | T2T p95 | E2E p50 | E2E p95 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, _ in ARMS:
        r = rows[arm]
        lines.append("| {label} | {a:.2f} | {b:.2f} | {c:.2f} | {d:.2f} | {e:.2f} | {f:.2f} |".format(
            label=r["label"], a=r["ttft"]["p50"], b=r["ttft"]["p95"],
            c=r["t2t"]["p50"], d=r["t2t"]["p95"], e=r["total"]["p50"], f=r["total"]["p95"]))
    return "\n".join(lines) + "\n"


def plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    metrics = (("ttft", "TTFT (ms)"), ("t2t", "T2T (ms/token)"), ("total", "E2E (ms)"))
    fig, axes = plt.subplots(1, len(metrics), figsize=(13.8, 4.8))
    width = 0.36
    for ax, (field, title) in zip(axes, metrics):
        values = [rows[arm][f"{field}_mean"] for arm, _ in ARMS]
        tails = [rows[arm][field]["p99"] for arm, _ in ARMS]
        xs = list(range(len(ARMS)))
        bars_mean = [ax.bar(x - width / 2, values[i], width,
                            color=COLORS[ARMS[i][0]])[0] for i, x in enumerate(xs)]
        bars_p99 = [ax.bar(x + width / 2, tails[i], width, color=COLORS[ARMS[i][0]],
                           alpha=0.45, hatch="//", edgecolor="white")[0]
                    for i, x in enumerate(xs)]
        for bars in (bars_mean, bars_p99):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height * 1.01,
                        f"{height:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([label.replace(" ", "\n", 1) for _, label in ARMS], fontsize=8.5)
        ax.set_title(title)
        ax.set_ylabel("latency (ms)")
        ax.set_ylim(0, max(max(values), max(tails)) * 1.18)
        ax.grid(axis="y", color="#e6eaee")
        ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor="#8a8f96", label="mean"),
                        Patch(facecolor="#8a8f96", alpha=0.45, hatch="//",
                              edgecolor="white", label="p99")],
               loc="upper right")
    fig.suptitle("Four-arm latency (TTFT / T2T / E2E): mean vs p99 "
                 f"({rows[ARMS[0][0]]['n']} requests per arm)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", nargs="?", type=Path, default=None,
                        help="Four-arm benchmark run directory (default: the latest run)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output prefix; writes <out>.txt, <out>.json, <out>.png")
    args = parser.parse_args()
    run_root = args.run_root or default_root()
    output = args.out or run_root / "latency_ttft_t2t"
    rows = summarize(run_root)
    report = render(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".txt").write_text(report, encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    plot(rows, output.with_suffix(".png"))
    print(report)
    for arm, _ in ARMS:
        r = rows[arm]
        print("[check] {a}: n={n} actual_vs_rlen_mismatch={m} ttft_gt_total={b}".format(
            a=arm, n=r["n"], m=r["token_length_mismatch"], b=r["ttft_gt_total"]))
    print(f"\nWritten: {output.with_suffix('.txt')}, "
          f"{output.with_suffix('.json')}, {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
