#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-user multi-round interactive workload for Bidaw validation.

Simulates N conversational users against the vLLM+LMCache stack:
* every user has a unique long document prefix (its "history");
* rounds of short questions extend the conversation; the prompt sent at
  round r contains all previous questions and answers (multi-round reuse);
* after each answer the client reports its length to the LMCache server via
  `POST /bidaw/answer` (Bidaw's compute->storage awareness input);
* round gaps scale with the previous answer length (paper section 3.3.1);
* per-round TTFT is collected and a summary JSON is written.
"""

# Standard
import argparse
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Third Party
import requests

MODEL = "qwen2.5-1.5b-instruct"

_TOPIC = (
    "Interactive multi-round assistant serving pipelines must coordinate "
)


def build_doc(user_tag: str, doc_sentences: int) -> str:
    body = (
        f"{_TOPIC}"
        "the KV tensors of every conversation round are cached so that later "
        "rounds avoid recomputing identical prefixes, which reduces "
        "time-to-first-token and increases how many users one GPU can serve. "
    )
    return (f"[user-{user_tag}] " + body * doc_sentences)


def stream_ttft(base: str, prompt: str, salt: str, max_tokens: int):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "cache_salt": salt,
    }
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    with requests.post(
        f"{base}/v1/completions", json=payload, stream=True, timeout=600
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[len("data:"):].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("choices"):
                text = obj["choices"][0].get("text") or ""
                if text and ttft is None:
                    ttft = time.perf_counter() - t0
                chunks.append(text)
    return "".join(chunks).strip(), (ttft or time.perf_counter() - t0)


def report_answer(lmc: str, salt: str, answer_text: str) -> None:
    try:
        requests.post(
            f"{lmc}/bidaw/answer",
            json={"cache_salt": salt, "answer_tokens": len(answer_text.split())},
            timeout=10,
        )
    except Exception:  # noqa: BLE001
        pass


def run_user(user_idx: int, args: dict[str, Any]) -> list[dict]:
    base, lmc = args["vllm"], args["lmc"]
    salt = f"user{user_idx:03d}-{uuid.uuid4().hex[:6]}"
    doc = build_doc(salt, args["doc_sentences"])
    questions = [
        "Summarize the text above in one short sentence.",
        "What is the main benefit described? Answer in a few words.",
        "Name one risk of skipping KV reuse, briefly.",
        "Which component stores the tensors? One word.",
        "How does reuse affect throughput? One short phrase.",
        "Repeat the first sentence of the text.",
        "Give a one-line example use case.",
        "Any accuracy impact from KV reuse? Answer yes/no plus one clause.",
        "Who benefits the most from this caching? Few words.",
        "Which metric improves the most? One word.",
        "Explain the storage tiering in one sentence.",
        "What happens after an engine restart? One short clause.",
    ]
    history = ""
    records = []
    for r in range(args["rounds"]):
        q = questions[r % len(questions)]
        prompt = f"{doc}{history}\nQuestion: {q}\nAnswer:"
        try:
            text, ttft = stream_ttft(base, prompt, salt, args["max_tokens"])
        except Exception as exc:  # noqa: BLE001
            records.append({"user": salt, "round": r, "error": str(exc)})
            continue
        report_answer(lmc, salt, text)
        records.append(
            {
                "user": salt,
                "round": r,
                "ttft_ms": round(ttft * 1000, 1),
                "answer_tokens": len(text.split()),
            }
        )
        history += f"\nQuestion: {q}\nAnswer: {text}"
        # reading/composing gap grows with the answer length (paper 3.3.1)
        time.sleep(min(6.0, 0.15 * max(1, len(text.split()))))
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--doc-sentences", type=int, default=26)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--vllm", default="http://127.0.0.1:8000")
    ap.add_argument("--lmc", default="http://127.0.0.1:8080")
    ap.add_argument("--out", default="")
    args = vars(ap.parse_args())

    print(f"[bidaw-sim] users={args['users']} rounds={args['rounds']} "
          f"parallel={args['parallel']}", flush=True)
    all_records: list[dict] = []
    lock = threading.Lock()

    def _worker(i: int) -> None:
        recs = run_user(i, args)
        with lock:
            all_records.extend(recs)
        print(f"[bidaw-sim] progress: {len(all_records)} rounds done",
              flush=True)

    with ThreadPoolExecutor(max_workers=args["parallel"]) as pool:
        list(pool.map(_worker, range(args["users"])))

    ok = [r for r in all_records if "ttft_ms" in r]
    ok.sort(key=lambda r: r["round"])
    by_round: dict[int, list[float]] = {}
    for r in ok:
        by_round.setdefault(r["round"], []).append(r["ttft_ms"])

    print("\n=== TTFT by round (ms) ===")
    for rnd in sorted(by_round):
        xs = by_round[rnd]
        xs_sorted = sorted(xs)
        p50 = xs_sorted[len(xs_sorted) // 2]
        print(f"round {rnd}: n={len(xs)} p50={p50:.0f} mean="
              f"{sum(xs) / len(xs):.0f} max={max(xs):.0f}")

    summary = {"args": args, "records": all_records}
    out = args["out"] or str(
        Path(__file__).resolve().parents[1] / "logs" / "bidaw_sim.json"
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"[bidaw-sim] summary written to {out}")


if __name__ == "__main__":
    main()