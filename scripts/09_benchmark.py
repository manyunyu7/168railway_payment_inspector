"""
Benchmark hybrid pipeline end-to-end on known samples.

Usage:
    python scripts/09_benchmark.py --n 50 --url http://127.0.0.1:8001
"""
import argparse
import csv
import random
import time
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="samples per class")
    ap.add_argument("--url", type=str, default="http://127.0.0.1:8001")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    labels_csv = ROOT / "data" / "manifest" / "train_labels.csv"
    rows = []
    with labels_csv.open() as f:
        for r in csv.DictReader(f):
            p = ROOT / r["path"]
            if p.exists():
                rows.append((p, int(r["label"])))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    valid = [r for r in rows if r[1] == 0][:args.n]
    fake = [r for r in rows if r[1] == 1][:args.n]
    samples = valid + fake

    cm = {("valid", "auto_approve"): 0, ("valid", "auto_reject"): 0, ("valid", "manual_review"): 0,
          ("fake", "auto_approve"): 0, ("fake", "auto_reject"): 0, ("fake", "manual_review"): 0}
    timings_fast = []
    timings_slow = []

    print(f"Testing {len(samples)} samples ({args.n} valid + {args.n} fake)…")
    for path, label in samples:
        label_name = "valid" if label == 0 else "fake"
        t0 = time.time()
        try:
            with path.open("rb") as fh:
                r = requests.post(f"{args.url}/validate",
                                  files={"image": (path.name, fh, "image/jpeg")},
                                  timeout=30)
        except Exception as e:
            print(f"  ERR {path.name}: {e}")
            continue
        if r.status_code != 200:
            print(f"  HTTP {r.status_code} on {path.name}")
            continue
        d = r.json()
        v = d["verdict"]
        cm[(label_name, v)] += 1
        if d["stage_b"] is None:
            timings_fast.append(d["timings_ms"]["total"])
        else:
            timings_slow.append(d["timings_ms"]["total"])
        mark = "✓" if (label == 0 and v == "auto_approve") or (label == 1 and v == "auto_reject") else (
            "~" if v == "manual_review" else "✗")
        print(f"  {mark} [{label_name:<5}] {v:<15} "
              f"p_v={d['stage_a']['p_valid']:.2f} p_f={d['stage_a']['p_fake']:.2f} "
              f"({d['timings_ms']['total']}ms) — {d['review_reason'][:70]}")

    print("\n" + "=" * 70)
    print(f"HYBRID CONFUSION MATRIX (n={args.n*2}):")
    print(f"{'':<10} {'auto_approve':<15} {'auto_reject':<15} {'manual_review':<15}")
    for true in ("valid", "fake"):
        row = f"true={true:<5} "
        for pred in ("auto_approve", "auto_reject", "manual_review"):
            row += f"{cm[(true, pred)]:<15}"
        print(row)

    # Interpret
    print("\nINTERPRETATION:")
    tp = cm[("valid", "auto_approve")]
    tn = cm[("fake", "auto_reject")]
    fp = cm[("fake", "auto_approve")]  # DANGEROUS: fake lolos
    fn = cm[("valid", "auto_reject")]  # NOT NICE: valid di-reject
    vmr = cm[("valid", "manual_review")]
    fmr = cm[("fake", "manual_review")]
    total = args.n * 2
    print(f"  Auto-handled correctly:  {tp + tn}/{total} = {(tp+tn)/total*100:.1f}%")
    print(f"  Manual review needed:    {vmr + fmr}/{total} = {(vmr+fmr)/total*100:.1f}%")
    print(f"  Dangerous errors (fake auto-approved): {fp} ← harus 0!")
    print(f"  Annoying errors (valid auto-rejected): {fn}")

    if timings_fast:
        print(f"\nLatency fast-path (Stage A only): {sum(timings_fast)//len(timings_fast)}ms avg, "
              f"{len(timings_fast)}/{total} requests")
    if timings_slow:
        print(f"Latency full-path (Stage A+B OCR): {sum(timings_slow)//len(timings_slow)}ms avg, "
              f"{len(timings_slow)}/{total} requests")


if __name__ == "__main__":
    main()
