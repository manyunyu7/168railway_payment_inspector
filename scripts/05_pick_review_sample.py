"""
Pick N random downloaded valid images that haven't been labeled yet,
write their paths to a batch file for the model to review.
"""
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "completed"
LABELS = ROOT / "data" / "manifest" / "human_labels.csv"
OUT = ROOT / "data" / "manifest" / "review_batch.txt"

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200

labeled = set()
if LABELS.exists():
    with LABELS.open() as f:
        labeled = {r["filename"] for r in csv.DictReader(f)}

all_files = [p for p in RAW.iterdir() if p.is_file() and p.name not in labeled]
random.Random(42).shuffle(all_files)
pick = all_files[:N]

OUT.write_text("\n".join(str(p) for p in pick))
print(f"Wrote {len(pick)} paths to {OUT}")
