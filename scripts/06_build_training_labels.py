"""
Merge human_labels (ground truth) + auto labels (DB) into a single training manifest.

Rules:
- If a file has human_label → use that (human is ground truth, overrides DB)
- Else → use DB auto_label collapsed to binary: valid or fake
- Exclude: human_label=ambiguous, auto_label=valid_bonus, non-images

Output: data/manifest/train_labels.csv
Columns: path, label (0=valid, 1=fake)
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest" / "manifest.csv"
HUMAN = ROOT / "data" / "manifest" / "human_labels.csv"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "manifest" / "train_labels.csv"

# load human labels (ground truth)
human = {}
if HUMAN.exists():
    with HUMAN.open() as f:
        for r in csv.DictReader(f):
            human[r["filename"]] = r["human_label"]

# load manifest + build filename→(status, auto_label)
mani = {}
with MANIFEST.open() as f:
    for r in csv.DictReader(f):
        fn = r["payment_proof"].rsplit("/", 1)[-1]
        mani[fn] = (r["status"], r["label"])

rows = []
stats = {"valid": 0, "fake": 0, "excluded_ambig": 0, "excluded_bonus": 0,
         "excluded_missing_file": 0, "human_override_fake": 0, "human_override_valid": 0}

for status in ("completed", "rejected"):
    for p in (RAW / status).iterdir():
        if not p.is_file():
            continue
        fn = p.name
        if fn in human:
            hl = human[fn]
            if hl == "ambiguous":
                stats["excluded_ambig"] += 1
                continue
            label = 0 if hl == "valid" else 1
            # track overrides vs DB
            _, auto = mani.get(fn, ("", ""))
            if status == "completed" and label == 1:
                stats["human_override_fake"] += 1
            elif status == "rejected" and label == 0:
                stats["human_override_valid"] += 1
        else:
            # fallback to auto
            auto = mani.get(fn, ("", ""))[1]
            if auto == "valid_bonus":
                stats["excluded_bonus"] += 1
                continue
            if auto == "valid":
                label = 0
            elif auto.startswith("fake"):
                label = 1
            else:
                continue
        rows.append((str(p.relative_to(ROOT)), label))
        stats["valid" if label == 0 else "fake"] += 1

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["path", "label"])
    w.writerows(rows)

print(f"✓ Wrote {OUT} ({len(rows)} rows)")
for k, v in stats.items():
    print(f"  {k:<24} {v}")
