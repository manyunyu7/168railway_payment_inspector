"""
Query DB → write CSV manifest of all payment proofs with labels.

Output: data/manifest/manifest.csv
Columns: id, user_id, payment_proof, label, status, rejection_reason,
         amount, unique_code, payment_method_id, source, created_at
"""
import os
import sys
import csv
from pathlib import Path

import pymysql
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OUT = ROOT / "data" / "manifest" / "manifest.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = ("jpg", "jpeg", "png", "webp", "heic", "jfif")

QUERY = """
SELECT
    id, user_id, payment_proof, status, rejection_reason,
    amount, unique_code, payment_method_id, source, created_at,
    created_by_admin, notes
FROM user_subscriptions
WHERE payment_proof IS NOT NULL
  AND payment_proof != ''
  AND status IN ('completed', 'rejected')
ORDER BY id
"""

def label_of(status: str, reason: str | None, amount, created_by_admin: int, notes: str | None) -> str:
    """
    Labels:
      valid           - status=completed, real payment proof expected
      valid_bonus     - status=completed BUT prize winner / admin manual / amount=0.
                        Proof is often WA screenshot, not a receipt — EXCLUDE from training.
      fake_fake       - rejected w/ reason mentioning palsu/scam/fraud
      fake_invalid    - rejected w/ reason mentioning 'tidak valid'/'bukan bukti'
      fake_telegram   - rejected via telegram bot (Henry's reject button; still valid fake)
      fake_other      - rejected other (empty/noisy reason)
    """
    if status == "completed":
        n = (notes or "").lower()
        amt = float(amount or 0)
        bonus_markers = ("bonus", "via wa", "whatsapp", "campaign", "reward", "giveaway",
                         "lomba", "pemenang", "winner", "freebie", "gratis")
        if created_by_admin or amt == 0 or any(k in n for k in bonus_markers):
            return "valid_bonus"
        return "valid"
    r = (reason or "").lower().strip()
    if not r:
        return "fake_other"
    if "telegram" in r:
        return "fake_telegram"  # kept as training negative — bot rejects are still fakes
    if any(k in r for k in ("palsu", "fraud", "scam", "nipu", "koruptor", "maling", "penipuan")):
        return "fake_fake"
    if any(k in r for k in ("tidak valid", "ga valid", "gak valid", "yg valid", "yang valid", "screenshot tidak", "upload bukti", "upload yang", "upload ulang")):
        return "fake_invalid"
    return "fake_other"

def ext_of(path: str) -> str:
    return path.rsplit(".", 1)[-1].lower() if "." in path else ""

def main():
    conn = pymysql.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        user=os.environ["DB_USERNAME"],
        password=os.environ["DB_PASSWORD"],
        database=os.environ["DB_DATABASE"],
        charset="utf8mb4",
    )
    rows_total = 0
    rows_image = 0
    label_counts: dict[str, int] = {}

    with conn.cursor() as cur, OUT.open("w", newline="", encoding="utf-8") as f:
        cur.execute(QUERY)
        writer = csv.writer(f)
        writer.writerow([
            "id", "user_id", "payment_proof", "ext", "label", "status",
            "rejection_reason", "amount", "unique_code",
            "payment_method_id", "source", "created_at",
        ])
        for row in cur:
            (sid, uid, proof, status, reason, amount, uc, pm, src, created,
             cba, notes) = row
            rows_total += 1
            ext = ext_of(proof)
            label = label_of(status, reason, amount, cba or 0, notes)
            label_counts[label] = label_counts.get(label, 0) + 1
            is_image = ext in IMAGE_EXTS
            if is_image:
                rows_image += 1
            writer.writerow([
                sid, uid, proof, ext, label, status,
                (reason or "").replace("\n", " ").replace("\r", " "),
                amount, uc, pm, src, created,
            ])
    conn.close()

    print(f"✓ Wrote {OUT}")
    print(f"  Total rows:   {rows_total}")
    print(f"  Image rows:   {rows_image} ({rows_total - rows_image} non-image)")
    print(f"  Label counts:")
    for k, v in sorted(label_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<16} {v}")

if __name__ == "__main__":
    main()
