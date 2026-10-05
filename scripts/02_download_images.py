"""
Read manifest.csv → download each payment_proof from R2 to data/raw/{completed|rejected}/.

- Skips files that already exist (resumable).
- Downloads only image extensions.
- Writes failures to data/manifest/failures.csv.
- Concurrent with threads.

Usage:
    python scripts/02_download_images.py                # all images
    python scripts/02_download_images.py --limit 500    # smoke test
    python scripts/02_download_images.py --labels valid,fake_fake,fake_invalid
"""
import os
import csv
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MANIFEST = ROOT / "data" / "manifest" / "manifest.csv"
FAILURES = ROOT / "data" / "manifest" / "failures.csv"
RAW_DIR = ROOT / "data" / "raw"

IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "heic", "jfif"}


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
    )


def target_dir(status: str) -> Path:
    return RAW_DIR / ("completed" if status == "completed" else "rejected")


def download_one(s3, bucket: str, key: str, dest: Path) -> tuple[bool, str]:
    if dest.exists() and dest.stat().st_size > 0:
        return True, "skip"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        s3.download_file(bucket, key, str(tmp))
        tmp.rename(dest)
        return True, "ok"
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "?")
        return False, f"client:{code}"
    except Exception as e:
        return False, f"err:{type(e).__name__}:{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max rows (0=all)")
    ap.add_argument("--labels", type=str, default="",
                    help="comma list of labels to include (default: all)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    labels_filter = set(filter(None, args.labels.split(","))) if args.labels else None
    bucket = os.environ["R2_BUCKET"]

    jobs: list[tuple[str, Path]] = []  # (key, dest_path)
    with MANIFEST.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["ext"].lower() not in IMAGE_EXTS:
                continue
            if labels_filter and row["label"] not in labels_filter:
                continue
            key = row["payment_proof"]
            dest = target_dir(row["status"]) / Path(key).name
            jobs.append((key, dest))
            if args.limit and len(jobs) >= args.limit:
                break

    print(f"Queued {len(jobs)} downloads → data/raw/")
    s3 = s3_client()

    ok = skip = fail = 0
    failures: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(download_one, s3, bucket, k, d): (k, d) for k, d in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), unit="img"):
            k, d = futs[fut]
            success, status = fut.result()
            if success and status == "skip":
                skip += 1
            elif success:
                ok += 1
            else:
                fail += 1
                failures.append((k, status))

    print(f"\n✓ Done: {ok} downloaded, {skip} skipped (existed), {fail} failed")

    if failures:
        FAILURES.parent.mkdir(parents=True, exist_ok=True)
        with FAILURES.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["key", "reason"])
            w.writerows(failures)
        print(f"  Failures logged to {FAILURES}")


if __name__ == "__main__":
    main()
