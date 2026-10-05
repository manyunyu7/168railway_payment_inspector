"""
Scan downloaded images → extract metadata (dims, filesize, EXIF, phash dedup).

Output: data/manifest/metadata.csv
Columns: filename, status, path, bytes, width, height, aspect_ratio,
         exif_make, exif_model, exif_software, is_screenshot_guess, phash

- EXIF: Make/Model/Software give signal (screenshot vs photo vs edited)
- aspect_ratio helps distinguish screenshots (tall) from photos (varied)
- phash: perceptual hash to find duplicates / near-dupes across users
"""
import os
import csv
import hashlib
from pathlib import Path

from PIL import Image, ExifTags
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "manifest" / "metadata.csv"

EXIF_TAG_BY_NAME = {v: k for k, v in ExifTags.TAGS.items()}
MAKE_TAG = EXIF_TAG_BY_NAME.get("Make")
MODEL_TAG = EXIF_TAG_BY_NAME.get("Model")
SOFTWARE_TAG = EXIF_TAG_BY_NAME.get("Software")


def phash(img: Image.Image, size: int = 8) -> str:
    """Simple average-hash (fast, good enough for near-dup detection)."""
    g = img.convert("L").resize((size, size), Image.LANCZOS)
    px = list(g.getdata())
    avg = sum(px) / len(px)
    bits = "".join("1" if p >= avg else "0" for p in px)
    return f"{int(bits, 2):0{size * size // 4}x}"


def is_screenshot_guess(w: int, h: int, make: str, software: str) -> bool:
    # heuristic: common phone screen aspect ratios + no camera make
    aspect = h / max(w, 1)
    phone_ratios = [2340/1080, 2400/1080, 2532/1170, 2556/1179, 2778/1284, 2436/1125]
    close = any(abs(aspect - r) < 0.02 for r in phone_ratios)
    close_rev = any(abs((1/aspect) - r) < 0.02 for r in phone_ratios)
    if close or close_rev:
        return True
    return False


def extract_one(p: Path, status: str) -> list | None:
    try:
        st = p.stat()
        with Image.open(p) as img:
            w, h = img.size
            exif_make = exif_model = exif_software = ""
            try:
                exif = img.getexif()
                if exif:
                    exif_make = str(exif.get(MAKE_TAG, "") or "").strip()
                    exif_model = str(exif.get(MODEL_TAG, "") or "").strip()
                    exif_software = str(exif.get(SOFTWARE_TAG, "") or "").strip()
            except Exception:
                pass
            ph = phash(img)
            aspect = round(h / max(w, 1), 3)
            ss = is_screenshot_guess(w, h, exif_make, exif_software)
        return [
            p.name, status, str(p.relative_to(ROOT)),
            st.st_size, w, h, aspect,
            exif_make, exif_model, exif_software,
            int(ss), ph,
        ]
    except Exception as e:
        return [p.name, status, str(p.relative_to(ROOT)), 0, 0, 0, 0, "", "", "", 0, f"ERR:{e}"]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    files: list[tuple[Path, str]] = []
    for sub in ("completed", "rejected"):
        d = RAW / sub
        if d.exists():
            for p in d.iterdir():
                if p.is_file():
                    files.append((p, sub))

    print(f"Scanning {len(files)} images…")
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "filename", "status", "path", "bytes", "width", "height",
            "aspect_ratio", "exif_make", "exif_model", "exif_software",
            "is_screenshot_guess", "phash",
        ])
        for p, status in tqdm(files, unit="img"):
            row = extract_one(p, status)
            if row:
                w.writerow(row)

    print(f"✓ Wrote {OUT}")


if __name__ == "__main__":
    main()
