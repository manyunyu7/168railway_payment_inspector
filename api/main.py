"""
Payment Proof validation API.

POST /validate   multipart file "image"
  → { verdict, confidence, stage_a, stage_b: { ocr_text, matches, issues }, boxes[] }

Stage A: MobileNetV3 classifier (valid vs fake)
Stage B: OCR + whitelist/rule matching

Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import io
import re
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "stage_a_best.pt"

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# ─── whitelist for Stage B rule matching ────────────────────────────
RECIPIENT_PATTERNS = [
    r"HENRY\s+AUGUSTA\s+HARSONO",
    r"168\s*RAILWAY",
    r"168Railway",
]
LOCATION_HINTS = [
    r"KOTA\s+BEKASI",
    r"STASIUN\s+BARAT",
    r"KB\.?\s*JERUK",
    r"ANDIR",
    r"BANDUNG",
    r"40181",
    r"17156",
]
SUCCESS_WORDS = [
    r"\b(sukses|berhasil|successful|success|diterima|received|completed)\b",
]
AMOUNT_PATTERN = re.compile(
    r"(?:Rp\.?|IDR)\s*([\d,.]+)",
    re.IGNORECASE,
)
VALID_AMOUNTS = {5000, 8800, 9900, 10000, 12000, 15000, 20000, 25000, 35000, 50000}

# ─── model loader ───────────────────────────────────────────────────
_model = None
_transform = None

def load_model():
    global _model, _transform
    if _model is not None:
        return _model, _transform
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    img_size = ckpt.get("img_size", 320)
    m = models.mobilenet_v3_large(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
    m.load_state_dict(ckpt["state_dict"])
    m = m.to(DEVICE).eval()
    _transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    _model = m
    return _model, _transform


# ─── OCR (optional, PaddleOCR; falls back to pytesseract) ───────────
_ocr = None
def get_ocr():
    global _ocr
    if _ocr is not None:
        return _ocr
    try:
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        return _ocr
    except Exception:
        pass
    try:
        import pytesseract  # noqa
        _ocr = "tesseract"
        return _ocr
    except Exception:
        return None


def run_ocr(img: Image.Image) -> list[dict]:
    """Return list of {text, box}."""
    ocr = get_ocr()
    if ocr is None:
        return []
    if ocr == "tesseract":
        import pytesseract
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        results = []
        for i, t in enumerate(data["text"]):
            t = t.strip()
            if not t:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            results.append({"text": t, "box": [x, y, w, h]})
        return results
    # paddle
    import numpy as np
    arr = np.array(img.convert("RGB"))
    out = ocr.ocr(arr, cls=True)
    results = []
    if out and out[0]:
        for item in out[0]:
            box, (text, conf) = item
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            x, y = int(min(xs)), int(min(ys))
            w, h = int(max(xs) - x), int(max(ys) - y)
            results.append({"text": text, "box": [x, y, w, h], "conf": float(conf)})
    return results


def rule_check(ocr_results: list[dict]) -> dict:
    full_text = "\n".join(r["text"] for r in ocr_results)
    recipient_hit = any(re.search(p, full_text, re.I) for p in RECIPIENT_PATTERNS)
    location_hit = sum(1 for p in LOCATION_HINTS if re.search(p, full_text, re.I))
    success_hit = any(re.search(p, full_text, re.I) for p in SUCCESS_WORDS)
    amounts = []
    for m in AMOUNT_PATTERN.finditer(full_text):
        raw = m.group(1).replace(",", "").replace(".", "")
        try:
            amounts.append(int(raw))
        except ValueError:
            pass
    amount_match = any(a in VALID_AMOUNTS for a in amounts)

    issues = []
    if not recipient_hit:
        issues.append({
            "code": "wrong_recipient",
            "msg": "Nama penerima tidak cocok (bukan ke 168Railway / HENRY AUGUSTA HARSONO)",
        })
    if not success_hit:
        issues.append({
            "code": "no_success_marker",
            "msg": "Tidak ditemukan indikator status transaksi berhasil",
        })
    if amounts and not amount_match:
        issues.append({
            "code": "amount_not_in_plans",
            "msg": f"Nominal {amounts[0]} tidak sesuai paket (5rb/9900/12rb/15rb/25rb/35rb/dst)",
        })
    if not amounts:
        issues.append({
            "code": "no_amount",
            "msg": "Nominal pembayaran tidak terbaca",
        })

    return {
        "recipient_hit": recipient_hit,
        "location_hits": location_hit,
        "success_hit": success_hit,
        "amounts_detected": amounts,
        "amount_match_plan": amount_match,
        "issues": issues,
        "full_text": full_text[:1000],
    }


# ─── API ─────────────────────────────────────────────────────────────
app = FastAPI(title="Payment Proof Validator")


class ValidateResponse(BaseModel):
    verdict: str      # auto_approve | auto_reject | manual_review
    confidence: float
    stage_a: dict
    stage_b: dict


@app.on_event("startup")
def _startup():
    if MODEL_PATH.exists():
        load_model()
        print(f"✓ Model loaded on {DEVICE}")
    else:
        print(f"⚠ Model not found at {MODEL_PATH}")


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": _model is not None, "device": DEVICE}


@app.post("/validate", response_model=ValidateResponse)
async def validate(image: UploadFile = File(...)):
    data = await image.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"invalid image: {e}")

    # Stage A
    model, tf = load_model()
    with torch.no_grad():
        x = tf(img).unsqueeze(0).to(DEVICE)
        probs = torch.softmax(model(x), 1)[0]
        p_valid = float(probs[0])
        p_fake = float(probs[1])
    stage_a = {
        "p_valid": p_valid,
        "p_fake": p_fake,
        "label": "valid" if p_valid >= 0.5 else "fake",
    }

    # Stage B — OCR + rules (only if Stage A says valid-ish)
    stage_b = {"skipped": True}
    if p_valid >= 0.3:  # even weak valid gets OCR'd for detail check
        ocr_results = run_ocr(img)
        rules = rule_check(ocr_results)
        stage_b = {"ocr_boxes": ocr_results[:50], **rules, "skipped": False}

    # Verdict policy
    if p_fake >= 0.95:
        verdict = "auto_reject"
        conf = p_fake
    elif stage_b.get("skipped") is False and stage_b["recipient_hit"] and stage_b["success_hit"] and stage_b["amount_match_plan"] and p_valid >= 0.85:
        verdict = "auto_approve"
        conf = p_valid
    else:
        verdict = "manual_review"
        conf = max(p_valid, p_fake)

    return ValidateResponse(
        verdict=verdict, confidence=round(conf, 3),
        stage_a=stage_a, stage_b=stage_b,
    )
