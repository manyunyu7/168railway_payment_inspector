"""
Payment Inspector API.

POST /validate   multipart file "image"
  → { verdict, confidence, stage_a, stage_b?, reasons[], boxes[] }

Pipeline (Opsi 2 — fast-path):
  1. Stage A classifier (MobileNetV3, ~200ms)
  2. If VERY confident (p >= 0.98) → skip OCR, decide immediately
  3. Else → run Stage B OCR + rule check (~1.5s), then decide

Verdict policy:
  auto_approve   — p_valid >= 0.98  OR  (p_valid >= 0.85 AND all rules match)
  auto_reject    — p_fake  >= 0.98
  manual_review  — everything else (ambiguous → admin verifies)

Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
"""
import io
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "stage_a_best.pt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# ─── config: thresholds ──────────────────────────────────────────────
# Fast-path FAKE: if stage A very confident fake → auto_reject, skip OCR.
# Fast-path VALID is DISABLED — Category 5 fakes (real receipt to wrong merchant)
# fool Stage A visually; we must always OCR-verify before approving.
P_SKIP_OCR_FAKE = 0.98

# Normal-path: stage A confident, need rule confirmation via OCR
P_APPROVE_WITH_RULES = 0.85
P_REJECT_WITH_RULES = 0.85

# ─── Stage B whitelist patterns ──────────────────────────────────────
RECIPIENT_PATTERNS = [
    r"HENRY\s+AUGUSTA\s+HARSONO",
    r"168\s*RAILWAY",
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
    r"\b(sukses|berhasil|successful|success|diterima|received|completed|paid)\b",
]
AMOUNT_PATTERN = re.compile(r"(?:Rp\.?|IDR)\s*([\d.,]+)", re.IGNORECASE)
VALID_AMOUNTS = {5000, 8800, 9900, 10000, 12000, 15000, 20000, 25000, 35000, 50000}
AMOUNT_TOLERANCE = 100  # allow Rp 15.100 to match 15000 (merchant fees)

# ─── model loader ────────────────────────────────────────────────────
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


# ─── OCR (PaddleOCR → fallback pytesseract) ──────────────────────────
_ocr = None
_ocr_kind = None


def get_ocr():
    global _ocr, _ocr_kind
    if _ocr is not None:
        return _ocr, _ocr_kind
    try:
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        _ocr_kind = "paddle"
        return _ocr, _ocr_kind
    except Exception as e:
        print(f"PaddleOCR unavailable: {e}")
    try:
        import pytesseract  # noqa
        _ocr = "tesseract"
        _ocr_kind = "tesseract"
        return _ocr, _ocr_kind
    except Exception as e:
        print(f"Tesseract unavailable: {e}")
    return None, None


def run_ocr(img: Image.Image) -> list[dict]:
    ocr, kind = get_ocr()
    if ocr is None:
        return []
    if kind == "tesseract":
        import pytesseract
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        out = []
        for i, t in enumerate(data["text"]):
            t = t.strip()
            if not t:
                continue
            out.append({
                "text": t,
                "box": [data["left"][i], data["top"][i], data["width"][i], data["height"][i]],
                "conf": float(data["conf"][i]) / 100.0 if data["conf"][i] != "-1" else 0.0,
            })
        return out
    # paddle
    import numpy as np
    arr = np.array(img.convert("RGB"))
    out_raw = ocr.ocr(arr, cls=True)
    results = []
    if out_raw and out_raw[0]:
        for item in out_raw[0]:
            box, (text, conf) = item
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x, y = int(min(xs)), int(min(ys))
            w, h = int(max(xs) - x), int(max(ys) - y)
            results.append({"text": text, "box": [x, y, w, h], "conf": float(conf)})
    return results


def parse_amount(raw: str) -> int | None:
    # normalize "15.000,00" or "15,000" or "15000" → 15000
    s = raw.replace(" ", "")
    # Indonesian format: "." is thousands, "," is decimal
    if "," in s and "." in s:
        s = s.replace(".", "").split(",")[0]
    elif "." in s:
        # could be either; if last group is 3 digits, treat as thousands sep
        parts = s.split(".")
        if all(len(p) == 3 for p in parts[1:]):
            s = s.replace(".", "")
        else:
            s = parts[0]
    elif "," in s:
        parts = s.split(",")
        if all(len(p) == 3 for p in parts[1:]):
            s = s.replace(",", "")
        else:
            s = parts[0]
    try:
        return int(s)
    except ValueError:
        return None


def rule_check(ocr_results: list[dict], expected_amount: int | None = None) -> dict:
    """
    If expected_amount is provided, amount must match THAT specific value
    (not just any plan amount). This closes the loophole where a user pays
    Rp 8.800 for a Rp 15.000 plan.
    """
    full_text = "\n".join(r["text"] for r in ocr_results)

    recipient_hit = any(re.search(p, full_text, re.I) for p in RECIPIENT_PATTERNS)
    location_hit = sum(1 for p in LOCATION_HINTS if re.search(p, full_text, re.I))
    success_hit = any(re.search(p, full_text, re.I) for p in SUCCESS_WORDS)

    amounts = []
    for m in AMOUNT_PATTERN.finditer(full_text):
        a = parse_amount(m.group(1))
        if a is not None and 1000 <= a <= 10_000_000:
            amounts.append(a)

    if expected_amount is not None:
        amount_match = any(abs(a - expected_amount) <= AMOUNT_TOLERANCE for a in amounts)
        amount_check_mode = "expected"
    else:
        amount_match = any(
            any(abs(a - va) <= AMOUNT_TOLERANCE for va in VALID_AMOUNTS)
            for a in amounts
        )
        amount_check_mode = "plan_whitelist"

    issues = []
    if not recipient_hit:
        issues.append({
            "code": "wrong_recipient",
            "msg": "Nama penerima tidak terbaca atau bukan ke 168Railway",
        })
    if not success_hit:
        issues.append({
            "code": "no_success_marker",
            "msg": "Indikator transaksi berhasil tidak ditemukan (sukses/berhasil/successful)",
        })
    if amounts and not amount_match:
        if expected_amount is not None:
            issues.append({
                "code": "amount_mismatch",
                "msg": f"Nominal {amounts[0]} tidak sesuai paket yang dipilih (Rp {expected_amount:,})".replace(",", "."),
            })
        else:
            issues.append({
                "code": "amount_not_in_plans",
                "msg": f"Nominal {amounts[0]} tidak sesuai paket (5rb/9.900/12rb/15rb/25rb/35rb/dst)",
            })
    if not amounts:
        issues.append({
            "code": "no_amount",
            "msg": "Nominal pembayaran tidak terbaca di gambar",
        })

    return {
        "recipient_hit": recipient_hit,
        "location_hits": location_hit,
        "success_hit": success_hit,
        "amounts_detected": amounts,
        "amount_match_plan": amount_match,
        "amount_check_mode": amount_check_mode,
        "expected_amount": expected_amount,
        "issues": issues,
        "rules_all_pass": recipient_hit and success_hit and amount_match,
    }


# ─── API ─────────────────────────────────────────────────────────────
app = FastAPI(title="168Railway Payment Inspector", version="1.0.0")


class ValidateResponse(BaseModel):
    verdict: str              # auto_approve | auto_reject | manual_review
    confidence: float
    review_reason: str        # human-readable explanation
    stage_a: dict
    stage_b: dict | None
    ocr_boxes: list[dict]
    timings_ms: dict


@app.on_event("startup")
def _startup():
    if MODEL_PATH.exists():
        load_model()
        print(f"✓ Model loaded on {DEVICE}")
    else:
        print(f"⚠ Model not found at {MODEL_PATH}")


@app.get("/health")
def health():
    _, kind = get_ocr()
    return {
        "ok": True,
        "model_loaded": _model is not None,
        "device": DEVICE,
        "ocr": kind,
    }


@app.post("/validate", response_model=ValidateResponse)
async def validate(
    image: UploadFile = File(...),
    expected_amount: int | None = Form(default=None, description="Order amount from DB (Rp)"),
):
    t_start = time.time()
    data = await image.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"invalid image: {e}")

    # ─── Stage A ────────────────────────────────────────────────────
    t_a = time.time()
    model, tf = load_model()
    with torch.no_grad():
        x = tf(img).unsqueeze(0).to(DEVICE)
        probs = torch.softmax(model(x), 1)[0]
        p_valid = float(probs[0])
        p_fake = float(probs[1])
    stage_a = {
        "p_valid": round(p_valid, 4),
        "p_fake": round(p_fake, 4),
        "label": "valid" if p_valid >= 0.5 else "fake",
    }
    ms_a = int((time.time() - t_a) * 1000)

    # ─── Fast path: skip OCR only on confident FAKE ─────────────────
    # Valid fast-path removed: cat-5 fakes (real receipt to wrong merchant)
    # score very high on Stage A — we MUST OCR-verify before approving.
    if p_fake >= P_SKIP_OCR_FAKE:
        return ValidateResponse(
            verdict="auto_reject",
            confidence=round(p_fake, 3),
            review_reason="Gambar tidak terdeteksi sebagai bukti bayar yang valid",
            stage_a=stage_a,
            stage_b=None,
            ocr_boxes=[],
            timings_ms={"stage_a": ms_a, "stage_b": 0, "total": int((time.time() - t_start) * 1000)},
        )

    # ─── Stage B: OCR + rule check (always run for valid-looking) ───
    t_b = time.time()
    ocr_results = run_ocr(img)
    rules = rule_check(ocr_results, expected_amount=expected_amount)
    ms_b = int((time.time() - t_b) * 1000)

    stage_b = {
        **{k: v for k, v in rules.items() if k != "ocr_boxes"},
    }

    # ─── Decide ─────────────────────────────────────────────────────
    if p_valid >= P_APPROVE_WITH_RULES and rules["rules_all_pass"]:
        verdict = "auto_approve"
        conf = p_valid
        reason = "Receipt valid dengan semua pengecekan lolos"
    elif p_fake >= P_REJECT_WITH_RULES and len(rules["issues"]) >= 2:
        verdict = "auto_reject"
        conf = p_fake
        reason = "Multiple pengecekan gagal: " + "; ".join(i["msg"] for i in rules["issues"][:2])
    else:
        verdict = "manual_review"
        conf = max(p_valid, p_fake)
        if rules["issues"]:
            reason = "Perlu verifikasi admin: " + rules["issues"][0]["msg"]
        else:
            reason = "Model tidak yakin, perlu verifikasi admin"

    return ValidateResponse(
        verdict=verdict,
        confidence=round(conf, 3),
        review_reason=reason,
        stage_a=stage_a,
        stage_b=stage_b,
        ocr_boxes=ocr_results[:50],
        timings_ms={
            "stage_a": ms_a,
            "stage_b": ms_b,
            "total": int((time.time() - t_start) * 1000),
        },
    )
