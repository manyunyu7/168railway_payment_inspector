# 168Railway Payment Inspector

ML-based payment proof validator untuk 168Railway. Inspector memeriksa bukti bayar
yang di-upload user, memutuskan apakah valid, palsu, atau perlu review manual.

## Arsitektur

**Hybrid 2-stage** jalan di server, dipanggil dari Flutter app:

```
Flutter upload foto
    ↓
[Stage A] MobileNetV3-Large classifier → p(valid) vs p(fake)
    ↓
[Stage B] PaddleOCR + whitelist/rule check
    ↓
Verdict: auto_approve | auto_reject | manual_review
         + bounding boxes untuk highlight di app
```

Policy:
- `p_fake ≥ 0.95` → **auto_reject**
- `p_valid ≥ 0.85` AND recipient match AND success marker AND amount valid → **auto_approve**
- Else → **manual_review** (admin tetap verifikasi)

## Setup

```bash
cp .env.example .env   # isi DB + R2 credentials
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

MySQL tunnel ke prod aktif di `127.0.0.1:33306` buat Step 1-2.

## Pipeline

```bash
# 1. Dump manifest dari DB → data/manifest/manifest.csv
python scripts/01_dump_manifest.py

# 2. Download image dari R2
python scripts/02_download_images.py --labels fake_fake,fake_invalid,fake_other,fake_telegram
python scripts/05_pick_review_sample.py   # sample random positive

# 3. (opsional) Ekstrak metadata (EXIF, dim, phash dedup)
python scripts/03_extract_metadata.py

# 4. Review manual via labeling UI
python scripts/04_labeling_server.py
# → http://localhost:5055  (A=valid, S=fake, D=ambiguous)

# 5. Merge human + DB labels → training set
python scripts/06_build_training_labels.py

# 6. Train Stage A classifier (MPS/CUDA auto-detect)
python scripts/07_train.py

# 7. Serve API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Dataset

Dari DB (auto label dari `status` + `rejection_reason`):

| Label | Count | Role |
|---|---|---|
| `valid` | 36.083 | Positive (sample ~2000 untuk training) |
| `valid_bonus` | 30 | Excluded (admin manual / pemenang lomba, isi WA screenshot) |
| `fake_fake` | 116 | Negative — reason explicit "palsu / fraud" |
| `fake_invalid` | 206 | Negative — "upload yg valid" |
| `fake_other` | 229 | Negative — no reason |
| `fake_telegram` | 272 | Negative — reject via Telegram bot |

Label quality (dari 330 human review):
- Positive: 97.2% clean (noise = user transfer ke orang lain / nominal kurang, tapi admin tetap approve manual)
- Negative: 100% clean

Final training set: **2862 labels** (2130 valid, 732 fake) stratified 85/15 train/val.

## API

```bash
curl -X POST http://localhost:8000/validate -F "image=@receipt.jpg"
```

Response:
```json
{
  "verdict": "auto_approve",
  "confidence": 0.97,
  "stage_a": { "p_valid": 0.97, "p_fake": 0.03, "label": "valid" },
  "stage_b": {
    "recipient_hit": true,
    "success_hit": true,
    "amounts_detected": [15000],
    "amount_match_plan": true,
    "issues": [],
    "ocr_boxes": [{ "text": "Rp 15.000", "box": [x,y,w,h] }, ...]
  }
}
```

## Flutter integration

```dart
final res = await dio.post('/validate',
  data: FormData.fromMap({'image': await MultipartFile.fromFile(path)}));
switch (res.data['verdict']) {
  case 'auto_reject':
    showReasons(res.data['stage_b']['issues']);
    showHighlights(res.data['stage_b']['ocr_boxes']);
    break;
  case 'auto_approve':
    showSuccess();
    break;
  case 'manual_review':
    showPending();
}
```
