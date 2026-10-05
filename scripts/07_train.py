"""
Train Stage A classifier: binary valid vs fake payment proof.

Model: MobileNetV3-Large, fine-tuned from ImageNet weights.
Framework: PyTorch, MPS backend for Mac M-series.
Output: models/stage_a_best.pt + metrics plot
"""
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data" / "manifest" / "train_labels.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

# ─── config ──────────────────────────────────────────────────────────
IMG_SIZE = 320
BATCH = 32
EPOCHS = 15
LR = 1e-4
VAL_SPLIT = 0.15
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ─── dataset ─────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.6, 1.4)),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 1.5))], p=0.2),
    transforms.RandomApply([transforms.RandomRotation(8)], p=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ProofDS(Dataset):
    def __init__(self, items, tf):
        self.items = items
        self.tf = tf

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        try:
            img = Image.open(ROOT / path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        return self.tf(img), label


def load_items():
    items = []
    with LABELS.open() as f:
        for r in csv.DictReader(f):
            p = ROOT / r["path"]
            if p.exists():
                items.append((r["path"], int(r["label"])))
    return items


def main():
    items = load_items()
    print(f"Loaded {len(items)} items")
    y = [l for _, l in items]
    print(f"  valid (0): {y.count(0)}")
    print(f"  fake  (1): {y.count(1)}")

    train_items, val_items = train_test_split(
        items, test_size=VAL_SPLIT, random_state=SEED,
        stratify=[l for _, l in items]
    )
    print(f"Train: {len(train_items)}  Val: {len(val_items)}")

    train_ds = ProofDS(train_items, train_tf)
    val_ds = ProofDS(val_items, val_tf)

    # weighted sampler to balance classes per batch
    y_train = [l for _, l in train_items]
    class_count = [y_train.count(0), y_train.count(1)]
    weights = [1.0 / class_count[l] for l in y_train]
    sampler = WeightedRandomSampler(weights, num_samples=len(y_train), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=4, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=False)

    # ─── model ──────────────────────────────────────────────────────
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    in_feat = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feat, 2)
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        # train
        model.train()
        tl = tc = tn = 0
        for x, y_ in train_dl:
            x, y_ = x.to(DEVICE), y_.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y_)
            loss.backward()
            optimizer.step()
            tl += loss.item() * x.size(0)
            tc += (logits.argmax(1) == y_).sum().item()
            tn += x.size(0)
        tr_loss = tl / tn
        tr_acc = tc / tn

        # val
        model.eval()
        vl = vc = vn = 0
        all_y, all_p, all_pr = [], [], []
        with torch.no_grad():
            for x, y_ in val_dl:
                x, y_ = x.to(DEVICE), y_.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y_)
                probs = torch.softmax(logits, 1)[:, 1]
                vl += loss.item() * x.size(0)
                pred = logits.argmax(1)
                vc += (pred == y_).sum().item()
                vn += x.size(0)
                all_y.extend(y_.cpu().tolist())
                all_p.extend(pred.cpu().tolist())
                all_pr.extend(probs.cpu().tolist())
        va_loss = vl / vn
        va_acc = vc / vn

        rep = classification_report(all_y, all_p, output_dict=True, zero_division=0)
        f1 = rep["weighted avg"]["f1-score"]
        f1_fake = rep.get("1", {}).get("f1-score", 0)
        recall_fake = rep.get("1", {}).get("recall", 0)
        precision_fake = rep.get("1", {}).get("precision", 0)
        try:
            auc = roc_auc_score(all_y, all_pr)
        except ValueError:
            auc = 0.0

        scheduler.step()
        dt = time.time() - t0

        print(f"ep {epoch:02d}  loss {tr_loss:.3f}/{va_loss:.3f}  "
              f"acc {tr_acc:.3f}/{va_acc:.3f}  "
              f"F1(fake) {f1_fake:.3f}  P/R {precision_fake:.3f}/{recall_fake:.3f}  "
              f"AUC {auc:.3f}  ({dt:.0f}s)")

        history.append({
            "epoch": epoch, "tr_loss": tr_loss, "va_loss": va_loss,
            "tr_acc": tr_acc, "va_acc": va_acc, "f1_fake": f1_fake,
            "precision_fake": precision_fake, "recall_fake": recall_fake, "auc": auc,
        })

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": rep,
                "classes": ["valid", "fake"],
                "img_size": IMG_SIZE,
            }, MODELS / "stage_a_best.pt")
            print(f"  ✓ saved best (F1 weighted {f1:.3f})")

    # final confusion matrix on val
    print("\nFinal val confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(all_y, all_p)
    print(f"             pred_valid  pred_fake")
    print(f"true_valid   {cm[0,0]:>10} {cm[0,1]:>10}")
    print(f"true_fake    {cm[1,0]:>10} {cm[1,1]:>10}")

    (MODELS / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"\n✓ Done. Best model at {MODELS / 'stage_a_best.pt'}")


if __name__ == "__main__":
    main()
