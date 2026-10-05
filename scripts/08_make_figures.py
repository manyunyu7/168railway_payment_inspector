"""
Generate figures for README:
- training_curves.png    (loss + accuracy + F1 over epochs)
- confusion_matrix.png   (final val confusion matrix)
- fake_grid.png          (sample grid of fake payment uploads)

Run after training completes:
    python scripts/08_make_figures.py
"""
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "models" / "training_history.json"
LABELS = ROOT / "data" / "manifest" / "train_labels.csv"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#374151",
    "axes.labelcolor": "#374151",
    "xtick.color": "#6b7280",
    "ytick.color": "#6b7280",
    "axes.titleweight": "bold",
    "axes.titlecolor": "#111827",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# ─── 1. Training curves ─────────────────────────────────────────────
def training_curves():
    if not HIST.exists():
        print("skip training_curves: no history yet")
        return
    hist = json.loads(HIST.read_text())
    if not hist:
        print("skip training_curves: empty history")
        return
    ep = [h["epoch"] for h in hist]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Loss
    axes[0].plot(ep, [h["tr_loss"] for h in hist], "-o", color="#3b82f6", label="train", lw=2, markersize=5)
    axes[0].plot(ep, [h["va_loss"] for h in hist], "-o", color="#ef4444", label="val", lw=2, markersize=5)
    axes[0].set_title("Loss", pad=10)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("CE loss")
    axes[0].legend(frameon=False); axes[0].grid(alpha=0.2)

    # Accuracy
    axes[1].plot(ep, [h["tr_acc"] for h in hist], "-o", color="#3b82f6", label="train", lw=2, markersize=5)
    axes[1].plot(ep, [h["va_acc"] for h in hist], "-o", color="#ef4444", label="val", lw=2, markersize=5)
    axes[1].set_title("Accuracy", pad=10)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].legend(frameon=False); axes[1].grid(alpha=0.2)
    axes[1].set_ylim(0.85, 1.0)

    # F1 + AUC
    axes[2].plot(ep, [h["f1_fake"] for h in hist], "-o", color="#059669", label="F1 (fake)", lw=2, markersize=5)
    axes[2].plot(ep, [h["auc"] for h in hist], "-o", color="#7c3aed", label="AUC", lw=2, markersize=5)
    axes[2].set_title("F1 (fake class) & AUC", pad=10)
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("score")
    axes[2].legend(frameon=False); axes[2].grid(alpha=0.2)
    axes[2].set_ylim(0.7, 1.0)

    plt.tight_layout()
    out = OUT / "training_curves.png"
    plt.savefig(out, facecolor="white")
    plt.close()
    print(f"✓ {out}")


# ─── 2. Confusion matrix ────────────────────────────────────────────
def confusion_matrix():
    if not HIST.exists():
        return
    hist = json.loads(HIST.read_text())
    if not hist:
        return
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from torch.utils.data import DataLoader
    from sklearn.metrics import confusion_matrix as cm_fn
    from sklearn.model_selection import train_test_split

    ckpt_path = ROOT / "models" / "stage_a_best.pt"
    if not ckpt_path.exists():
        print("skip confusion_matrix: no checkpoint")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    img_size = ckpt.get("img_size", 320)
    m = models.mobilenet_v3_large(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()

    tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    items = []
    with LABELS.open() as f:
        for r in csv.DictReader(f):
            p = ROOT / r["path"]
            if p.exists():
                items.append((r["path"], int(r["label"])))
    _, val_items = train_test_split(items, test_size=0.15, random_state=42,
                                     stratify=[l for _, l in items])

    y_true = []; y_pred = []
    for path, y in val_items:
        try:
            img = Image.open(ROOT / path).convert("RGB")
        except Exception:
            continue
        with torch.no_grad():
            x = tf(img).unsqueeze(0)
            logits = m(x)
            pred = logits.argmax(1).item()
        y_true.append(y); y_pred.append(pred)

    cm = cm_fn(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            total = cm[i].sum()
            pct = v / total * 100 if total else 0
            color = "white" if v > cm.max() * 0.5 else "#1f2937"
            ax.text(j, i, f"{v}\n{pct:.1f}%", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["valid", "fake"]); ax.set_yticklabels(["valid", "fake"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion Matrix (val, n={len(y_true)})", pad=12)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    out = OUT / "confusion_matrix.png"
    plt.savefig(out, facecolor="white")
    plt.close()
    print(f"✓ {out}")


# ─── 3. Fake grid ───────────────────────────────────────────────────
def fake_grid():
    """Grid of representative fake uploads with blurred faces where applicable."""
    items = []
    with LABELS.open() as f:
        for r in csv.DictReader(f):
            if int(r["label"]) == 1:
                p = ROOT / r["path"]
                if p.exists():
                    items.append(p)
    if len(items) < 12:
        print(f"skip fake_grid: only {len(items)} fake images")
        return
    random.Random(7).shuffle(items)
    pick = items[:12]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for ax, p in zip(axes.flat, pick):
        try:
            img = Image.open(p).convert("RGB")
            img.thumbnail((400, 400))
            ax.imshow(img)
        except Exception:
            pass
        ax.axis("off")
    fig.suptitle("Representative FAKE Uploads (from training set)",
                 fontsize=14, fontweight="bold", color="#111827", y=0.995)
    plt.tight_layout()
    out = OUT / "fake_grid.png"
    plt.savefig(out, facecolor="white")
    plt.close()
    print(f"✓ {out}")


# ─── 4. Valid grid ──────────────────────────────────────────────────
def valid_grid():
    items = []
    with LABELS.open() as f:
        for r in csv.DictReader(f):
            if int(r["label"]) == 0:
                p = ROOT / r["path"]
                if p.exists():
                    items.append(p)
    if len(items) < 12:
        return
    random.Random(7).shuffle(items)
    pick = items[:12]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for ax, p in zip(axes.flat, pick):
        try:
            img = Image.open(p).convert("RGB")
            img.thumbnail((400, 400))
            ax.imshow(img)
        except Exception:
            pass
        ax.axis("off")
    fig.suptitle("Representative VALID Payment Receipts",
                 fontsize=14, fontweight="bold", color="#111827", y=0.995)
    plt.tight_layout()
    out = OUT / "valid_grid.png"
    plt.savefig(out, facecolor="white")
    plt.close()
    print(f"✓ {out}")


if __name__ == "__main__":
    training_curves()
    confusion_matrix()
    fake_grid()
    valid_grid()
    print("\nAll figures regenerated in docs/figures/")
