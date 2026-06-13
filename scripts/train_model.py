"""
KrishiMitra AI — Model Training Script
Trains EfficientNet-B0 on PlantVillage dataset.
Run: python scripts/train_model.py
"""

import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.models.disease_classifier import build_model, get_device
from app.utils.image_preprocessor import get_training_transforms, get_inference_transforms
from app.core.logger import logger

# ── Config ─────────────────────────────────────────────────
PROCESSED_DIR   = Path("data/processed")
CHECKPOINT_DIR  = Path("models/checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "efficientnet_b0_best.pth"

EPOCHS      = 10
BATCH_SIZE  = 16
LR          = 1e-3
PATIENCE    = 3   # Early stopping patience


def load_datasets():
    """Load train/val datasets using ImageFolder."""
    train_dir = PROCESSED_DIR / "train"
    val_dir   = PROCESSED_DIR / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training data not found at {train_dir}. Run download_dataset.py first.")

    train_ds = datasets.ImageFolder(str(train_dir), transform=get_training_transforms())
    val_ds   = datasets.ImageFolder(str(val_dir),   transform=get_inference_transforms())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    logger.info(f"Train: {len(train_ds)} images | {len(train_ds.classes)} classes")
    logger.info(f"Val  : {len(val_ds)} images")

    return train_loader, val_loader, train_ds.classes


def train_epoch(model, loader, optimizer, criterion, device):
    """One training epoch."""
    model.train()
    # Unfreeze classifier head for training
    for param in model.classifier.parameters():
        param.requires_grad = True

    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

        if (batch_idx + 1) % 5 == 0:
            print(f"   Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    """Validation loop."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


def train():
    """Main training loop with early stopping."""
    print("\n🌱 KrishiMitra AI — Model Training")
    print("=" * 50)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    # Load data
    train_loader, val_loader, classes = load_datasets()
    num_classes = len(classes)
    print(f"Classes detected: {num_classes}")
    print(f"Device: {device}\n")

    # Build model
    model     = build_model(num_classes=num_classes, pretrained=True)
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_val_acc  = 0.0
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 40)
        start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start
        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print(f"  Time: {elapsed:.1f}s")

        logger.info(
            f"Epoch {epoch} | train_acc={train_acc:.2f}% | "
            f"val_acc={val_acc:.2f}% | val_loss={val_loss:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  ✅ Best model saved! Val Acc: {val_acc:.2f}%")
            patience_count = 0
        else:
            patience_count += 1
            print(f"  ⏳ No improvement ({patience_count}/{PATIENCE})")

        # Early stopping
        if patience_count >= PATIENCE:
            print(f"\n⏹  Early stopping triggered at epoch {epoch}")
            break

    print(f"\n{'='*50}")
    print(f"✅ Training complete!")
    print(f"   Best Val Accuracy : {best_val_acc:.2f}%")
    print(f"   Checkpoint saved  : {CHECKPOINT_PATH}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    train()
