"""
KrishiMitra AI — Model Evaluator
Evaluates trained model on test set.
Run: python scripts/evaluate_model.py
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from app.models.disease_classifier import build_model, get_device
from app.utils.image_preprocessor import get_inference_transforms
from app.core.logger import logger

PROCESSED_DIR   = Path("data/processed")
CHECKPOINT_PATH = Path("models/checkpoints/efficientnet_b0_best.pth")


def evaluate():
    print("\n🌱 KrishiMitra AI — Model Evaluation")
    print("=" * 50)

    device = get_device()

    # ── Load test dataset ─────────────────────────────────
    test_dir = PROCESSED_DIR / "test"
    if not test_dir.exists():
        print(f"❌ Test data not found at {test_dir}")
        sys.exit(1)

    test_ds     = datasets.ImageFolder(str(test_dir), transform=get_inference_transforms())
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)
    classes     = test_ds.classes
    num_classes = len(classes)

    print(f"Test images : {len(test_ds)}")
    print(f"Classes     : {num_classes}")
    print(f"Classes     : {classes}\n")

    # ── Load model ────────────────────────────────────────
    model = build_model(num_classes=num_classes, pretrained=False)
    if CHECKPOINT_PATH.exists():
        state = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state)
        print(f"✅ Checkpoint loaded: {CHECKPOINT_PATH}")
    else:
        print(f"⚠️  No checkpoint found — using untrained model")

    model.to(device)
    model.eval()

    # ── Evaluation loop ───────────────────────────────────
    criterion   = nn.CrossEntropyLoss()
    total_loss  = 0.0
    correct     = 0
    total       = 0

    # Per-class tracking
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)
    all_preds     = []
    all_labels    = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs  = model(images)
            loss     = criterion(outputs, labels)
            total_loss += loss.item()

            probs        = torch.softmax(outputs, dim=1)
            confidences, predicted = probs.max(1)

            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

            for pred, label in zip(predicted, labels):
                class_correct[label.item()] += (pred == label).item()
                class_total[label.item()]   += 1

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # ── Results ───────────────────────────────────────────
    overall_acc  = 100.0 * correct / total
    avg_loss     = total_loss / len(test_loader)

    print(f"\n{'='*50}")
    print(f"📊 EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Overall Accuracy : {overall_acc:.2f}%")
    print(f"  Average Loss     : {avg_loss:.4f}")
    print(f"  Total Samples    : {total}")
    print(f"\n  Per-Class Accuracy:")
    print(f"  {'-'*40}")

    for class_idx, class_name in enumerate(classes):
        if class_total[class_idx] > 0:
            acc = 100.0 * class_correct[class_idx] / class_total[class_idx]
            bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
            print(f"  {class_name[:30]:30s} | {bar} | {acc:.1f}%")

    print(f"\n{'='*50}")

    # ── Pass/Fail for internship ───────────────────────────
    TARGET_ACC = 75.0  # Realistic target for demo dataset
    if overall_acc >= TARGET_ACC:
        print(f"  ✅ PASS — Accuracy {overall_acc:.2f}% >= {TARGET_ACC}% target")
    else:
        print(f"  ⚠️  BELOW TARGET — {overall_acc:.2f}% < {TARGET_ACC}%")
        print(f"     (Expected on real PlantVillage data: 85%+)")

    print(f"{'='*50}\n")

    logger.info(f"Evaluation complete | accuracy={overall_acc:.2f}% | loss={avg_loss:.4f}")
    return overall_acc


if __name__ == "__main__":
    evaluate()
