"""
KrishiMitra AI — PlantVillage Dataset Downloader
Creates realistic synthetic demo data for pipeline testing.
Run: python scripts/download_dataset.py
"""

import sys
import random
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.core.logger import logger
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Class-specific color signatures to make classes distinguishable
CLASS_PROFILES = {
    "Tomato___Early_blight": {
        "bg": (34, 120, 40),      # green leaf
        "spot_color": (80, 40, 10),  # brown spots
        "spots": 8
    },
    "Tomato___Late_blight":  {
        "bg": (30, 100, 35),
        "spot_color": (20, 20, 60),  # dark blue-brown spots
        "spots": 12
    },
    "Tomato___healthy":      {
        "bg": (40, 160, 50),      # bright healthy green
        "spot_color": None,
        "spots": 0
    },
    "Potato___Early_blight": {
        "bg": (50, 130, 45),
        "spot_color": (100, 60, 10),
        "spots": 6
    },
    "Potato___healthy":      {
        "bg": (55, 150, 55),
        "spot_color": None,
        "spots": 0
    },
}


def make_leaf_image(profile: dict, size: int = 256) -> Image.Image:
    """Creates a synthetic leaf image with class-specific features."""
    r, g, b = profile["bg"]
    # Add noise for variety
    r = max(0, min(255, r + random.randint(-20, 20)))
    g = max(0, min(255, g + random.randint(-20, 20)))
    b = max(0, min(255, b + random.randint(-20, 20)))

    img = Image.new("RGB", (size, size), color=(r, g, b))
    draw = ImageDraw.Draw(img)

    # Draw leaf vein pattern
    for _ in range(5):
        x1 = random.randint(0, size)
        y1 = random.randint(0, size)
        x2 = random.randint(0, size)
        y2 = random.randint(0, size)
        vein_color = (max(0,r-20), min(255,g+10), max(0,b-10))
        draw.line([(x1,y1),(x2,y2)], fill=vein_color, width=2)

    # Draw disease spots if applicable
    if profile["spot_color"] and profile["spots"] > 0:
        sr, sg, sb = profile["spot_color"]
        for _ in range(profile["spots"]):
            x = random.randint(20, size-20)
            y = random.randint(20, size-20)
            radius = random.randint(8, 25)
            draw.ellipse(
                [(x-radius, y-radius), (x+radius, y+radius)],
                fill=(sr, sg, sb)
            )

    # Slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    return img


def create_demo_dataset() -> bool:
    try:
        logger.info("Creating improved demo dataset...")

        splits = {"train": 60, "val": 20, "test": 10}

        for split, count in splits.items():
            for cls, profile in CLASS_PROFILES.items():
                class_dir = PROCESSED_DIR / split / cls
                class_dir.mkdir(parents=True, exist_ok=True)
                for i in range(count):
                    img = make_leaf_image(profile)
                    img.save(class_dir / f"img_{i:04d}.jpg", quality=90)

        total = sum(1 for _ in PROCESSED_DIR.rglob("*.jpg"))
        logger.info(f"Demo dataset created | {total} images | {len(CLASS_PROFILES)} classes")
        print(f"\n{'='*50}")
        print(f"✅ Improved demo dataset ready!")
        print(f"   Classes : {len(CLASS_PROFILES)}")
        print(f"   Images  : {total}")
        print(f"   Location: {PROCESSED_DIR}")
        print(f"{'='*50}\n")
        return True

    except Exception as e:
        logger.error(f"Demo dataset creation failed: {e}")
        return False


def verify_dataset() -> dict:
    stats = {}
    for split in ["train", "val", "test"]:
        split_dir = PROCESSED_DIR / split
        if split_dir.exists():
            classes = [d.name for d in split_dir.iterdir() if d.is_dir()]
            images  = list(split_dir.rglob("*.jpg")) + list(split_dir.rglob("*.png"))
            stats[split] = {"classes": len(classes), "images": len(images)}
    return stats


if __name__ == "__main__":
    import shutil
    print("\n🌱 KrishiMitra AI — Dataset Setup")
    print("="*50)

    # Clean old synthetic data
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
        print("🗑  Cleaned old dataset")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    success = create_demo_dataset()

    stats = verify_dataset()
    print("\n📊 Dataset Statistics:")
    for split, info in stats.items():
        print(f"   {split:6s}: {info['images']:4d} images | {info['classes']} classes")
