"""
Simple script to download MNIST dataset using utils_data.py
"""
import sys
from pathlib import Path

# Add src to path if needed
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils_data import get_torchvision_dataset

def main():
    print("Downloading MNIST dataset...")

    # Download MNIST to ../../data/bigdata directory
    train_ds, test_ds, info = get_torchvision_dataset(
        name="mnist",
        root="../../data/bigdata",
        download=True,
    )

    print(f"\n✓ MNIST dataset downloaded successfully!")
    print(f"\nDataset Info:")
    print(f"  Name: {info.name}")
    print(f"  Location: {info.root}")
    print(f"  Image size: {info.image_size}x{info.image_size}")
    print(f"  Channels: {info.channels}")
    print(f"  Classes: {info.num_classes}")
    print(f"  Mean: {info.mean}")
    print(f"  Std: {info.std}")
    print(f"\nDataset Sizes:")
    print(f"  Training samples: {len(train_ds)}")
    print(f"  Test samples: {len(test_ds)}")

    # Test loading a sample
    sample_img, sample_label = train_ds[0]
    print(f"\nSample image shape: {sample_img.shape}")
    print(f"Sample label: {sample_label}")

if __name__ == "__main__":
    main()
