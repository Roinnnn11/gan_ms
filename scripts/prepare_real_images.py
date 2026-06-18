from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.image_io import prepare_real_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a deterministic real CelebA image cache for FID.")
    parser.add_argument("--source-dir", required=True, help="Directory containing original CelebA images.")
    parser.add_argument("--output-dir", required=True, help="Empty directory for processed PNG images.")
    parser.add_argument("--image-size", type=int, default=64, help="Square output image size.")
    parser.add_argument("--num-images", type=int, default=10000, help="Number of real images to export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = prepare_real_images(
        args.source_dir,
        args.output_dir,
        image_size=args.image_size,
        num_images=args.num_images,
    )
    print(f"Prepared {count} real images in {args.output_dir}")


if __name__ == "__main__":
    main()
