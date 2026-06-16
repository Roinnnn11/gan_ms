from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mindspore as ms
import mindspore.ops as ops

from src.config import load_config
from src.models.dcgan import Generator
from src.utils.images import save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images from a trained generator.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to generator checkpoint.")
    parser.add_argument("--output", default="outputs/generated.png", help="Output image path.")
    parser.add_argument("--num-images", type=int, default=64, help="Number of images to generate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    config = load_config(args.config)
    ms.set_context(
        mode=ms.GRAPH_MODE,
        device_target=config.device_target,
        device_id=config.device_id,
    )
    params = config.model_params
    generator = Generator(params.latent_dim, params.image_channels, params.feature_maps_g)
    ms.load_checkpoint(str(checkpoint), net=generator)
    noise = ops.randn((args.num_images, params.latent_dim, 1, 1), ms.float32)
    images = generator(noise).asnumpy()
    save_image_grid(images, args.output)


if __name__ == "__main__":
    main()
