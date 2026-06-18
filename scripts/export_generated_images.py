from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mindspore as ms
import mindspore.ops as ops

from src.config import load_config
from src.metrics.image_io import save_generated_images
from src.models.dcgan import Generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export individual generated images for FID and IS evaluation.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a generator checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Empty output directory for generated PNG images.")
    parser.add_argument("--num-images", type=int, default=10000, help="Number of images to generate.")
    parser.add_argument("--batch-size", type=int, default=128, help="Generation batch size.")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed used for latent noise.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint}")
    if args.num_images <= 0 or args.batch_size <= 0:
        raise ValueError("num-images and batch-size must be positive.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    ms.set_seed(args.seed)
    ms.set_context(
        mode=ms.GRAPH_MODE,
        device_target=config.device_target,
        device_id=config.device_id,
    )
    params = config.model_params
    generator = Generator(params.latent_dim, params.image_channels, params.feature_maps_g)
    ms.load_checkpoint(str(checkpoint), net=generator)
    generator.set_train(False)

    index = 0
    while index < args.num_images:
        current_batch = min(args.batch_size, args.num_images - index)
        noise = ops.randn((current_batch, params.latent_dim, 1, 1), ms.float32)
        images = generator(noise).asnumpy()
        index = save_generated_images(images, output_dir, start_index=index)

    print(f"Exported {index} generated images to {output_dir}")


if __name__ == "__main__":
    main()
