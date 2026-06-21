from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Pre-load GPU plugin so MindSpore C++ can find it via dlopen
_gpu_plugin = PROJECT_ROOT / "src" / ".." / "."
_ms_plugin = Path(sys.prefix) / "lib/python3.9/site-packages/mindspore/lib/plugin/libmindspore_gpu.so.11.6"
if _ms_plugin.exists():
    ctypes.CDLL(str(_ms_plugin), mode=ctypes.RTLD_GLOBAL)

import mindspore as ms

from src.config import load_config
from src.data.celeba import create_celeba_dataset
from src.trainers.dcgan_trainer import DCGANTrainer
from src.trainers.wgan_gp_trainer import WGANGPTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DCGAN or WGAN-GP on CelebA.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--resume-epoch", type=int, default=1,
                        help="Resume from this epoch (loads epoch-1 checkpoint). Default: 1 (fresh).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ms.set_seed(config.seed)
    ms.set_context(
        mode=ms.GRAPH_MODE,
        device_target=config.device_target,
        device_id=config.device_id,
    )
    dataset = create_celeba_dataset(config.data)

    if config.model == "dcgan":
        trainer = DCGANTrainer(config)
    elif config.model == "wgan_gp":
        trainer = WGANGPTrainer(config)
    else:
        raise ValueError(f"Unsupported model: {config.model}")

    trainer.train(dataset, start_epoch=args.resume_epoch)


if __name__ == "__main__":
    main()
