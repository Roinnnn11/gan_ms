from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Pre-load GPU plugin
_ms_plugin = Path(sys.prefix) / "lib/python3.9/site-packages/mindspore/lib/plugin/libmindspore_gpu.so.11.6"
if _ms_plugin.exists():
    ctypes.CDLL(str(_ms_plugin), mode=ctypes.RTLD_GLOBAL)

import mindspore as ms
from mindspore.communication import init, get_rank, get_group_size

from src.config import load_config
from src.data.celeba import create_celeba_dataset_dist
from src.trainers.dcgan_trainer_dist import DCGANDistTrainer
from src.trainers.wgan_gp_trainer_dist import WGANGPDistTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-GPU train DCGAN or WGAN-GP on CelebA.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")
    init("nccl")
    rank = get_rank()
    world_size = get_group_size()
    ms.set_seed(config.seed + rank)

    dataset = create_celeba_dataset_dist(config.data, rank, world_size)

    if config.model == "dcgan":
        trainer = DCGANDistTrainer(config)
    elif config.model == "wgan_gp":
        trainer = WGANGPDistTrainer(config)
    else:
        raise ValueError(f"Unsupported model: {config.model}")

    if rank == 0:
        print(f"[rank0] Starting {config.model} on {world_size} GPUs")
    trainer.train(dataset)


if __name__ == "__main__":
    main()
