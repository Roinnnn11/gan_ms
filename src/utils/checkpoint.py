from __future__ import annotations

from pathlib import Path

import mindspore as ms


def ensure_run_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    paths = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "samples": root / "samples",
        "logs": root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_named_checkpoint(network: ms.nn.Cell, checkpoint_dir: Path, name: str, epoch: int) -> None:
    latest_path = checkpoint_dir / f"{name}_latest.ckpt"
    epoch_path = checkpoint_dir / f"{name}_epoch_{epoch:03d}.ckpt"
    ms.save_checkpoint(network, str(latest_path))
    ms.save_checkpoint(network, str(epoch_path))
