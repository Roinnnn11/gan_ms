from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.image_io import discover_image_paths
from src.metrics.mindspore_backend import evaluate_with_mindspore
from src.metrics.torch_fidelity_backend import evaluate_with_torch_fidelity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate generated images with FID and Inception Score.")
    parser.add_argument("--real-dir", required=True, help="Prepared real image directory.")
    parser.add_argument("--fake-dir", required=True, help="Generated image directory.")
    parser.add_argument(
        "--backend",
        choices=("torch", "mindspore", "both"),
        default="torch",
        help="Evaluation backend. torch-fidelity is the primary report protocol.",
    )
    parser.add_argument("--output", required=True, help="JSON file for metrics and protocol metadata.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inception inference batch size.")
    parser.add_argument("--splits", type=int, default=10, help="Number of Inception Score splits.")
    parser.add_argument("--seed", type=int, default=2020, help="Seed used when shuffling IS splits.")
    parser.add_argument("--torch-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--device-target", choices=("GPU", "CPU", "Ascend"), default="GPU")
    parser.add_argument("--device-id", type=int, default=0)
    return parser


def run_evaluation(args: argparse.Namespace) -> dict[str, object]:
    real_paths = discover_image_paths(args.real_dir)
    fake_paths = discover_image_paths(args.fake_dir)
    if len(real_paths) != len(fake_paths):
        raise ValueError(
            "Real and generated directories must contain the same number of images: "
            f"real={len(real_paths)}, fake={len(fake_paths)}."
        )
    if len(fake_paths) < 2:
        raise ValueError("Evaluation requires at least two real and two generated images.")
    if args.splits <= 0 or args.splits > len(fake_paths):
        raise ValueError("splits must be positive and no greater than the image count.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")

    results: dict[str, object] = {}
    if args.backend in ("torch", "both"):
        results["torch"] = evaluate_with_torch_fidelity(
            args.real_dir,
            args.fake_dir,
            batch_size=args.batch_size,
            cuda=args.torch_device == "cuda",
            splits=args.splits,
            seed=args.seed,
        )
    if args.backend in ("mindspore", "both"):
        results["mindspore"] = evaluate_with_mindspore(
            args.real_dir,
            args.fake_dir,
            batch_size=args.batch_size,
            splits=args.splits,
            seed=args.seed,
            device_target=args.device_target,
            device_id=args.device_id,
        )

    payload: dict[str, object] = {
        "protocol": {
            "backend": args.backend,
            "real_dir": str(Path(args.real_dir).resolve()),
            "fake_dir": str(Path(args.fake_dir).resolve()),
            "real_samples": len(real_paths),
            "fake_samples": len(fake_paths),
            "batch_size": args.batch_size,
            "is_splits": args.splits,
            "seed": args.seed,
        },
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    args = build_parser().parse_args()
    payload = run_evaluation(args)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
