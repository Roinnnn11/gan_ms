# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Use the `lry` conda environment for all training and evaluation tasks. Full conda activation is required — the activate scripts set critical `LD_LIBRARY_PATH` entries for CUDA. Using the bare Python binary bypasses these and fails with "Unsupported device target GPU".

- MindSpore 2.9.0 (GPU), CUDA 11.6 shim, cuDNN 8.4
- Do NOT use `ms_gpu`, `base`, or system Python

## Commands

**Train:**
```bash
# DCGAN
bash -c "source /data1/liurongying/miniconda3/etc/profile.d/conda.sh && conda activate lry && python scripts/train.py --config configs/dcgan_celeba.yaml"

# WGAN-GP
bash -c "source /data1/liurongying/miniconda3/etc/profile.d/conda.sh && conda activate lry && python scripts/train.py --config configs/wgan_gp_celeba.yaml"

# Resume from checkpoint (e.g. epoch 26)
bash -c "source ... && conda activate lry && python scripts/train.py --config configs/wgan_gp_celeba.yaml --resume-epoch 26"
```

**Background training:**
```bash
bash -c "source /data1/liurongying/miniconda3/etc/profile.d/conda.sh && conda activate lry && python scripts/train.py --config configs/wgan_gp_celeba.yaml" > outputs/wgan_gp/logs/train.log 2>&1 &
```

**Tests:**
```bash
bash -c "source /data1/liurongying/miniconda3/etc/profile.d/conda.sh && conda activate lry && python -m pytest tests/ -v"
```

**Single test:**
```bash
bash -c "source /data1/liurongying/miniconda3/etc/profile.d/conda.sh && conda activate lry && python -m pytest tests/test_models.py::test_generator_output_shape -v"
```

**Evaluate FID + IS:**
```bash
# 1. Prepare real images (one-time, deterministic names)
python scripts/prepare_real_images.py --source-dir data/celeba/img_align_celeba --output-dir outputs/eval/real_10k --num-images 10000

# 2. Export generated images
python scripts/export_generated_images.py --config configs/dcgan_celeba.yaml --checkpoint outputs/dcgan/checkpoints/generator_latest.ckpt --output-dir outputs/eval/dcgan_10k --num-images 10000 --seed 2020

# 3. Run evaluation (primary: torch-fidelity backend)
python scripts/evaluate.py --real-dir outputs/eval/real_10k --fake-dir outputs/eval/dcgan_10k --backend torch --output outputs/eval/dcgan_torch.json
```

## Architecture

**Config → Trainer → Models** is the main data flow. `src/config.py` defines frozen dataclasses (`ExperimentConfig`, `DataConfig`, `TrainConfig`, `WganGpConfig`) loaded from YAML. `scripts/train.py` loads config, sets MindSpore context to `GRAPH_MODE`, and instantiates the appropriate trainer.

**Models** (`src/models/dcgan.py`): Single file with `Generator` and `Discriminator`. Both are used for DCGAN and WGAN-GP — the key difference is `use_sigmoid=False` and no BatchNorm layers in the Discriminator when used as a WGAN-GP critic. The WGAN-GP Critic must not use BatchNorm because it creates intra-batch correlations that invalidate the gradient penalty.

**Trainers**: DCGAN trainer wraps `_DLoss`/`_GLoss` cells in `nn.TrainOneStepCell`. WGAN-GP trainer uses `ops.value_and_grad` directly. The gradient penalty is computed with `ops.grad` on an inline `critic_sum` closure. Critic is updated every step; generator every `critic_steps` steps (default 5).

**Metrics** (`src/metrics/`): Two evaluation backends — `torch_fidelity_backend.py` (primary for reports) and `mindspore_backend.py` (native). Do not mix backends in the same comparison table. Core math (FID via Fréchet distance, IS via KL divergence splits) is in `core.py` — pure NumPy, backend-independent.

**Data** (`src/data/celeba.py`): `ImageBytesDataset` reads raw bytes; MindSpore's `map()` pipeline decodes and applies transforms (resize to 64×64, HWC2CHW, rescale to [-1, 1]).

**Outputs layout:**
```
outputs/<experiment>/
  checkpoints/   generator/critic/discriminator_epoch_NNN.ckpt
  samples/       epoch_NNN.png  (grid of 64 images)
  logs/losses.csv
```

## Known MindSpore 2.9 Constraints

- `ops.randn` must be called as `ops.randn(n, c, h, w).astype(ms.float32)`, not `ops.randn((tuple,), dtype=...)`.
- `GroupNorm` backprop is not defined (`GroupNormGrad bprop not defined`); do not use it.
- Adam `beta1` must be strictly positive (> 0.0); use 0.001 as WGAN-GP paper's β₁=0 approximation.
- Two `ops.value_and_grad` calls that share model parameters in GRAPH_MODE can trigger "param already exists" at graph compile time. The DCGAN trainer avoids this by using `nn.TrainOneStepCell` with `auto_prefix=False`.

## Test Results (2026-06-20) — 20 passed, 0 failed

```
tests/test_config.py, test_evaluate_cli.py, test_images.py,
test_metric_image_io.py, test_metrics_backends.py, test_metrics_core.py,
test_models.py — all pass
```
