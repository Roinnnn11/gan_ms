# gan_ms Design

## Goal

Build a MindSpore project for the course assignment topic "GAN image generation on CelebA". The first version must train and compare DCGAN and WGAN-GP at 64x64 resolution, save checkpoints, generate sample grids, and keep metric placeholders for later FID and Inception Score work.

## Scope

This version focuses on a runnable training pipeline:

- CelebA image folder loading and preprocessing.
- DCGAN generator and discriminator.
- WGAN-GP generator and critic.
- Separate training commands for DCGAN and WGAN-GP.
- Periodic sample image export and checkpoint saving.
- Loss CSV logging for report plots.
- Git-managed project structure.

FID and Inception Score are not implemented in the first version. The project reserves `src/metrics/` and CLI placeholders so those can be added cleanly in the next version.

## Architecture

The code is split by responsibility:

- `src/config.py` loads YAML config files into typed dataclasses.
- `src/data/celeba.py` builds a MindSpore dataset from an image directory.
- `src/models/dcgan.py` contains reusable generator and discriminator/critic networks.
- `src/trainers/dcgan_trainer.py` handles BCE-based DCGAN training.
- `src/trainers/wgan_gp_trainer.py` handles Wasserstein loss and gradient penalty training.
- `src/utils/images.py` saves normalized tensors as sample grids.
- `src/utils/checkpoint.py` centralizes checkpoint paths and saving.
- `scripts/train.py` is the main training entrypoint.
- `scripts/generate.py` loads a checkpoint and writes generated samples.

## Data Flow

The user points `data_root` to a CelebA image directory such as `data/celeba/img_align_celeba`. The dataset loader decodes images, resizes to 64x64, normalizes pixels to `[-1, 1]`, batches them, and feeds them to a trainer. Each trainer samples latent noise, updates the discriminator or critic, updates the generator, logs losses, saves checkpoints, and writes sample image grids.

## Error Handling

The scripts fail early when the config file is missing, the data directory does not exist, or a checkpoint path is invalid. Training output directories are created automatically. FID and Inception Score were added in the second implementation iteration with separate standard and MindSpore backends.

## Testing

The first version includes lightweight tests that do not require real CelebA data:

- Config loading from YAML.
- Network forward shape checks.
- Image grid utility shape/range behavior.

Full training validation is done by a smoke command with a tiny image folder after MindSpore and CelebA are available.
