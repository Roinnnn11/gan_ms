# gan_ms

MindSpore implementation of GAN image generation on CelebA for the course project.

The first version provides:

- DCGAN training on 64x64 CelebA images.
- WGAN-GP training on 64x64 CelebA images.
- Checkpoint saving.
- Generated sample grids.
- CSV loss logs for report plots.

FID and Inception Score will be added in a later version.

## Environment

Use a MindSpore GPU environment that matches your CUDA driver. Install the basic Python dependencies after MindSpore is installed:

```powershell
pip install -r requirements.txt
```

## Data Layout

Put CelebA aligned images under:

```text
data/celeba/img_align_celeba/
```

You can also change `data_root` in the YAML configs.

## Train

```powershell
python scripts/train.py --config configs/dcgan_celeba.yaml
python scripts/train.py --config configs/wgan_gp_celeba.yaml
```

Training outputs are written to:

```text
outputs/<experiment>/
  checkpoints/
  samples/
  logs/losses.csv
```

## Generate

```powershell
python scripts/generate.py --config configs/dcgan_celeba.yaml --checkpoint outputs/dcgan/checkpoints/generator_latest.ckpt
```

## Project Structure

```text
configs/              YAML experiment configs
scripts/              CLI entrypoints
src/data/             CelebA image loading
src/models/           DCGAN generator and discriminator
src/trainers/         DCGAN and WGAN-GP training loops
src/utils/            Checkpoint and image utilities
src/metrics/          Reserved for FID and Inception Score
tests/                Lightweight unit tests
```

## Next Version

Add FID and Inception Score under `src/metrics/`, then write generated images to a temporary folder and compare them with real CelebA images.
