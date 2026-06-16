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

## Generate

```powershell
python scripts/generate.py --config configs/dcgan_celeba.yaml --checkpoint outputs/dcgan/checkpoints/generator_latest.ckpt
```
