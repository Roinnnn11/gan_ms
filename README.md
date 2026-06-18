# gan_ms

MindSpore implementation of GAN image generation on CelebA for the course project.

The first version provides:

- DCGAN training on 64x64 CelebA images.
- WGAN-GP training on 64x64 CelebA images.
- Checkpoint saving.
- Generated sample grids.
- CSV loss logs for report plots.

FID and Inception Score are available through standard `torch-fidelity` and native MindSpore/MindCV backends.

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

## Evaluate FID and Inception Score

Prepare exactly the same number of real and generated 64x64 images. A 10,000-image run is useful for debugging; use 50,000 images for the final report when time allows.

```powershell
python scripts/prepare_real_images.py --source-dir data/celeba/img_align_celeba --output-dir outputs/eval/real_10k --num-images 10000

python scripts/export_generated_images.py --config configs/dcgan_celeba.yaml --checkpoint outputs/dcgan/checkpoints/generator_latest.ckpt --output-dir outputs/eval/dcgan_10k --num-images 10000
```

The report's primary metric should use `torch-fidelity` in a PyTorch evaluation environment:

```powershell
pip install -r requirements-eval-torch.txt
python scripts/evaluate.py --real-dir outputs/eval/real_10k --fake-dir outputs/eval/dcgan_10k --backend torch --output outputs/eval/dcgan_torch.json
```

The native comparison uses MindCV InceptionV3 in the MindSpore environment:

```powershell
pip install -r requirements-eval-mindspore.txt
python scripts/evaluate.py --real-dir outputs/eval/real_10k --fake-dir outputs/eval/dcgan_10k --backend mindspore --output outputs/eval/dcgan_mindspore.json
```

If both dependency stacks are available in one environment, use `--backend both`. Do not mix the two backends in one comparison table column because their Inception weights and preprocessing protocols differ. See `docs/METRICS_SOURCES.md` for citations and reporting guidance.

## Project Structure

```text
configs/              YAML experiment configs
scripts/              CLI entrypoints
src/data/             CelebA image loading
src/models/           DCGAN generator and discriminator
src/trainers/         DCGAN and WGAN-GP training loops
src/utils/            Checkpoint and image utilities
src/metrics/          FID and Inception Score formulas and backends
tests/                Lightweight unit tests
```
