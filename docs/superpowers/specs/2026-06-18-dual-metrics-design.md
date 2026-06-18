# Dual FID and Inception Score Evaluation Design

## Goal

Add reproducible FID and Inception Score evaluation to `gan_ms` with two backends: a standard `torch-fidelity` backend for reportable primary results and a MindSpore/MindCV backend for a native ecosystem comparison.

## Evaluation Protocol

- Real and generated images are stored as individual RGB PNG/JPEG files.
- Real CelebA images are resized to the same 64x64 square format used by training before evaluation.
- Debug runs use 10,000 images; final report runs should use 50,000 images when time allows.
- DCGAN and WGAN-GP use the same real-image cache, sample count, seed, and split count.
- The standard and MindCV results are labeled separately because their Inception weights and preprocessing are not numerically equivalent.

## Components

- `src/metrics/core.py`: framework-independent FID and Inception Score formulas, validation, and result records.
- `src/metrics/torch_fidelity_backend.py`: delegates feature extraction and standard FID/IS calculation to `torch-fidelity`.
- `src/metrics/mindspore_backend.py`: loads pretrained MindCV InceptionV3, extracts 2048-dimensional pooled features and 1000-class logits, then calls the shared formulas.
- `scripts/prepare_real_images.py`: creates a deterministic 64x64 real-image cache from CelebA.
- `scripts/export_generated_images.py`: loads a MindSpore generator checkpoint and writes individual generated images in batches.
- `scripts/evaluate.py`: runs `torch`, `mindspore`, or `both` backends and writes JSON.
- `docs/METRICS_SOURCES.md`: records papers, source repositories, licenses, formulas, implementation differences, and report-ready citations.

## Error Handling

Each backend imports optional dependencies only when selected. Missing `torch-fidelity`, MindCV, SciPy, checkpoints, image directories, or insufficient samples produce actionable errors. The CLI refuses unknown backends and prevents empty image directories from reaching feature extraction.

## Testing

Pure NumPy tests verify that identical distributions have approximately zero FID, shifted distributions have positive FID, uniform logits have IS approximately one, and diverse confident logits score above one. Backend tests mock optional libraries, while syntax checks cover MindSpore-dependent files when those packages are unavailable locally.

## Reporting

`torch-fidelity` values are the primary table values and are labeled `FID/IS (torch-fidelity)`. MindCV values are labeled `FID/IS (MindCV)` and used as an implementation comparison, not mixed directly with the standard values.
