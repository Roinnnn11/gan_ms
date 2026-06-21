# gan_ms Project Notes

## Environment

Use the `lry` conda environment for all training and evaluation tasks.

```bash
conda activate lry
```

This environment is configured with:
- MindSpore 2.9.0 (GPU)
- CUDA 11.6 shim at `/data1/liurongying/miniconda3/envs/lry/cuda-11.6-shim`
- cuDNN 8.4 + CUDA 11 runtime libraries
- Auto-activation scripts at:
  - `/data1/liurongying/miniconda3/envs/lry/etc/conda/activate.d/mindspore-gpu.sh`
  - `/data1/liurongying/miniconda3/envs/lry/etc/conda/deactivate.d/mindspore-gpu.sh`

Do NOT use `ms_gpu`, `base`, or system Python — they lack the correct CUDA/MindSpore setup.

## Running Training

Always use full conda activation — the activate scripts set critical `LD_LIBRARY_PATH` entries for CUDA. Using the bare Python binary (`/envs/lry/bin/python`) bypasses these and will fail with "Unsupported device target GPU".

```bash
# DCGAN
bash -c "source /data1/liurongying/miniconda3/etc/profile.d/conda.sh && conda activate lry && python scripts/train.py --config configs/dcgan_celeba.yaml"

# WGAN-GP
bash -c "source /data1/liurongying/miniconda3/etc/profile.d/conda.sh && conda activate lry && python scripts/train.py --config configs/wgan_gp_celeba.yaml"
```

Background (append `&` and redirect logs):
```bash
bash -c "source /data1/liurongying/miniconda3/etc/profile.d/conda.sh && conda activate lry && python scripts/train.py --config configs/wgan_gp_celeba.yaml" > outputs/wgan_gp/logs/train.log 2>&1 &
```

## Running Tests

```bash
bash -c "source /data1/liurongying/miniconda3/etc/profile.d/conda.sh && conda activate lry && python -m pytest tests/ -v"
```

### Test results (2026-06-20) — 20 passed, 0 failed, 4.44s

| Test | Result |
|------|--------|
| test_config.py::test_load_dcgan_config | PASSED |
| test_evaluate_cli.py::test_both_backend_results_are_written_to_json | PASSED |
| test_evaluate_cli.py::test_evaluation_rejects_different_sample_counts | PASSED |
| test_images.py::test_tensor_to_uint8_images_converts_nchw_range | PASSED |
| test_images.py::test_make_image_grid_uses_square_layout_by_default | PASSED |
| test_metric_image_io.py::test_discovery_is_recursive_and_sorted | PASSED |
| test_metric_image_io.py::test_prepare_real_images_rejects_nonempty_output | PASSED |
| test_metric_image_io.py::test_prepare_real_images_resizes_and_uses_stable_names | PASSED |
| test_metric_image_io.py::test_save_generated_images_converts_minus_one_to_one_range | PASSED |
| test_metrics_backends.py::test_backend_normalizes_metric_names | PASSED |
| test_metrics_backends.py::test_backend_rejects_missing_directory_before_import | PASSED |
| test_metrics_backends.py::test_shared_formula_adapter_returns_normalized_metrics | PASSED |
| test_metrics_core.py::test_confident_diverse_logits_have_higher_inception_score | PASSED |
| test_metrics_core.py::test_identical_features_have_zero_fid | PASSED |
| test_metrics_core.py::test_inception_score_rejects_more_splits_than_samples | PASSED |
| test_metrics_core.py::test_shifted_features_have_positive_fid | PASSED |
| test_metrics_core.py::test_statistics_reject_single_sample | PASSED |
| test_metrics_core.py::test_uniform_logits_have_inception_score_one | PASSED |
| test_models.py::test_generator_output_shape | PASSED |
| test_models.py::test_discriminator_output_shape | PASSED |
