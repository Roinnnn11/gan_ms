# DCGAN 测试结果

**日期：** 2026-06-20  
**环境：** conda `ms_gpu` / Python 3.9.25 / pytest 8.4.2  
**运行命令：**
```
pytest tests/test_models.py tests/test_config.py tests/test_images.py -v
```

## 结果汇总

| 测试文件 | 测试用例 | 结果 |
|---|---|---|
| `test_models.py` | `test_generator_output_shape` | PASSED |
| `test_models.py` | `test_discriminator_output_shape` | PASSED |
| `test_config.py` | `test_load_dcgan_config` | PASSED |
| `test_images.py` | `test_tensor_to_uint8_images_converts_nchw_range` | PASSED |
| `test_images.py` | `test_make_image_grid_uses_square_layout_by_default` | PASSED |

**5 passed, 0 failed，耗时 1.40s**

---

## 性能指标评估

**日期：** 2026-06-20  
**检查点：** `outputs/dcgan/checkpoints/generator_epoch_025.ckpt`（25 epoch，2026-06-20 训练完成）  
**评估集：** 10,000 张生成图像 vs. 10,000 张真实 CelebA 图像  
**评估工具：** torch-fidelity（`scripts/evaluate.py --backend torch`）  
**训练配置：** epochs=25, lr=0.0002, batch_size=128, latent_dim=100, feature_maps=64

| 指标 | 数值 |
|---|---|
| FID (↓ 越低越好) | **30.62** |
| Inception Score 均值 (↑ 越高越好) | **2.78** |
| Inception Score 标准差 | ±0.060 |

**解读：**

- FID 30.62 处于 DCGAN 在 CelebA 64×64 上的典型范围（学术复现通常 20–50），说明生成分布与真实分布有一定对齐，但仍有优化空间。
- IS 2.78 偏低，主要原因是 CelebA 属于单类别（人脸）数据集，IS 对多样性的衡量在此场景下天然偏低，该值在 CelebA 上属正常水平，不宜与 ImageNet 结果直接对比。
- 进一步改善方向：增加训练 epoch、调高 feature_maps、或切换至 WGAN-GP。

完整评估报告：`outputs/eval/dcgan_metrics.json`

---

## 各用例说明

- **test_generator_output_shape**：Generator(latent_dim=100, feature_maps=16) 接收 `(2, 100, 1, 1)` 噪声，输出形状正确为 `(2, 3, 64, 64)`。
- **test_discriminator_output_shape**：Discriminator(feature_maps=16) 接收 `(2, 3, 64, 64)` 图像，输出形状正确为 `(2, 1)`。
- **test_load_dcgan_config**：从 `configs/dcgan_celeba.yaml` 加载配置，model/image_size/batch_size/latent_dim/output_dir 字段均符合预期。
- **test_tensor_to_uint8_images_converts_nchw_range**：NCHW float32 张量（值域 `[-1, 1]` 及溢出值）正确转换为 NHWC uint8，裁剪与缩放行为符合预期。
- **test_make_image_grid_uses_square_layout_by_default**：5 张 `(3, 4, 4)` 图像默认按最小正方形布局拼成 `(8, 12, 3)` 网格。
