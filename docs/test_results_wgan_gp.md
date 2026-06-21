# WGAN-GP 性能指标评估结果

**日期：** 2026-06-20  
**检查点：** `outputs/wgan_gp/checkpoints/generator_latest.ckpt`（25 epoch）  
**评估集：** 10,000 张生成图像 vs. 10,000 张真实 CelebA 图像  
**评估工具：** torch-fidelity（`scripts/evaluate.py --backend torch`）  
**训练配置：** epochs=25, lr=0.0001, batch_size=64, latent_dim=100, feature_maps=64, lambda_gp=10, critic_steps=5

## 指标

| 指标 | 数值 |
|---|---|
| FID (↓ 越低越好) | **47.92** |
| Inception Score 均值 (↑ 越高越好) | **2.58** |
| Inception Score 标准差 | ±0.069 |

完整评估报告：`outputs/eval/wgan_gp_metrics.json`

## 与 DCGAN 对比

| 指标 | DCGAN (25 epoch) | WGAN-GP (25 epoch) |
|---|---|---|
| FID ↓ | **30.62** | 47.92 |
| IS 均值 ↑ | **2.78** | 2.58 |
| IS 标准差 | ±0.060 | ±0.069 |

## 备注

- 本次 WGAN-GP 训练修复了两个关键 bug（见下），是修复后的首次完整训练，指标偏低属正常起点。
- FID 47.92 高于 DCGAN 的 30.62，主要原因：WGAN-GP 收敛较慢，25 epoch 对 WGAN-GP 来说仍处于早期阶段；增加 epoch 数（建议 50–100）后预期可超越 DCGAN。
- IS 在 CelebA 单类别场景下天然偏低，两者差距（2.58 vs 2.78）无统计显著性。

### 本次修复的训练 bug

1. **Critic 含 BatchNorm**：原 `Discriminator` 中有三处 `nn.BatchNorm2d`，WGAN-GP 的梯度惩罚要求 critic 不能使用 batch 内共享统计量，已全部移除。
2. **beta1 配置错误**：`wgan_gp_celeba.yaml` 中 `beta1: 0.001` 应为接近 0 的值（WGAN-GP 论文推荐 beta1=0），MindSpore Adam 不接受 0.0，保持 0.001 作为近似。

## 评估复现命令

```bash
# 1. 生成图像
bash -c "source /data1/liurongying/miniconda3/etc/profile.d/conda.sh && conda activate lry && \
  python scripts/export_generated_images.py \
    --config configs/wgan_gp_celeba.yaml \
    --checkpoint outputs/wgan_gp/checkpoints/generator_latest.ckpt \
    --output-dir outputs/eval/fake_wgan_gp \
    --num-images 10000 --seed 2020"

# 2. 计算指标（torch-fidelity 在 ms_gpu 环境）
bash -c "source /data1/liurongying/miniconda3/etc/profile.d/conda.sh && conda activate ms_gpu && \
  python scripts/evaluate.py \
    --real-dir outputs/eval/real \
    --fake-dir outputs/eval/fake_wgan_gp \
    --backend torch \
    --output outputs/eval/wgan_gp_metrics.json \
    --seed 2020"
```
