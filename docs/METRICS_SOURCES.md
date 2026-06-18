# FID and Inception Score Sources

本文档记录 `gan_ms` 评测代码的理论来源、参考实现、许可证和报告写法。评测实现于 2026-06-18 核对了下列公开源码。

## 1. 指标论文

### Frechet Inception Distance

FID由 Heusel 等人在 TTUR 论文中提出。它将真实图像和生成图像输入 Inception 网络，分别拟合特征的多元高斯分布，再计算两组均值和协方差之间的 Frechet 距离：

```text
FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2(Sigma_r Sigma_g)^(1/2))
```

- 论文：Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*, NeurIPS 2017.
- arXiv：https://arxiv.org/abs/1706.08500
- NeurIPS：https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html

### Inception Score

IS由 Salimans 等人提出。它计算每个生成样本的条件类别分布 `p(y|x)` 与所有生成样本边缘类别分布 `p(y)` 的 KL 散度，并对平均KL取指数：

```text
IS = exp(E_x[KL(p(y|x) || p(y))])
```

- 论文：Salimans et al., *Improved Techniques for Training GANs*, NeurIPS 2016.
- arXiv：https://arxiv.org/abs/1606.03498
- NeurIPS：https://proceedings.neurips.cc/paper/2016/hash/8a3363abe792db2d8761d6403605aeb7-Abstract.html

### InceptionV3

- 论文：Szegedy et al., *Rethinking the Inception Architecture for Computer Vision*, CVPR 2016.
- arXiv：https://arxiv.org/abs/1512.00567

## 2. 参考实现

### TTUR

- 仓库：https://github.com/bioinf-jku/TTUR
- 作用：FID论文作者提供的原始TensorFlow实现。
- 本项目没有复制其源码；其公式和评测协议作为理论与历史基准。

### pytorch-fid

- 仓库：https://github.com/mseitzer/pytorch-fid
- 核对提交：`b9c18118d082cbd263c1b8963fc4221dc1cbb659`
- 关键文件：`src/pytorch_fid/fid_score.py`、`src/pytorch_fid/inception.py`
- 作用：核对2048维pool特征、均值/协方差统计和FID数值处理。
- 许可证：Apache License 2.0。

### torch-fidelity

- 仓库：https://github.com/toshas/torch-fidelity
- 核对提交：`5e211a950a7b45206bd4976813ffd6aed6cf4ccc`
- 关键文件：`torch_fidelity/metric_fid.py`、`metric_isc.py`、`feature_extractor_inceptionv3.py`。
- 作用：本项目标准后端通过公开API `torch_fidelity.calculate_metrics` 直接调用该库，同时计算FID和10折IS。
- 特征协议：TensorFlow兼容InceptionV3；FID使用2048维特征，IS使用1000类logits。
- 许可证：Apache License 2.0。
- DOI：https://doi.org/10.5281/zenodo.3786539

### MindCV

- 仓库：https://gitee.com/mindspore-lab/mindcv
- 核对提交：`51b0636da1e38a44a52edf76cf314d1c7c18883a`
- 关键文件：`mindcv/models/inceptionv3.py`
- 预训练权重：`https://download.mindspore.cn/toolkits/mindcv/inception_v3/inception_v3-38f67890.ckpt`
- 作用：MindSpore原生后端调用 `mindcv.create_model("inception_v3", pretrained=True)`；对 `forward_features()` 输出做全局平均池化，得到2048维特征，并通过分类头取得1000类logits。
- 许可证：Apache License 2.0。

## 3. 本项目采用和重写的范围

| 文件 | 作用 | 来源说明 |
|---|---|---|
| `src/metrics/core.py` | NumPy实现均值、协方差、FID和分块IS公式 | 根据论文公式独立重写；数值处理参考 `pytorch-fid` 和 `torch-fidelity` |
| `src/metrics/torch_fidelity_backend.py` | 标准FID/IS | 不复制第三方算法代码，调用 `torch-fidelity` 公共API |
| `src/metrics/mindspore_backend.py` | MindSpore/MindCV版FID/IS | 使用MindCV公开模型接口，统计部分调用本项目共享公式 |
| `scripts/prepare_real_images.py` | 导出真实64x64评测图 | 本项目编写，与训练预处理保持一致 |
| `scripts/export_generated_images.py` | 从checkpoint导出独立生成图 | 本项目编写 |
| `scripts/evaluate.py` | 双后端调度和JSON记录 | 本项目编写 |

本仓库没有复制或打包 `pytorch-fid`、`torch-fidelity`、TTUR或MindCV源码。它们作为依赖或算法参考使用，原仓库许可证仍分别适用。

## 4. 两个后端不能混用数值口径

报告主表建议采用 `torch-fidelity` 结果，因为它面向生成模型指标兼容性进行了专门实现。MindCV版本用于验证MindSpore生态下的完整评测流程。

两者结果可能不同，原因包括：

1. InceptionV3权重不同：标准后端采用TensorFlow兼容权重，MindCV采用其ImageNet预训练checkpoint。
2. 图像缩放和归一化细节不同。
3. 网络实现细节和浮点计算后端不同。

因此表格列名应写成：

```text
FID (torch-fidelity) ↓
IS (torch-fidelity) ↑
FID (MindCV) ↓
IS (MindCV) ↑
```

不要用一个后端计算DCGAN、另一个后端计算WGAN-GP后直接比较。

## 5. 实验协议建议

- 调试：每个模型生成10,000张图片。
- 最终报告：条件允许时生成50,000张图片。
- DCGAN与WGAN-GP必须使用相同真实图缓存、生成样本数、IS分块数和随机种子。
- 保存 `scripts/evaluate.py` 输出的JSON，其中包含样本数、batch size、分块数、随机种子和后端名称。
- FID越低表示生成分布更接近真实分布；IS越高通常表示样本更清晰且更多样。

CelebA是人脸数据，而IS依赖ImageNet类别分类器，因此IS的类别语义并不完全适合人脸。报告中应把IS用于同一数据集、同一后端下的相对比较，并结合FID和生成样例共同分析。

## 6. 报告可直接采用的描述

> 本实验使用FID和Inception Score评价生成质量。标准结果通过torch-fidelity计算，该实现采用TensorFlow兼容的InceptionV3特征提取协议；同时使用MindCV预训练InceptionV3实现MindSpore原生复算。由于两种后端的预训练权重和预处理存在差异，本文分别报告两组结果，不进行跨后端数值比较。DCGAN与WGAN-GP采用相同真实样本、生成样本数、随机种子和10折IS设置，以保证模型对比公平。

## 7. BibTeX

```bibtex
@inproceedings{heusel2017gans,
  title={GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium},
  author={Heusel, Martin and Ramsauer, Hubert and Unterthiner, Thomas and Nessler, Bernhard and Hochreiter, Sepp},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}

@inproceedings{salimans2016improved,
  title={Improved Techniques for Training GANs},
  author={Salimans, Tim and Goodfellow, Ian and Zaremba, Wojciech and Cheung, Vicki and Radford, Alec and Chen, Xi},
  booktitle={Advances in Neural Information Processing Systems},
  year={2016}
}

@inproceedings{szegedy2016rethinking,
  title={Rethinking the Inception Architecture for Computer Vision},
  author={Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jon and Wojna, Zbigniew},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2016}
}

@misc{obukhov2020torchfidelity,
  author={Anton Obukhov and Maximilian Seitzer and Po-Wei Wu and Semen Zhydenko and Jonathan Kyl and Elvis Yu-Jing Lin},
  title={High-fidelity Performance Metrics for Generative Models in PyTorch},
  year={2020},
  publisher={Zenodo},
  doi={10.5281/zenodo.3786539},
  url={https://github.com/toshas/torch-fidelity}
}
```
