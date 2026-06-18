# Dual GAN Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible FID and Inception Score evaluation with standard torch-fidelity and native MindSpore/MindCV backends.

**Architecture:** Framework-independent NumPy formulas sit below two optional feature-extraction backends. Separate preparation, generation, and evaluation CLIs keep expensive stages reproducible and restartable.

**Tech Stack:** Python, NumPy, SciPy, Pillow, MindSpore, MindCV, PyTorch, torch-fidelity, pytest.

---

### Task 1: Metric Core

**Files:**
- Create: `tests/test_metrics_core.py`
- Create: `src/metrics/core.py`
- Modify: `src/metrics/__init__.py`
- Delete: `src/metrics/placeholders.py`

- [ ] Write tests asserting zero FID for equal features, positive FID for shifted features, IS=1 for uniform logits, and IS>1 for confident diverse logits.
- [ ] Run `pytest tests/test_metrics_core.py -q` and verify the missing core module fails.
- [ ] Implement `activation_statistics`, `frechet_distance`, and `inception_score_from_logits` with finite-value and shape validation.
- [ ] Re-run the tests and commit the core implementation.

### Task 2: Optional Evaluation Backends

**Files:**
- Create: `src/metrics/torch_fidelity_backend.py`
- Create: `src/metrics/mindspore_backend.py`
- Create: `tests/test_metrics_backends.py`

- [ ] Test path validation and normalized backend result keys without requiring GPU frameworks.
- [ ] Implement lazy `torch_fidelity.calculate_metrics` integration returning `fid`, `is_mean`, and `is_std`.
- [ ] Implement MindCV InceptionV3 preprocessing, pooled feature/logit extraction, and shared-formula evaluation.
- [ ] Run backend tests and commit both backends.

### Task 3: Reproducible Image Preparation

**Files:**
- Create: `src/metrics/image_io.py`
- Create: `scripts/prepare_real_images.py`
- Create: `scripts/export_generated_images.py`
- Create: `tests/test_metric_image_io.py`

- [ ] Test deterministic image discovery, resizing, and generated tensor conversion.
- [ ] Implement real CelebA cache export and batched checkpoint generation using a fixed seed.
- [ ] Run image I/O tests and commit the preparation scripts.

### Task 4: Unified CLI and Dependencies

**Files:**
- Create: `scripts/evaluate.py`
- Create: `requirements-eval-torch.txt`
- Create: `requirements-eval-mindspore.txt`
- Modify: `README.md`

- [ ] Implement `--backend torch|mindspore|both`, JSON output, batch size, device, splits, seed, and maximum sample arguments.
- [ ] Add separate dependency files so training and evaluation environments do not conflict.
- [ ] Document complete 10k debug and 50k final commands.
- [ ] Run CLI help and syntax checks, then commit.

### Task 5: Attribution and Final Verification

**Files:**
- Create: `docs/METRICS_SOURCES.md`

- [ ] Document the FID and IS papers, TTUR, pytorch-fid, torch-fidelity, and MindCV links, licenses, formulas, and adopted behavior.
- [ ] Add report-ready BibTeX and a warning that the two Inception variants are different protocols.
- [ ] Run all locally available tests, inspect git status, commit, and push `main` to GitHub.
