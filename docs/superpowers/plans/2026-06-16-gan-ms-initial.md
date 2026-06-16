# gan_ms Initial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first runnable MindSpore DCGAN/WGAN-GP project for CelebA image generation.

**Architecture:** A compact Python package under `src/` separates config loading, data loading, models, trainers, and utilities. CLI scripts in `scripts/` call into the package with YAML configs.

**Tech Stack:** Python, MindSpore, NumPy, Pillow, PyYAML, pytest.

---

### Task 1: Project Metadata

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/models/__init__.py`
- Create: `src/trainers/__init__.py`
- Create: `src/utils/__init__.py`

- [ ] Add project metadata and ignored runtime outputs.
- [ ] Initialize git and commit the metadata files.

### Task 2: Config System

**Files:**
- Create: `configs/dcgan_celeba.yaml`
- Create: `configs/wgan_gp_celeba.yaml`
- Create: `src/config.py`
- Create: `tests/test_config.py`

- [ ] Add typed config dataclasses and YAML loading.
- [ ] Add DCGAN and WGAN-GP default configs for 64x64 CelebA.
- [ ] Test config parsing.
- [ ] Commit config files and tests.

### Task 3: Data and Image Utilities

**Files:**
- Create: `src/data/celeba.py`
- Create: `src/utils/images.py`
- Create: `tests/test_images.py`

- [ ] Implement MindSpore image-folder dataset creation.
- [ ] Implement generated image grid saving.
- [ ] Test grid conversion without MindSpore dataset dependencies.
- [ ] Commit data and image utilities.

### Task 4: Models

**Files:**
- Create: `src/models/dcgan.py`
- Create: `tests/test_models.py`

- [ ] Implement DCGAN generator.
- [ ] Implement discriminator usable as DCGAN discriminator or WGAN critic.
- [ ] Test output shapes.
- [ ] Commit model code.

### Task 5: Trainers and Scripts

**Files:**
- Create: `src/utils/checkpoint.py`
- Create: `src/trainers/dcgan_trainer.py`
- Create: `src/trainers/wgan_gp_trainer.py`
- Create: `scripts/train.py`
- Create: `scripts/generate.py`

- [ ] Implement DCGAN training loop.
- [ ] Implement WGAN-GP training loop.
- [ ] Implement train and generate CLIs.
- [ ] Commit trainer and script code.

### Task 6: Verification

**Files:**
- Modify: `README.md`

- [ ] Run lightweight tests.
- [ ] Run import checks if MindSpore is unavailable.
- [ ] Document exact commands for training and generation.
- [ ] Commit documentation updates.
