#!/bin/bash
LOG=/data1/liurongying/workplace/gan_ms/setup_train.log
echo "=== $(date) START ===" >> $LOG

# Step 1: install CUDA 11 libs
echo "--- Installing CUDA 11 libs ---" >> $LOG
/data1/liurongying/miniconda3/envs/ms_gpu/bin/pip install \
  nvidia-cuda-runtime-cu11==11.8.89 \
  nvidia-cublas-cu11 \
  nvidia-cudnn-cu11 \
  nvidia-cufft-cu11 \
  nvidia-cusparse-cu11 \
  nvidia-curand-cu11 \
  nvidia-cusolver-cu11 >> $LOG 2>&1
echo "--- CUDA libs install done, exit=$? ---" >> $LOG

# Step 2: install other deps
echo "--- Installing project deps ---" >> $LOG
/data1/liurongying/miniconda3/envs/ms_gpu/bin/pip install \
  pyyaml pillow tqdm >> $LOG 2>&1

# Step 3: verify GPU, fall back to CPU if needed
echo "--- Checking GPU support ---" >> $LOG
GPU_OK=$(/data1/liurongying/miniconda3/envs/ms_gpu/bin/python -c "
import mindspore as ms
try:
    ms.set_context(mode=ms.GRAPH_MODE, device_target='GPU', device_id=0)
    print('GPU')
except Exception as e:
    print('CPU')
" 2>/dev/null)
echo "Device: $GPU_OK" >> $LOG

# update configs based on detected device
if [ "$GPU_OK" = "GPU" ]; then
  sed -i 's/device_target: CPU/device_target: GPU/' /data1/liurongying/workplace/gan_ms/configs/dcgan_celeba.yaml
  sed -i 's/device_target: CPU/device_target: GPU/' /data1/liurongying/workplace/gan_ms/configs/wgan_gp_celeba.yaml
fi

# Step 4: train DCGAN
echo "--- Starting DCGAN training ---" >> $LOG
cd /data1/liurongying/workplace/gan_ms
/data1/liurongying/miniconda3/envs/ms_gpu/bin/python scripts/train.py \
  --config configs/dcgan_celeba.yaml >> $LOG 2>&1
echo "=== $(date) DCGAN DONE, exit=$? ===" >> $LOG

# Step 5: train WGAN-GP
echo "--- Starting WGAN-GP training ---" >> $LOG
/data1/liurongying/miniconda3/envs/ms_gpu/bin/python scripts/train.py \
  --config configs/wgan_gp_celeba.yaml >> $LOG 2>&1
echo "=== $(date) ALL DONE ===" >> $LOG
