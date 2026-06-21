#!/bin/bash
LOG=/data1/liurongying/workplace/gan_ms/train_all.log
cd /data1/liurongying/workplace/gan_ms

echo "=== $(date) DCGAN START ===" | tee -a $LOG
conda run -n ms_gpu python scripts/train.py --config configs/dcgan_celeba.yaml >> $LOG 2>&1
echo "=== $(date) DCGAN DONE exit=$? ===" | tee -a $LOG

echo "=== $(date) WGAN-GP START ===" | tee -a $LOG
conda run -n ms_gpu python scripts/train.py --config configs/wgan_gp_celeba.yaml >> $LOG 2>&1
echo "=== $(date) WGAN-GP DONE exit=$? ===" | tee -a $LOG
