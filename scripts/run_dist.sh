#!/bin/bash
# Launch multi-GPU training with msrun
# Usage: bash scripts/run_dist.sh configs/wgan_gp_celeba.yaml [num_gpus] [start_gpu]

CONFIG=${1:-configs/wgan_gp_celeba.yaml}
NUM_GPUS=${2:-4}
START_GPU=${3:-0}

LOG=/data1/liurongying/workplace/gan_ms/train_dist.log
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# Build CUDA_VISIBLE_DEVICES
DEVICES=""
for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=$((START_GPU + i))
    DEVICES="${DEVICES}${GPU_ID}"
    [ $i -lt $((NUM_GPUS - 1)) ] && DEVICES="${DEVICES},"
done

echo "=== $(date) Launching ${NUM_GPUS}-GPU training ===" | tee -a $LOG
echo "Config: $CONFIG  GPUs: $DEVICES" | tee -a $LOG

cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES=$DEVICES \
conda run -n ms_gpu \
msrun --worker_num=$NUM_GPUS --local_worker_num=$NUM_GPUS --log_dir=./msrun_log \
    python scripts/train_dist.py --config "$CONFIG" >> $LOG 2>&1

echo "=== $(date) DONE exit=$? ===" | tee -a $LOG
