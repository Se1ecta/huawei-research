#!/bin/bash

set -e

SEED=42
MODEL="Qwen/Qwen2.5-0.5B"
DATASET="openwebtext-100k"

LR=1e-4
WD=0.01
BATCH=2
ACCUM=8
SEQ_LEN=512
EPOCHS=1
WARMUP=0.1

EXP_NAME="muon_lr${LR}_bs${BATCH}_seed${SEED}"
OUTPUT_DIR="experiments/${EXP_NAME}"

mkdir -p ${OUTPUT_DIR}

python train.py \
  --optimizer muon \
  --model ${MODEL} \
  --dataset ${DATASET} \
  --batch_size ${BATCH} \
  --grad_accum ${ACCUM} \
  --epochs ${EPOCHS} \
  --seq_len ${SEQ_LEN} \
  --lr ${LR} \
  --wd ${WD} \
  --warmup_ratio ${WARMUP} \
  --seed ${SEED} \
  --output_dir ${OUTPUT_DIR} \
  2>&1 | tee ${OUTPUT_DIR}/train.log