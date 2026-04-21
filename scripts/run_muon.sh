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
#!/bin/bash

SCRIPT_PATH="/content/huawei-research/src/train.py"

python "$SCRIPT_PATH" \
  --model_name Qwen/Qwen2.5-0.5B \
  --optimizer muon \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-3 \
  --seq_length 512 \
  --num_train_epochs 1 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.01 \
  --weight_decay 0.05 \
  --logging_steps 10 \
  --push_to_hub False \
  --report_to clearml \
  --seed 42 \
  --output_dir ./Qwen2.5-0.5B_muon