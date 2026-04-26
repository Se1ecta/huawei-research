#!/bin/bash

SCRIPT_PATH="src/train.py"

python "$SCRIPT_PATH" \
  --model_name Qwen/Qwen2.5-0.5B \
  --optimizer mezo \
  --zo_eps 1e-3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --seq_length 512 \
  --num_train_epochs 1 \
  --lr_scheduler_type cosine \
  --warmup_steps 0.01 \
  --weight_decay 0 \
  --logging_steps 10 \
  --logging_strategy="steps" \
  --push_to_hub True \
  --report_to clearml \
  --seed 42 \
  --output_dir ./Qwen2.5-0.5B_mezo