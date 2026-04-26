

SCRIPT_PATH="src/train.py"

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