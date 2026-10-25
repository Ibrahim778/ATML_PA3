#!/usr/bin/env bash
# Task 1: LLM fine-tuning. Every training command is resumable (re-run the same line).
set -euo pipefail
C=configs/task1_ft.yaml

python -m scripts.prepare_data_task1 --config $C            # ids/task1_train_ids.json, ids/task1_probe_ids.json

# optional runtime estimate before the full runs
# python -m task1_finetune.train --config $C --method full --max-updates 5

for M in full l2sp lora4 lora16 oplora16; do
  python -m task1_finetune.train --config $C --method $M
done

python -m task1_finetune.merge_export --config $C           # LoRA/OPLoRA -> merged FP16 models + logit check

for COND in baseline full_625 full_1250 l2sp_1250 lora4_1250 lora16_1250 oplora16_1250; do
  python -m task1_finetune.evaluate --config $C --condition $COND
done

python -m task1_finetune.diagnostics --config $C            # needs baseline GSM8K log for exact prompts
python -m task1_finetune.qualitative --config $C            # needs data/first_plot_questions.yaml
