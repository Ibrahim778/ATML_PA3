#!/usr/bin/env bash
# Task 2: SigLIP -> Qwen3 VLM. Training is resumable (re-run the same line).
set -euo pipefail
C=configs/task2_vlm.yaml

python -m scripts.prepare_data_task2 --config $C            # ids/task2_*.json
python -m task2_vlm.cache_features --config $C              # FP16 SigLIP features, encoder freed afterwards
python -m task2_vlm.train --config $C                       # stages 1-3, dev eval at 0/500/.../2250/3750/4500
python -m task2_vlm.evaluate_final --config $C              # 4 image conditions, bootstrap CIs
python -m task2_vlm.qualitative --config $C

# optional 49-token variant: configs/task2_vlm49.yaml sets run_name vlm49 and pool_2x2 true
# python -m task2_vlm.train --config configs/task2_vlm49.yaml   (and the two eval commands)
