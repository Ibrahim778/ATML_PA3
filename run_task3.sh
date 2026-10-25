#!/usr/bin/env bash
# Task 3: test-time compute. Offline steps run on CPU; the live step needs a GPU.
set -euo pipefail
C=configs/task3_ttc.yaml

python -m scripts.prepare_data_task3 --config $C --inspect  # check column names against README.MD
python -m scripts.prepare_data_task3 --config $C            # joins, alignment, answer groups, splits, orders
python -m task3_ttc.offline_eval --config $C                # Steps 1-2
python -m task3_ttc.signals --config $C                     # Step 3
python -m task3_ttc.adaptive --config $C                    # Step 5
python -m task3_ttc.live_generation --config $C             # Step 4 (GPU, resumable)
