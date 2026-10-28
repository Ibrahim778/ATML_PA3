#!/usr/bin/env bash
# Task 2: SigLIP -> Qwen3 VLM. Steps whose outputs already exist are skipped; FORCE=1 reruns everything.
# Unfinished training resumes from its newest checkpoint.
set -euo pipefail
C=${CONFIG:-configs/task2_vlm.yaml}      # CONFIG=configs/task2_vlm49.yaml for the optional 49-token run
RUN=$(python -c "import sys; sys.path.insert(0, '.'); from common.io import load_yaml; print(load_yaml('$C')['run_name'])")
R=results/task2

skip() { [[ -z "${FORCE:-}" ]] && [[ -e "$1" ]]; }
run() { local marker=$1; shift; if skip "$marker"; then echo "[skip] $* ($marker exists)"; else "$@"; fi; }

run ids/task2_mismatch_map.json python -m scripts.prepare_data_task2 --config $C
run $R/feature_cache_summary.json python -m task2_vlm.cache_features --config $C
run outputs/task2/$RUN/stage3_vqa/connector.safetensors python -m task2_vlm.train --config $C
run $R/${RUN}_final_eval.json python -m task2_vlm.evaluate_final --config $C
run $R/${RUN}_qualitative_examples.jsonl python -m task2_vlm.qualitative --config $C
