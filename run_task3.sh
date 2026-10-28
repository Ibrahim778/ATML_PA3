#!/usr/bin/env bash
# Task 3: test-time compute. Steps whose outputs already exist are skipped; FORCE=1 reruns every
# step from scratch (the live run discards its per-question log).
# Offline steps run on CPU; the live step needs a GPU and resumes per question.
set -euo pipefail
C=configs/task3_ttc.yaml
R=results/task3

skip() { [[ -z "${FORCE:-}" ]] && [[ -e "$1" ]]; }
run() { local marker=$1; shift; if skip "$marker"; then echo "[skip] $* ($marker exists)"; else "$@"; fi; }

# check column names against ATML-PA3-Data/README.MD first:
#   python -m scripts.prepare_data_task3 --config $C --inspect
run $R/data_report.json python -m scripts.prepare_data_task3 --config $C   # splits, orders, answer groups
run $R/mathvista_offline_eval.json python -m task3_ttc.offline_eval --config $C   # Steps 1-2
run $R/signals_mathvista.json python -m task3_ttc.signals --config $C           # Step 3
run $R/mathvista_adaptive.json python -m task3_ttc.adaptive --config $C         # Step 5
run $R/live_eval.json python -m task3_ttc.live_generation --config $C ${FORCE:+--fresh}           # Step 4 (GPU)
