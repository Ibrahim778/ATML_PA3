#!/usr/bin/env bash
# Task 1: LLM fine-tuning. Steps whose outputs already exist are skipped; unfinished training runs
# resume from their newest checkpoint. FORCE=1 reruns every step from scratch (training discards
# outputs/task1/<method>; evaluation and qualitative ignore earlier outputs).
set -euo pipefail
C=configs/task1_ft.yaml
R=results/task1

skip() { [[ -z "${FORCE:-}" ]] && [[ -e "$1" ]]; }
run() { local marker=$1; shift; if skip "$marker"; then echo "[skip] $* ($marker exists)"; else "$@"; fi; }
trained() {  # summary exists and the run reached the final update
  [[ -z "${FORCE:-}" ]] && python - "$R/$1_summary.json" "$C" <<'PY'
import json, sys, yaml, os
p, c = sys.argv[1], sys.argv[2]
ok = os.path.exists(p) and json.load(open(p)).get("updates_done", 0) >= yaml.safe_load(open(c))["train"]["num_updates"]
sys.exit(0 if ok else 1)
PY
}
evaluated() {  # all four benchmarks present
  [[ -z "${FORCE:-}" ]] && python - "$R/$1_eval.json" <<'PY'
import json, sys, os
p = sys.argv[1]
ok = os.path.exists(p) and {"gsm8k", "hellaswag", "boolq", "arc_easy"} <= set(json.load(open(p)).get("benchmarks", {}))
sys.exit(0 if ok else 1)
PY
}

run ids/task1_probe_ids.json python -m scripts.prepare_data_task1 --config $C

# optional runtime estimate before the full runs
# python -m task1_finetune.train --config $C --method full --max-updates 5

for M in full l2sp lora4 lora16 oplora16; do
  if trained $M; then echo "[skip] train $M (finished)"; else python -m task1_finetune.train --config $C --method $M ${FORCE:+--fresh}; fi
done

TODO=()
for COND in lora4_1250 lora16_1250 oplora16_1250; do
  skip $R/merge_check_$COND.json || TODO+=($COND)
done
if (( ${#TODO[@]} )); then python -m task1_finetune.merge_export --config $C --conditions "${TODO[@]}"; else echo "[skip] merge_export (done)"; fi

for COND in baseline full_625 full_1250 l2sp_1250 lora4_1250 lora16_1250 oplora16_1250; do
  if evaluated $COND; then echo "[skip] evaluate $COND (done)"; else python -m task1_finetune.evaluate --config $C --condition $COND ${FORCE:+--fresh}; fi
done

run $R/diagnostics_activation.json python -m task1_finetune.diagnostics --config $C   # needs baseline GSM8K log
run $R/annotation_sheet.csv python -m task1_finetune.qualitative --config $C ${FORCE:+--fresh}           # needs data/first_plot_questions.yaml
