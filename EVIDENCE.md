# Evidence checklist

One row per Required-Evidence bullet: producing script -> result file(s) -> table/figure.

| Task | Evidence | Script | Result file(s) | Table / figure |
|---|---|---|---|---|
| 1 | Benchmark table, 7 conditions x 4 benchmarks, delta pp | `task1_finetune.evaluate` | `results/task1/<cond>_eval.json`, `<cond>_per_example_<bench>.jsonl` | `tab:t1_bench` |
| 1 | Combined SFT loss (5 runs) + L2-SP penalty and displacement | `task1_finetune.train` | `results/task1/<method>_train_log.jsonl`, `<method>_summary.json` | `t1_sft_loss.pdf`, `t1_l2sp_penalty_displacement.pdf`, `tab:t1_l2sp` |
| 1 | Resources (params, time, peak memory, checkpoint size) | `task1_finetune.train`, `merge_export` | `results/task1/<method>_summary.json`, `merge_check_<cond>.json` | `tab:t1_resources` |
| 1 | Diagnostics (rho_k, eta, math vs non-math R(z)) | `task1_finetune.diagnostics` | `results/task1/diagnostics_weights.json`, `diagnostics_activation*.json[l]` | `t1_diagnostics.pdf`, `tab:t1_diag` |
| 1 | Qualitative: 42-response annotation + aligned excerpts | `task1_finetune.qualitative` | `results/task1/qualitative_responses.jsonl`, `annotation_sheet.csv` | manual |
| 2 | Learning curves with stage boundaries | `task2_vlm.train` | `results/task2/<run>_train_log.jsonl`, `<run>_dev_log.jsonl` | `t2_learning_curves.pdf` |
| 2 | Final tables, both suites x 4 conditions, types, pair metrics, NLL, advantages, bootstrap CIs | `task2_vlm.evaluate_final` | `results/task2/<run>_final_eval.json`, `<run>_final_per_example.jsonl` | `tab:t2_grounding*`, `tab:t2_vqa*` |
| 2 | Aligned complementary-image qualitative examples | `task2_vlm.qualitative` | `results/task2/<run>_qualitative_examples.jsonl` | manual |
| 2 | Resources by stage incl. feature caching | `task2_vlm.cache_features`, `task2_vlm.train` | `results/task2/feature_cache_summary.json`, `<run>_summary.json` | `tab:t2_resources` |
| 3 | Accuracy / coverage curves, method comparison, sampling vs selection split | `task3_ttc.offline_eval` | `results/task3/<ds>_offline_eval.json`, `<ds>_per_question.parquet`, `<ds>_coverage.parquet` | `t3_<ds>_accuracy_coverage_cost.pdf`, `tab:t3_<ds>_methods`, `tab:t3_<ds>_fail` |
| 3 | Signal analysis (length/correctness, confidence-vision, tokenwise vs trajectory fusion) | `task3_ttc.signals`, `offline_eval` | `results/task3/signals_<ds>.json`, `signals_trajectories_<ds>.json` | manual / appendix |
| 3 | Live inference table (accuracy, tokens, runtime, truncation/extraction, transitions) | `task3_ttc.live_generation` | `results/task3/live_eval.json`, `live_per_example.jsonl` | `tab:t3_live` |
| 3 | Adaptive thresholds, held-out accuracy-cost, stopping distributions, fixed-N comparison | `task3_ttc.adaptive` | `results/task3/<ds>_adaptive.json`, `<ds>_adaptive_*_heldout.parquet` | `tab:t3_<ds>_adaptive`, cost panel of `t3_<ds>_*.pdf` |
| 3 | Qualitative selection / refinement / stopping examples | `offline_eval`, `live_generation`, `adaptive` | per-question parquet + live JSONL | manual |
