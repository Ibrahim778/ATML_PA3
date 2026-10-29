# ATML PA3: Fine-Tuning, VLM Grounding, Test-Time Compute

Code for EE-5102/CS-6304 Programming Assignment 3 (Fall 2026). Three tasks:

1. **LLM fine-tuning** (`task1_finetune/`): Qwen3-0.6B on MetaMathQA with Full FT, Full FT + L2-SP,
   LoRA-4, LoRA-16 and OPLoRA-16; lm-eval benchmarks, weight/activation diagnostics, qualitative.
2. **VLM grounding** (`task2_vlm/`): frozen SigLIP -> MLP connector -> Qwen3-0.6B (+LoRA), three-stage
   curriculum, complementary-pair evaluation under four image conditions.
3. **Test-time compute** (`task3_ttc/`): offline best-of-N / self-consistency / self-certainty /
   VisConf / RWC on the supplied rollouts, signal analysis, adaptive stopping, a live 50-question
   MATH experiment (single vs parallel vs refinement).

Global seed **6304** everywhere. All tunable settings live in `configs/*.yaml`; scripts take only
`--config` plus a selector (`--method`, `--condition`). The exact command order for each task is in
`run_task1.sh`, `run_task2.sh`, `run_task3.sh`. `EVIDENCE.md` maps every required-evidence bullet to
its script, result file and table/figure.

## Layout

```
configs/            base.yaml + one config per task (task2_vlm49.yaml = optional 49-token run)
common/             io/config merge, seeding + RNG state, chat formatting/label masking,
                    AMP training utilities and both loss normalisations, LoRA/OPLoRA, metrics
scripts/            prepare_data_task{1,2,3}.py, make_tables.py, make_figures.py
task1_finetune/     train, merge_export, evaluate (lm-eval), diagnostics, qualitative
task2_vlm/          cache_features, model, data, train, evaluate_dev, evaluate_final, qualitative
task3_ttc/          data, answers, selection, offline_eval, signals, adaptive, live_generation
ids/                committed selections: example/image/question/pair IDs, splits, orders, probes
results/task{1,2,3} committed machine-readable results (JSON/JSONL/CSV/parquet)
report/             tables.tex and figures/ generated from results/
data/, outputs/     raw data, caches, checkpoints (git-ignored)
```

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is not pinned (pinned wheels such as `torch==2.6.0` do not exist for newer Python
versions). The code is tested with torch 2.11-2.15, transformers 5.21, lm-eval 0.4.13 and math-verify 0.9;
it needs `transformers>=4.56`. Record the versions actually used for the reported runs with
`pip freeze > results/env.txt` and commit that file.
Device selection is automatic (`cuda` -> `mps` -> `cpu`); set `device:` in `configs/base.yaml` to force one.
Hardware used for the reported runs: _fill in (GPU model, VRAM)_.

## Data

| Task | Data | Where it goes |
|---|---|---|
| 1 | `meta-math/MetaMathQA`, `openai/gsm8k` (test, for decontamination), `Rowan/hellaswag` (probe IDs) | HF datasets cache (downloaded automatically) |
| 1 | `first_plot_questions.yaml` (course file) | `data/first_plot_questions.yaml` |
| 2 | COCO 2014 `train2014/`, `val2014/`, `annotations/` (captions + instances) | `data/coco/` |
| 2 | VQAv2 questions, annotations and complementary pairs for train2014/val2014 | `data/vqav2/` |
| 3 | ATML-PA3-Data (both model folders) | `data/ATML-PA3-Data/` |

Before running Task 3, read `data/ATML-PA3-Data/README.MD` and check `configs/task3_ttc.yaml`:
the folder names (`data.datasets.*.folder`), the parquet column names (`data.columns`), where the
MathVista metadata lives (`data.mathvista_metadata` if it is a separate file) and the exact
Qwen2.5-3B-Instruct revision for the live run (`live.revision`). The PDF asks for the *original*
model revision listed in that README, i.e. a specific commit hash of the Hugging Face repo; the
default `main` would silently use whatever the latest upload is.
`python -m scripts.prepare_data_task3 --inspect` prints the joined schema.

## Commands

### Task 1
```bash
python -m scripts.prepare_data_task1 --config configs/task1_ft.yaml
python -m task1_finetune.train --config configs/task1_ft.yaml --method full     # also: l2sp lora4 lora16 oplora16
python -m task1_finetune.merge_export --config configs/task1_ft.yaml
python -m task1_finetune.evaluate --config configs/task1_ft.yaml --condition baseline   # and the other 6 conditions
python -m task1_finetune.diagnostics --config configs/task1_ft.yaml
python -m task1_finetune.qualitative --config configs/task1_ft.yaml
```
**Resume:** re-run the same `train` command; the newest `outputs/task1/<method>/ckpt_<u>/resume.pt`
(model/adapters, optimizer, scheduler, GradScaler, RNG, data position) is loaded and the training log is
truncated to that update. `--max-updates k` stops after k updates (runtime estimate). Evaluation is
resumable per benchmark.

### Task 2
```bash
python -m scripts.prepare_data_task2 --config configs/task2_vlm.yaml
python -m task2_vlm.cache_features --config configs/task2_vlm.yaml
python -m task2_vlm.train --config configs/task2_vlm.yaml
python -m task2_vlm.evaluate_final --config configs/task2_vlm.yaml
python -m task2_vlm.qualitative --config configs/task2_vlm.yaml
```
**Resume:** re-run `task2_vlm.train`; checkpoints every 250 updates hold connector, adapters,
optimizer, scheduler, scaler, sampler state, stage progress and RNG. Feature caching is resumable too.
Optional 49-token run: same commands with `--config configs/task2_vlm49.yaml`.

### Task 3
```bash
python -m scripts.prepare_data_task3 --config configs/task3_ttc.yaml
python -m task3_ttc.offline_eval --config configs/task3_ttc.yaml
python -m task3_ttc.signals --config configs/task3_ttc.yaml
python -m task3_ttc.adaptive --config configs/task3_ttc.yaml
python -m task3_ttc.live_generation --config configs/task3_ttc.yaml     # GPU; resumable per question
```

### Running on Modal
`modal_app.py` runs each task on one Modal GPU: a single function call runs the steps of
`scripts/pipeline.py` (the Python version of `run_taskN.sh`) in order, calling each task's
`main(argv)` directly and skipping finished steps. State (`outputs/`, `results/`, `ids/`,
`data/`, HF cache) lives on the v2 Volume `atml-pa3-v2`; after preemption or a crash Modal
retries the call and it resumes from the last finished step or checkpoint.
```bash
pip install modal && modal setup
modal volume put atml-pa3-v2 data/first_plot_questions.yaml data/first_plot_questions.yaml
modal run --detach modal_app.py --task 1            # --force to start over, --gpu H100 etc.
modal volume get atml-pa3-v2 results results         # fetch results
```
Without Modal, `python -m scripts.pipeline --task N [--force]` runs the same steps in one process.

### Re-running and starting over
The `run_task*.sh` scripts skip every step whose output already exists, and unfinished training resumes
from its newest checkpoint. `FORCE=1 ./run_taskN.sh` reruns every step from scratch: training discards
that run's checkpoints (`outputs/task1/<method>`, `outputs/task2/<run>`), evaluation, qualitative and the
live run ignore earlier outputs, and feature caching rebuilds the cache. Do not use `FORCE=1` to resume
an interrupted run. The same behaviour is available per command with `--fresh`.

### Report tables and figures
```bash
python -m scripts.make_tables     # -> report/tables.tex
python -m scripts.make_figures    # -> report/figures/*.pdf
```

## Design choices and deviations

**Shared**
- Chat formatting renders the prompt with the tokenizer's template (`enable_thinking=False`), then appends
  response tokens and `<|im_end|>`; prompt and response are tokenised separately so the supervised span is
  exact. Labels cover the response and the end-of-turn token only.
- AMP: FP16 autocast + dynamic `GradScaler`; `unscale_` -> clip (1.0) -> step. A skipped update is detected
  when the scaler lowers its scale and is logged. The LR scheduler advances on skipped updates too, so the
  schedule length stays 1,250 / per-stage updates.
- Weight decay 0.01 on parameters with >= 2 dimensions that are not biases; none otherwise.
- Cosine schedule: linear warmup (update u uses (u+1)/warmup), then cosine to 0 (Task 1) or to 10% of
  peak (Task 2).
- Only supervised positions are passed through the LM head (avoids materialising full-vocabulary logits);
  CE is computed in FP32.

**Task 1**
- Loss: sum of token NLL per microbatch divided by the number of supervised tokens of the whole
  32-example effective batch (counted before the update), so the 16 microbatches sum to a token mean.
- GSM8K decontamination normalisation: NFKC, lower-case, non-alphanumerics -> space, collapsed whitespace;
  an example is excluded if `query` or `original_question` matches any normalised GSM8K test question.
- L2-SP: `Omega = 0.5 * sum (theta - theta0)^2` over unique parameters (`named_parameters()` de-duplicates
  the tied embedding), FP32, added once per update (on the last microbatch) as `lambda * Omega`.
  `theta0` is an FP32 on-device copy; its size and copy time are logged as overhead. Displacement
  `||theta - theta0||_2` is logged at every checkpoint for both full-FT runs.
- LoRA/OPLoRA are implemented in `common/lora.py` (not PEFT) so OPLoRA's factor projections are explicit.
  A: kaiming-uniform (as PEFT), B: zeros, factors FP32 on a frozen FP16 backbone. The same seed is set right
  before injection, so LoRA-16 and OPLoRA-16 start from identical unprojected factors. OPLoRA bases are the
  SVD of the FP16 backbone weights cast to FP32, computed once and stored as buffers; the projections act on
  the factors (`B - U_k(U_k^T B)`, `A - (A V_k) V_k^T`). Exports contain the projected factors.
- Checkpoints at 312/625/937/1250 keep evaluation weights (FP16 model for full FT; adapter factors
  otherwise) plus FP32 copies of the diagnosed `down_proj` weights; only the newest checkpoint keeps the
  (large) optimizer/resume state.
- `merge_export` merges into a fresh FP32 backbone, compares merged vs unmerged logits (FP32) and, for
  OPLoRA, verifies `||U_k^T B~||` and `||A~ V_k||` against freshly computed bases.
- `enable_thinking=False` for lm-eval is enforced by overriding `HFLM.apply_chat_template` and checked on the
  logged GSM8K prompts (they must contain the empty think block). GSM8K generation uses
  `max_gen_toks=1024`, greedy, stock stop sequences and filters. Truncation flag: response length >= 1,024 tokens.
- Activation probes use the exact GSM8K prompts from the baseline lm-eval log and the stock HellaSwag
  context (`activity_label: ctx_a ctx_b`), batch size 1 so the last token is the last prompt token.
- The qualitative annotation sheet pre-fills automatic flags (truncation, JSON validity, 4-gram
  repetition, math markers); the judgement columns are left for manual annotation.

**Task 2**
- Visual tokens are SigLIP `last_hidden_state` (post-LayerNorm applied once; pooling head unused) in
  row-major 14x14 order, cached in FP16.
- Prompt: user turn = `[visual tokens]\n{question}\nAnswer the question using a single word or number.`
  (captions: `Write a short caption for this image.`), chat template with thinking disabled. The no-image
  condition drops the visual tokens and rebuilds masks/positions; text is unchanged.
- **Connector scale calibration (training data only):** a fixed scalar
  `s = mean ||Qwen text-token embedding|| / mean ||connector_init(F)||`, measured on 256 training caption
  images and their captions/questions before Stage 1; stored as a buffer and recorded in
  `results/task2/<run>_calibration.json`.
- Loss: per-example mean token NLL, then mean over the 16-example effective batch.
- Each stage starts a fresh AdamW (connector and LoRA groups with their own peak LRs), 5% warmup, cosine
  to 10% of peak. Stage 1 walks two seeded shuffled passes over the 18k captions; Stages 2-3 draw components
  with the stated probabilities (each draw = one pair or two captions), no quotas.
- Data: val2014 is split 50/50 (seed 6304) into disjoint DEV and TEST image pools; every evaluation set draws
  only from its pool and uses each image once per suite. Grounding evaluation pairs are half presence, half
  counting (dev 12+12, test 32+32). Training pools are built from 8,000 train2014 images (all 6,000 caption
  images + 2,000 more) to bound the feature cache.
- COCO label noise: instances that are crowd annotations or cover < 0.5% of the image make that category
  ambiguous for that image; such (image, category) combinations are excluded from presence and counting.
  Presence questions use an article (`Does this image contain a dog?`); counting uses simple plurals.
- VQA consensus: answers normalised with the official VQA rules (punctuation, digits, articles,
  contractions); consensus = most frequent normalised answer with >= 8/10 agreement; target = the most
  frequent original string within the consensus group. Questions > 80 tokens and answers > 16 tokens are not
  selected (no truncation); captions > 64 tokens likewise.
- Mismatched images: a fixed seeded derangement over the pairs of a suite (example (p, side) takes the image
  of (perm[p], side)), saved to `ids/task2_mismatch_map.json`.

**Task 3**
- Answer groups never use gold: pairwise equivalence among a question's rollouts is precomputed
  (math-verify, symmetric; MathVista: equality after normalisation) and any revealed prefix is grouped
  greedily in reveal order. Missing extractions form one `NO_ANSWER` group (always wrong).
- MathVista normalisation follows MathVista's `normalize_extracted_answer` (integer, float with precision,
  list/text), except multiple-choice answers are mapped to option letters (letter, `(X)` pattern, exact option
  text, else the closest option by Levenshtein distance). Gold answers go through the same rules.
- Scorer versions are written into every Task 3 result JSON.
- Rollout orders: `default_rng(6304)`, 10 permutations per question, questions in lexicographic order
  (`ids/task3_orders_<ds>.json`). Ties: earliest supporting rollout in the current order; RWC uses average
  ranks. VisConf calibration (Q.05/Q.95, linear interpolation) is recomputed on every prefix.
- FirstAns costs only its first candidate's tokens; all other methods cost the sum over the prefix.
- Adaptive stopping: SC state and RWC state are computed at n = 4, 8, 16, 32 from revealed candidates only;
  selection on dev follows the stated rule (always-32 included as fallback), then the setting is frozen for
  held-out.
- Live run: per-call seeds are a hash of (6304, question, strategy, call index). The termination token is
  not counted in output tokens; a call is truncated if it hits its cap without emitting EOS.

## Assumptions to verify

- Task 3 parquet schemas and folder names (see Data above); the code fails loudly on a join mismatch.
- `first_plot_questions.yaml` structure: the loader accepts either entries with `plain/json/template`
  keys or flat entries with `id/format/prompt`; it asserts 24 prompts.
- lm-eval 0.4.8 sample-log fields (`resps`, `filtered_resps`, `filter`, `arguments`) are parsed defensively;
  check `results/task1/baseline_eval.json -> example_prompt` after the first run.

## Testing done

The pipelines were smoke-tested on CPU with a 2-layer random Qwen3 model and synthetic data: all five
Task 1 methods (including exact resume and merged-vs-unmerged logits; OPLoRA rho_k = 0 by construction),
the full Task 2 curriculum with resume at a stage boundary and mid-stage (bit-identical weights), final
evaluation and qualitative export, and the complete offline Task 3 pipeline. lm-eval evaluation, feature
caching on real images and live generation were not run in that test.

## Attribution

- VQA answer normalisation (`task2_vlm/vqa_norm.py`) follows the official VQA evaluation code
  (GT-Vision-Lab/VQA, `vqaEval.py`).
- MathVista answer normalisation (`task3_ttc/answers.py`) follows `normalize_extracted_answer` from the
  MathVista repository (lupantech/MathVista).
- Math equivalence uses `math-verify` (Hugging Face). Benchmarks use EleutherAI `lm-evaluation-harness`.
- LoRA initialisation mirrors PEFT's default (kaiming-uniform A, zero B).
- Code was written with assistance from an AI coding assistant (allowed by the course policy for code);
  the report text is the student's own.
