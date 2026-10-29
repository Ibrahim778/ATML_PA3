"""Python version of run_task{1,2,3}.sh: the ordered steps of each task, the "already done" test
for each step, and a local runner that calls every step's main(argv) in this process.

Steps are grouped into stages; steps within a stage are independent (they may run in parallel,
e.g. on Modal, see modal_app.py). A step is skipped when its output already exists, unless
force=True, which also passes --fresh to the steps that would otherwise resume.

python -m scripts.pipeline --task 1 [--force]
"""
import argparse
import importlib
import json
import os
from dataclasses import dataclass, field

from common.io import load_yaml, repo_path

C1, C2, C3 = "configs/task1_ft.yaml", "configs/task2_vlm.yaml", "configs/task3_ttc.yaml"
R1, R2, R3 = "results/task1", "results/task2", "results/task3"
METHODS = ["full", "l2sp", "lora4", "lora16", "oplora16"]
CONDITIONS = ["baseline", "full_625", "full_1250", "l2sp_1250", "lora4_1250", "lora16_1250", "oplora16_1250"]
ADAPTER_CONDS = ["lora4_1250", "lora16_1250", "oplora16_1250"]
BENCHES = {"gsm8k", "hellaswag", "boolq", "arc_easy"}


@dataclass
class Step:
    name: str
    module: str                  # module with main(argv), or "module:function" called with no args
    argv: list = field(default_factory=list)
    done: object = None          # callable -> bool
    gpu: bool = False
    fresh: bool = False          # accepts --fresh (resumable step)

    def args(self, force):
        return self.argv + (["--fresh"] if force and self.fresh else [])


def exists(path):
    return lambda: os.path.exists(repo_path(path))


def _json(path):
    p = repo_path(path)
    return json.load(open(p)) if os.path.exists(p) else None


def t1_trained(m):
    """Finished = summary reached the last update AND the final weights are still on disk
    (results/ can be restored from git while outputs/ is gone)."""
    def f():
        s = _json(f"{R1}/{m}_summary.json")
        n = load_yaml(C1)["train"]["num_updates"]
        final = f"outputs/task1/{m}/ckpt_{n}/" + ("model" if m in ("full", "l2sp") else "adapter.safetensors")
        return bool(s) and s.get("updates_done", 0) >= n and os.path.exists(repo_path(final))
    return f


def t1_evaluated(c):
    def f():
        s = _json(f"{R1}/{c}_eval.json")
        return bool(s) and BENCHES <= set(s.get("benchmarks", {}))
    return f


def stages(task, cfg2=C2):
    """List of stages (each a list of Steps) for one task."""
    if task == 1:
        return [
            [Step("prepare", "scripts.prepare_data_task1", ["--config", C1], exists("ids/task1_probe_ids.json"))],
            [Step("tokenize", "task1_finetune.train:build_token_cache", [], lambda: False)],
            [Step(f"train_{m}", "task1_finetune.train", ["--config", C1, "--method", m], t1_trained(m), gpu=True, fresh=True)
             for m in METHODS],
            [Step(f"merge_{c}", "task1_finetune.merge_export", ["--config", C1, "--conditions", c],
                  lambda c=c: exists(f"{R1}/merge_check_{c}.json")() and exists(f"outputs/task1/merged/{c}")())
             for c in ADAPTER_CONDS],
            [Step("eval_data", "task1_finetune.evaluate:warm_datasets", [], lambda: False)],
            [Step(f"eval_{c}", "task1_finetune.evaluate", ["--config", C1, "--condition", c], t1_evaluated(c),
                  gpu=True, fresh=True) for c in CONDITIONS],
            [Step("diagnostics", "task1_finetune.diagnostics", ["--config", C1], exists(f"{R1}/diagnostics_activation.json"), gpu=True),
             Step("qualitative", "task1_finetune.qualitative", ["--config", C1], exists(f"{R1}/annotation_sheet.csv"), gpu=True, fresh=True)],
        ]
    if task == 2:
        run = load_yaml(cfg2)["run_name"]
        return [
            [Step("prepare", "scripts.prepare_data_task2", ["--config", cfg2], exists("ids/task2_mismatch_map.json"))],
            [Step("cache_features", "task2_vlm.cache_features", ["--config", cfg2], exists(f"{R2}/feature_cache_summary.json"), gpu=True, fresh=True)],
            [Step("train", "task2_vlm.train", ["--config", cfg2], exists(f"outputs/task2/{run}/stage3_vqa/connector.safetensors"), gpu=True, fresh=True)],
            [Step("evaluate_final", "task2_vlm.evaluate_final", ["--config", cfg2], exists(f"{R2}/{run}_final_eval.json"), gpu=True)],
            [Step("qualitative", "task2_vlm.qualitative", ["--config", cfg2], exists(f"{R2}/{run}_qualitative_examples.jsonl"))],
        ]
    if task == 3:
        return [
            [Step("prepare", "scripts.prepare_data_task3", ["--config", C3], exists(f"{R3}/data_report.json"))],
            [Step("offline_eval", "task3_ttc.offline_eval", ["--config", C3], exists(f"{R3}/mathvista_offline_eval.json"))],
            [Step("signals", "task3_ttc.signals", ["--config", C3], exists(f"{R3}/signals_mathvista.json")),
             Step("adaptive", "task3_ttc.adaptive", ["--config", C3], exists(f"{R3}/mathvista_adaptive.json")),
             Step("live", "task3_ttc.live_generation", ["--config", C3], exists(f"{R3}/live_eval.json"), gpu=True, fresh=True)],
        ]
    raise ValueError(task)


def call(module, argv):
    """Import a step and run it in this process (no subprocess)."""
    mod, _, fn = module.partition(":")
    m = importlib.import_module(mod)
    return getattr(m, fn)() if fn else m.main(argv)


def pending(task, force, cfg2=C2):
    """Yield (stage_index, [steps to run]) stage by stage; re-evaluates `done` lazily."""
    for i, st in enumerate(stages(task, cfg2)):
        todo = [s for s in st if force or not s.done()]
        for s in st:
            if s not in todo:
                print(f"[skip] {s.name} (done)", flush=True)
        yield i, todo


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--force", action="store_true", help="rerun every step from scratch")
    ap.add_argument("--config2", default=C2, help="Task 2 config (configs/task2_vlm49.yaml for the 49-token run)")
    args = ap.parse_args(argv)
    os.chdir(repo_path("."))
    for _, todo in pending(args.task, args.force, args.config2):
        for s in todo:
            print(f"[run] {s.name}: {s.module} {' '.join(s.args(args.force))}", flush=True)
            call(s.module, s.args(args.force))


if __name__ == "__main__":
    main()
