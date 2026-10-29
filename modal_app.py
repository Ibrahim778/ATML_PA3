"""Run the PA3 pipelines natively on Modal (https://modal.com).

Structure (following Modal's long-training example):
  * every pipeline step (scripts/pipeline.py) runs as its own Modal function call that imports
    the task module and calls its main(argv) directly: `gpu_step` on a GPU, `cpu_step` on CPU;
  * steps in the same stage run in parallel (e.g. the five Task 1 trainings, the seven evals);
  * `orchestrate` (CPU, non-preemptible) walks the stages, skips finished steps and waits for
    each stage before starting the next; the local entrypoint only spawns it, so `--detach`
    keeps the whole pipeline alive after the terminal closes;
  * all state (outputs/, results/, ids/, data/, Hugging Face cache) lives on a v2 Volume;
    step functions retry automatically after preemption or a crash and resume from the
    checkpoints on the Volume (the training scripts checkpoint and resume on their own).

    pip install modal && modal setup                                   # once
    modal volume put atml-pa3-v2 data/first_plot_questions.yaml data/first_plot_questions.yaml
    modal run --detach modal_app.py --task 1                           # --force to start over
    modal run --detach modal_app.py --task 2 --gpu H100
    modal volume get atml-pa3-v2 results results                       # fetch results

Follow progress on the Modal dashboard (one row per step) or `modal app logs atml-pa3`.
"""
import os

import modal

APP = "atml-pa3"
VOL = "/vol"
REPO = "/root/ATML_PA3"
SEED = "/root/seed"                              # committed ids/ and results/ from the repo
STATE = ("outputs", "results", "ids", "data")    # live on the volume, symlinked into the repo
DEFAULT_GPU = "A100-80GB"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .env({"HF_HOME": f"{VOL}/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(".", REPO, ignore=["data", "outputs", "results", "ids", "logs", ".git", ".venv",
                                      "venv", "**/__pycache__", "*.html"])
    .add_local_dir("ids", f"{SEED}/ids")
    .add_local_dir("results", f"{SEED}/results")
)
vol = modal.Volume.from_name("atml-pa3-v2", create_if_missing=True, version=2)
app = modal.App(APP, image=image)

# Restart immediately after preemption or a crash; the steps resume from their checkpoints.
RETRIES = modal.Retries(initial_delay=0.0, max_retries=5)


def _setup():
    """Point the repo's state folders at the volume; seed ids/results from the repo once."""
    import shutil
    import sys
    for d in STATE:
        target = f"{VOL}/{d}"
        os.makedirs(target, exist_ok=True)
        if os.path.isdir(f"{SEED}/{d}"):
            for root, _, files in os.walk(f"{SEED}/{d}"):
                for f in files:
                    src = os.path.join(root, f)
                    dst = os.path.join(target, os.path.relpath(src, f"{SEED}/{d}"))
                    if not os.path.exists(dst):
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copy2(src, dst)
        link = f"{REPO}/{d}"
        if not os.path.islink(link):
            shutil.rmtree(link, ignore_errors=True)
            os.symlink(target, link)
    os.chdir(REPO)
    if REPO not in sys.path:
        sys.path.insert(0, REPO)


def _run_step(module, argv):
    _setup()
    # A retried call (preemption/crash) must resume, not wipe its own progress again.
    if "--fresh" in argv:
        marker = f"{VOL}/.fresh_{modal.current_function_call_id()}"
        if os.path.exists(marker):
            argv = [a for a in argv if a != "--fresh"]
            print("[modal] retry of a --fresh step: resuming instead", flush=True)
        else:
            open(marker, "w").close()
            vol.commit()
    from scripts.pipeline import call
    try:
        call(module, argv)
    finally:
        vol.commit()


@app.function(gpu=DEFAULT_GPU, cpu=4, memory=16384, timeout=24 * 3600, volumes={VOL: vol},
              retries=RETRIES, single_use_containers=True)
def gpu_step(module: str, argv: list):
    _run_step(module, argv)


@app.function(cpu=4, memory=16384, timeout=24 * 3600, volumes={VOL: vol}, retries=RETRIES,
              single_use_containers=True)
def cpu_step(module: str, argv: list):
    _run_step(module, argv)


# Only waits on other calls: tiny reservation keeps the 3x non-preemptible rate negligible.
@app.function(cpu=0.25, memory=1024, timeout=24 * 3600, volumes={VOL: vol}, nonpreemptible=True)
def orchestrate(task: int, force: bool = False, gpu: str = DEFAULT_GPU, config2: str = "configs/task2_vlm.yaml"):
    _setup()
    from scripts.pipeline import pending
    g = gpu_step.with_options(gpu=gpu)
    for i, todo in pending(task, force, config2):
        calls = []
        for s in todo:
            print(f"[stage {i}] start {s.name}: {s.module} {' '.join(s.args(force))}", flush=True)
            calls.append((s, (g if s.gpu else cpu_step).spawn(s.module, s.args(force))))
        failed = []
        for s, fc in calls:
            try:
                fc.get()
                print(f"[stage {i}] done  {s.name}", flush=True)
            except Exception as e:
                print(f"[stage {i}] FAILED {s.name}: {e!r}", flush=True)
                failed.append(s.name)
        vol.reload()   # see what the step containers committed before checking the next stage
        if failed:
            raise RuntimeError(f"stage {i} failed: {failed}; rerun the same command to resume")
    print(f"task {task}: all steps finished", flush=True)


@app.local_entrypoint()
def main(task: int = 1, force: bool = False, gpu: str = DEFAULT_GPU, config2: str = "configs/task2_vlm.yaml"):
    # spawn + get (as in Modal's long-training example): with --detach the orchestrator, and
    # through it every step, keeps running after this client exits.
    orchestrate.spawn(task, force, gpu, config2).get()
