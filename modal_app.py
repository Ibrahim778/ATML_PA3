"""Run the PA3 pipelines natively on Modal (https://modal.com).

Structure (following Modal's long-training example):
  * `run_task` runs one task's steps (scripts/pipeline.py) in order inside ONE GPU container,
    calling each task module's main(argv) directly; CPU-only steps run in the same container;
  * all state (outputs/, results/, ids/, data/, Hugging Face cache) lives on a v2 Volume;
    the function retries automatically after preemption or a crash and resumes: finished steps
    are skipped and training restarts from its last checkpoint on the Volume;
  * the local entrypoint uses spawn().get(), so `--detach` keeps the job alive after the
    terminal closes.

    pip install modal && modal setup                                   # once
    modal volume put atml-pa3-v2 data/first_plot_questions.yaml data/first_plot_questions.yaml
    modal run --detach modal_app.py --task 1                           # --force to start over
    modal run --detach modal_app.py --task 2 --gpu H100
    modal volume get atml-pa3-v2 results results                       # fetch results

Follow progress on the Modal dashboard or with `modal app logs atml-pa3`.
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


# The whole task, step after step, in ONE GPU container (one GPU billed at a time).
@app.function(gpu=DEFAULT_GPU, cpu=8, memory=32768, timeout=24 * 3600, volumes={VOL: vol},
              retries=RETRIES, single_use_containers=True)
def run_task(task: int, force: bool = False, config2: str = "configs/task2_vlm.yaml"):
    _setup()
    # A retry after preemption/crash resumes: finished steps are skipped, training resumes from
    # its checkpoint. Only the first attempt of a --force run wipes earlier progress.
    if force:
        marker = f"{VOL}/.force_{modal.current_function_call_id()}"
        if os.path.exists(marker):
            force = False
            print("[modal] retry of a --force run: resuming instead", flush=True)
        else:
            open(marker, "w").close()
            vol.commit()
    from scripts.pipeline import call, pending
    for i, todo in pending(task, force, config2):
        for s in todo:
            print(f"[stage {i}] {s.name}: {s.module} {' '.join(s.args(force))}", flush=True)
            try:
                call(s.module, s.args(force))
            finally:
                vol.commit()
    print(f"task {task}: all steps finished", flush=True)


@app.local_entrypoint()
def main(task: int = 1, force: bool = False, gpu: str = DEFAULT_GPU, config2: str = "configs/task2_vlm.yaml"):
    # spawn + get (as in Modal's long-training example): with --detach the job keeps running
    # after this client exits.
    run_task.with_options(gpu=gpu).spawn(task, force, config2).get()
