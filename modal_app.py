"""Run the PA3 pipelines on Modal (https://modal.com).

The job runs on the container's local disk; everything that must survive between runs
(outputs/, results/, ids/, logs/) is synced to a persistent Modal Volume every few minutes and
when the job ends, so the run scripts' skip/resume logic works across calls and after a
preemption (Modal restarts a preempted GPU function on the same input). data/ and the Hugging
Face cache are read straight from the Volume.

    pip install modal && modal setup                      # once
    modal run --detach modal_app.py --task 1              # Task 1 (add --force to start over)
    modal run --detach modal_app.py --task 2 --gpu H100
    modal volume get atml-pa3 ATML_PA3/results results    # copy results back
    modal volume put atml-pa3 data/first_plot_questions.yaml ATML_PA3/data/first_plot_questions.yaml

`--detach` keeps the run going after your terminal closes; follow it on the Modal dashboard
or with `modal app logs atml-pa3`. Start one run at a time: two runs would sync over each other.
"""
import modal

APP = "atml-pa3"
VOL_PATH = "/vol"
VOL_WORK = f"{VOL_PATH}/ATML_PA3"
LOCAL = "/root/work"
SRC = "/root/src"
SYNCED = ("outputs", "results", "ids", "logs")   # copied local <-> volume
SYNC_EVERY_S = 300

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "rsync")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", SRC, ignore=["data", "outputs", ".git", ".venv", "venv", "logs", "**/__pycache__"])
)
vol = modal.Volume.from_name(APP, create_if_missing=True)
app = modal.App(APP, image=image)


def _rsync(src, dst, *flags):
    import subprocess
    subprocess.run(["rsync", "-a", *flags, src, dst], check=True)


def _push(vol):
    """Mirror the local working state to the volume and commit it."""
    import os
    for d in SYNCED:
        if os.path.isdir(f"{LOCAL}/{d}"):
            os.makedirs(f"{VOL_WORK}/{d}", exist_ok=True)
            _rsync(f"{LOCAL}/{d}/", f"{VOL_WORK}/{d}/", "--delete", "--exclude", "*.tmp")
    vol.commit()


# Explicit CPU/RAM reservation: Full FT keeps an FP32 copy of the starting weights on the host and
# merge/diagnostics load FP32 models, so the small default request is not enough.
@app.function(gpu="A100-80GB", cpu=8, memory=32768, timeout=24 * 3600, volumes={VOL_PATH: vol})
def run(task: int = 1, force: bool = False):
    import os
    import subprocess
    import threading

    # 1. code from the image; persistent state from the volume (seeded from the repo if absent)
    _rsync(f"{SRC}/", f"{LOCAL}/")
    for d in SYNCED:
        os.makedirs(f"{LOCAL}/{d}", exist_ok=True)
        if os.path.isdir(f"{VOL_WORK}/{d}"):
            _rsync(f"{VOL_WORK}/{d}/", f"{LOCAL}/{d}/")
    os.makedirs(f"{VOL_WORK}/data", exist_ok=True)
    if not os.path.islink(f"{LOCAL}/data"):
        subprocess.run(["rm", "-rf", f"{LOCAL}/data"], check=True)
        os.symlink(f"{VOL_WORK}/data", f"{LOCAL}/data")

    # 2. periodic sync so checkpoints survive a preemption or crash
    stop = threading.Event()

    def syncer():
        while not stop.wait(SYNC_EVERY_S):
            try:
                _push(vol)
            except Exception as e:   # a failed sync must not kill training
                print(f"[sync] {e}", flush=True)

    t = threading.Thread(target=syncer, daemon=True)
    t.start()

    env = dict(os.environ, HF_HOME=f"{VOL_PATH}/hf", PYTHONUNBUFFERED="1")
    if force:
        # A preempted call is restarted on the same input; only the first attempt may wipe
        # progress, later attempts must resume from the synced checkpoints.
        marker = f"{VOL_WORK}/logs/.force_{modal.current_function_call_id()}"
        if os.path.exists(marker):
            print("[modal] restarted after preemption: resuming instead of --force", flush=True)
        else:
            os.makedirs(os.path.dirname(marker), exist_ok=True)
            open(marker, "w").close()
            vol.commit()
            env["FORCE"] = "1"
    cmd = f"./run_task{task}.sh 2>&1 | tee -a logs/task{task}.log; exit ${{PIPESTATUS[0]}}"
    try:
        subprocess.run(["bash", "-c", cmd], cwd=LOCAL, env=env, check=True)
    finally:
        stop.set()
        t.join()
        _push(vol)


@app.local_entrypoint()
def main(task: int = 1, force: bool = False, gpu: str = "A100-80GB"):
    run.with_options(gpu=gpu).remote(task=task, force=force)
