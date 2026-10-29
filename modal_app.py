"""Run the PA3 pipelines on Modal (https://modal.com).

The code is baked into the image; everything that must survive between runs (outputs/,
results/, ids/, data/ and the Hugging Face cache) lives on a persistent Modal Volume.
The run scripts' skip/resume logic therefore works across separate Modal calls.

    pip install modal && modal setup                      # once
    modal run --detach modal_app.py --task 1              # Task 1 (add --force to start over)
    modal run --detach modal_app.py --task 2 --gpu A100-80GB
    modal volume get atml-pa3 ATML_PA3/results results    # copy results back
    modal volume put atml-pa3 data/first_plot_questions.yaml ATML_PA3/data/first_plot_questions.yaml

`--detach` keeps the run going after your terminal closes; follow it on the Modal dashboard
or with `modal app logs atml-pa3`.
"""
import modal

APP = "atml-pa3"
VOL_PATH = "/vol"
WORK = f"{VOL_PATH}/ATML_PA3"
SRC = "/root/src"
PERSISTENT = ("outputs", "results", "ids", "data", "logs")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "rsync")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", SRC, ignore=["data", "outputs", ".git", ".venv", "venv", "logs", "**/__pycache__"])
)
vol = modal.Volume.from_name(APP, create_if_missing=True)
app = modal.App(APP, image=image)


@app.function(gpu="H100", timeout=24 * 3600, volumes={VOL_PATH: vol})
def run(task: int = 1, force: bool = False):
    import os
    import subprocess

    os.makedirs(WORK, exist_ok=True)
    # Code always comes from the image (latest local version); persistent folders are only seeded
    # from the repo when the volume does not have them yet, so earlier results are never overwritten.
    excl = sum((["--exclude", f"/{d}"] for d in PERSISTENT), [])
    subprocess.run(["rsync", "-a", *excl, f"{SRC}/", f"{WORK}/"], check=True)
    for d in PERSISTENT:
        if os.path.isdir(f"{SRC}/{d}"):
            subprocess.run(["rsync", "-a", "--ignore-existing", f"{SRC}/{d}/", f"{WORK}/{d}/"], check=True)
        os.makedirs(f"{WORK}/{d}", exist_ok=True)

    env = dict(os.environ, HF_HOME=f"{VOL_PATH}/hf", PYTHONUNBUFFERED="1")
    if force:
        env["FORCE"] = "1"
    cmd = f"./run_task{task}.sh 2>&1 | tee -a logs/task{task}.log; exit ${{PIPESTATUS[0]}}"
    try:
        subprocess.run(["bash", "-c", cmd], cwd=WORK, env=env, check=True)
    finally:
        vol.commit()   # make outputs/results visible to later runs and `modal volume get`


@app.local_entrypoint()
def main(task: int = 1, force: bool = False, gpu: str = "H100"):
    run.with_options(gpu=gpu).remote(task=task, force=force)
