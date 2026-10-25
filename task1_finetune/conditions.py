"""Where each Task 1 evaluation condition's weights live."""
from pathlib import Path

from common.io import outputs_dir


def condition_model_path(cfg, cond) -> str:
    c = cfg["conditions"][cond]
    if c["method"] is None:
        return cfg["model_id"]
    kind = cfg["methods"][c["method"]]["kind"]
    out = outputs_dir(cfg)
    if kind == "full":
        return str(out / c["method"] / f"ckpt_{c['update']}" / "model")
    return str(out / "merged" / cond)


def adapter_dir(cfg, cond) -> Path:
    c = cfg["conditions"][cond]
    return outputs_dir(cfg) / c["method"] / f"ckpt_{c['update']}"
