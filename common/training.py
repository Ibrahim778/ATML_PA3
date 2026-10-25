"""Shared training utilities: parameter groups, LR schedules, AMP stepping, the two
loss normalisations (Task 1 token-level, Task 2 example-weighted) and resumable checkpoints."""
import math
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

from common.chat import IGNORE
from common.seeding import get_rng_state, set_rng_state


# ----------------------------------------------------------------------------- optimiser
def param_groups(named_params, weight_decay: float, lr: float = None, extra: dict = None):
    """AdamW groups: no weight decay on biases and on parameters with fewer than two dims."""
    decay, no_decay = [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim < 2 or n.endswith(".bias")) else decay).append(p)
    groups = []
    for params, wd in ((decay, weight_decay), (no_decay, 0.0)):
        if params:
            g = {"params": params, "weight_decay": wd}
            if lr is not None:
                g["lr"] = lr
            if extra:
                g.update(extra)
            groups.append(g)
    return groups


def cosine_lambda(total: int, warmup: int, min_ratio: float = 0.0):
    """LR multiplier for update index u (0-based): linear warmup to 1 over `warmup` updates,
    then cosine decay to `min_ratio` at the final update."""
    def f(u):
        if warmup > 0 and u < warmup:
            return (u + 1) / warmup
        progress = (u - warmup) / max(1, total - warmup - 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return f


def make_scheduler(optimizer, total: int, warmup: int, min_ratio: float = 0.0):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lambda(total, warmup, min_ratio))


# ----------------------------------------------------------------------------- AMP
class AmpStepper:
    """FP16 autocast + dynamic GradScaler. Unscale -> clip -> step; detects skipped updates
    (the scaler lowers its scale when it finds inf/nan gradients)."""

    def __init__(self, optimizer, params, device: torch.device, clip_norm: float = 1.0,
                 enabled: bool = True, dtype=torch.float16):
        self.opt = optimizer
        self.params = [p for p in params if p.requires_grad]
        self.clip = clip_norm
        self.device = device
        self.enabled = enabled and device.type in ("cuda", "mps")
        self.dtype = dtype
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.enabled)
        self.n_skipped = 0

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.enabled)

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    def step(self):
        self.scaler.unscale_(self.opt)
        gnorm = torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        scale_before = self.scaler.get_scale() if self.enabled else 1.0
        self.scaler.step(self.opt)
        self.scaler.update()
        skipped = bool(self.enabled and self.scaler.get_scale() < scale_before)
        if not self.enabled and not torch.isfinite(gnorm):
            skipped = True
        self.n_skipped += int(skipped)
        self.opt.zero_grad(set_to_none=True)
        return float(gnorm), skipped

    def state_dict(self):
        return {"scaler": self.scaler.state_dict(), "n_skipped": self.n_skipped}

    def load_state_dict(self, sd):
        self.scaler.load_state_dict(sd["scaler"])
        self.n_skipped = sd["n_skipped"]


# ----------------------------------------------------------------------------- losses
def supervised_token_nll(hidden: torch.Tensor, labels: torch.Tensor, lm_head) -> tuple:
    """Per-token NLL (FP32) at supervised positions only.

    hidden: (B, T, H) final hidden states; labels: (B, T) with IGNORE on unsupervised positions.
    Position t predicts labels[t+1]. Only supervised rows go through the LM head, which avoids
    materialising (B, T, vocab) logits. Returns (nll[M], example_index[M]).
    """
    h = hidden[:, :-1, :]
    y = labels[:, 1:]
    sel = y != IGNORE
    ex_idx = torch.nonzero(sel, as_tuple=True)[0]
    logits = lm_head(h[sel]).float()
    nll = F.cross_entropy(logits, y[sel], reduction="none")
    return nll, ex_idx


def token_level_loss(nll: torch.Tensor, n_tokens_effective_batch: int) -> torch.Tensor:
    """Task 1: sum of token NLL in this microbatch divided by the number of supervised tokens in
    the whole effective batch (all accumulation microbatches). Summing over microbatches gives
    the token mean over the effective batch."""
    return nll.sum() / max(1, n_tokens_effective_batch)


def example_weighted_loss(nll: torch.Tensor, ex_idx: torch.Tensor, batch_size_micro: int,
                          n_examples_effective_batch: int) -> torch.Tensor:
    """Task 2: mean token NLL within each example, then mean over the effective batch."""
    sums = torch.zeros(batch_size_micro, device=nll.device, dtype=nll.dtype).index_add_(0, ex_idx, nll)
    cnts = torch.zeros(batch_size_micro, device=nll.device, dtype=nll.dtype).index_add_(
        0, ex_idx, torch.ones_like(nll))
    assert torch.all(cnts > 0), "every example needs at least one supervised token"
    return (sums / cnts).sum() / n_examples_effective_batch


# ----------------------------------------------------------------------------- checkpoints
def save_resume(path, *, model_state, optimizer, scheduler, stepper, update, data_pos, extra=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tmp = path / "resume.pt.tmp"
    torch.save({
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "stepper": stepper.state_dict(),
        "update": update,
        "data_pos": data_pos,
        "rng": get_rng_state(),
        "extra": extra or {},
    }, tmp)
    os.replace(tmp, path / "resume.pt")


def load_resume(path, map_location="cpu"):
    ck = torch.load(Path(path) / "resume.pt", map_location=map_location, weights_only=False)
    return ck


def restore_rng(ck):
    set_rng_state(ck["rng"])


def latest_checkpoint(run_dir) -> Path:
    """Most recent ckpt_<update> dir containing resume.pt, or None."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return None
    cands = []
    for d in run_dir.glob("ckpt_*"):
        if (d / "resume.pt").exists():
            try:
                cands.append((int(d.name.split("_")[1]), d))
            except ValueError:
                pass
    return max(cands)[1] if cands else None


def prune_resume_states(run_dir, keep: Path):
    """Delete optimizer/resume states of older checkpoints (model weights are kept)."""
    for d in Path(run_dir).glob("ckpt_*"):
        if d != keep and (d / "resume.pt").exists():
            (d / "resume.pt").unlink()


def dir_size_bytes(path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def copy_tree(src, dst):
    shutil.copytree(src, dst, dirs_exist_ok=True)
