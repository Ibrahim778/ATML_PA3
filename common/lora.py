"""LoRA and OPLoRA adapters on nn.Linear, plus injection, merging, export and counting.

LoRA:   W = W0 + (alpha/r) B A,          A in R^{r x d_in} (random), B in R^{d_out x r} (zeros)
OPLoRA: W = W0 + (alpha/r) B~ A~,        B~ = B - U_k (U_k^T B),  A~ = A - (A V_k) V_k^T
where W0 = U S V^T and U_k, V_k are the leading-k singular vectors, computed once in FP32
and frozen. The projections are applied to the factors (no dense d x d projectors).
"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha = r, alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        dev = base.weight.device
        # FP32 trainable factors; same init as PEFT (kaiming-uniform A, zero B).
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    # effective factors (overridden by OPLoRA)
    def factors(self):
        return self.lora_B, self.lora_A

    def delta_weight(self) -> torch.Tensor:
        """Dense (alpha/r) B A in FP32, for merging and diagnostics."""
        B, A = self.factors()
        return self.scaling * (B.float() @ A.float())

    def forward(self, x):
        out = self.base(x)
        B, A = self.factors()
        h = self.dropout(x)
        # scaling is folded into the small (d_out x r) factor, so no extra pass over the
        # full-size output; under autocast the FP32 factors are cast once and cached.
        Bs = B * self.scaling
        if torch.is_autocast_enabled(x.device.type):
            return out + F.linear(F.linear(h, A), Bs)
        return out + F.linear(F.linear(h.to(A.dtype), A), Bs).to(out.dtype)


class OPLoRALinear(LoRALinear):
    def __init__(self, base: nn.Linear, r: int, alpha: float, k: int, dropout: float = 0.0):
        super().__init__(base, r, alpha, dropout)
        self.k = k
        W0 = base.weight.detach().float()
        U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
        self.register_buffer("U_k", U[:, :k].contiguous(), persistent=True)     # (d_out, k)
        self.register_buffer("V_k", Vh[:k, :].T.contiguous(), persistent=True)  # (d_in, k)

    def factors(self):
        B, A = self.lora_B, self.lora_A
        Bt = B - self.U_k @ (self.U_k.T @ B)
        At = A - (A @ self.V_k) @ self.V_k.T
        return Bt, At


def _parent_and_name(model, full_name):
    parts = full_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_adapters(model, targets, r, alpha, dropout=0.0, kind="lora", k=None, verbose=True):
    """Replace every nn.Linear whose name ends with one of `targets` by a (OP)LoRA wrapper.
    Modules are visited in deterministic named_modules() order, so with the same seed LoRA-16 and
    OPLoRA-16 receive identical unprojected A initialisations. Returns preprocessing seconds."""
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and n.split(".")[-1] in targets]
    t0 = time.time()
    for n in names:
        parent, attr = _parent_and_name(model, n)
        base = getattr(parent, attr)
        if kind == "oplora":
            new = OPLoRALinear(base, r, alpha, k, dropout)
        else:
            new = LoRALinear(base, r, alpha, dropout)
        setattr(parent, attr, new)
    if verbose:
        print(f"[lora] injected {kind} into {len(names)} linear layers (r={r}, alpha={alpha}, k={k})")
    return time.time() - t0


def adapter_modules(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def adapter_state_dict(model, projected=True):
    """Trainable factors (and, for OPLoRA, the projected factors used for inference)."""
    sd = {}
    for n, m in adapter_modules(model).items():
        sd[f"{n}.lora_A"] = m.lora_A.detach().cpu()
        sd[f"{n}.lora_B"] = m.lora_B.detach().cpu()
        if projected and isinstance(m, OPLoRALinear):
            B, A = m.factors()
            sd[f"{n}.proj_A"] = A.detach().cpu()
            sd[f"{n}.proj_B"] = B.detach().cpu()
    return sd


def basis_state_dict(model):
    return {f"{n}.{b}": getattr(m, b).detach().cpu()
            for n, m in adapter_modules(model).items() if isinstance(m, OPLoRALinear)
            for b in ("U_k", "V_k")}


def load_adapter_factors(model, sd):
    for n, m in adapter_modules(model).items():
        m.lora_A.data.copy_(sd[f"{n}.lora_A"].to(m.lora_A.device))
        m.lora_B.data.copy_(sd[f"{n}.lora_B"].to(m.lora_B.device))


@torch.no_grad()
def merge_factors_into(model, sd, scaling, projected_prefix="proj"):
    """Merge exported factors into a *fresh* plain model in place: W <- W0 + scaling * B A.
    For OPLoRA exports, the projected factors (proj_A / proj_B) are used."""
    modules = dict(model.named_modules())
    keys = sorted({k.rsplit(".", 1)[0] for k in sd})
    for n in keys:
        lin = modules[n]
        if f"{n}.{projected_prefix}_A" in sd:
            A, B = sd[f"{n}.{projected_prefix}_A"], sd[f"{n}.{projected_prefix}_B"]
        else:
            A, B = sd[f"{n}.lora_A"], sd[f"{n}.lora_B"]
        dW = scaling * (B.float() @ A.float())
        lin.weight.data = (lin.weight.data.float() + dW.to(lin.weight.device)).to(lin.weight.dtype)
    return len(keys)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": train}


def unique_named_parameters(model):
    """named_parameters() already de-duplicates shared tensors (tied embeddings counted once)."""
    seen, out = set(), []
    for n, p in model.named_parameters(remove_duplicate=True):
        if id(p) in seen:
            continue
        seen.add(id(p))
        out.append((n, p))
    return out
