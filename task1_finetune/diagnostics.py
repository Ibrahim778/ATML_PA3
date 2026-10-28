"""Task 1 diagnostics (FP32).

Weight space: for down_proj in layers 0, 13, 27 and each diagnosed condition,
  rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2,  eta = ||dW||_F / ||W0||_F
with U_k the leading-k left singular vectors of the *starting* weight W0.
Full FT: dW = W_trained - W0 from FP32 copies saved during training (diag*.safetensors).
Adapters: dW = (alpha/r) B A from the exported factors (OPLoRA: projected factors).

Activation: forward pre-hooks cache the starting model's down_proj inputs z at the last
non-padding prompt token for the probe set (32 GSM8K + 32 HellaSwag, benchmark prompt
formats), then R(z) = ||dW z|| / (||W0 z|| + 1e-8) per probe and layer.

python -m task1_finetune.diagnostics --config configs/task1_ft.yaml
"""
import argparse
import json

import numpy as np
import torch
from safetensors.torch import load_file
from tqdm import tqdm

from common.io import load_json, load_yaml, outputs_dir, read_jsonl, results_dir, save_json
from common.seeding import get_device
from task1_finetune.conditions import adapter_dir


def key(li):
    return f"layers.{li}.mlp.down_proj.weight"


def delta_weights(cfg, cond, layers):
    c = cfg["conditions"][cond]
    m = cfg["methods"][c["method"]]
    out = outputs_dir(cfg) / c["method"]
    if m["kind"] == "full":
        w0 = load_file(out / "diag_init.safetensors")
        wt = load_file(out / f"ckpt_{c['update']}" / "diag.safetensors")
        return {li: (wt[key(li)].double() - w0[key(li)].double()).float() for li in layers}
    ad = adapter_dir(cfg, cond)
    sd = load_file(ad / "adapter.safetensors")
    acfg = json.load(open(ad / "adapter_config.json"))
    res = {}
    for li in layers:
        name = f"model.layers.{li}.mlp.down_proj"
        pre = "proj" if f"{name}.proj_A" in sd else "lora"
        res[li] = acfg["scaling"] * (sd[f"{name}.{pre}_B"].float() @ sd[f"{name}.{pre}_A"].float())
    return res


def probe_prompts(cfg, tok):
    """Benchmark-format prompts for the probe IDs; GSM8K prompts are taken verbatim from the
    baseline lm-eval log (chat template, thinking disabled) when available."""
    ids = load_json(cfg["diagnostics"]["probe_ids_file"])
    rd = results_dir(cfg)
    from datasets import load_dataset
    base = {r["doc_id"]: r for r in read_jsonl(rd / "baseline_per_example_gsm8k.jsonl")}
    gsm = load_dataset(cfg["data"]["gsm8k_id"], cfg["data"]["gsm8k_config"], split="test")
    prompts = []
    for i in ids["gsm8k_test"]:
        if i in base and base[i].get("prompt"):
            p = base[i]["prompt"]
        else:  # stock gsm8k_cot zero-shot text inside the non-thinking chat template
            p = tok.apply_chat_template([{"role": "user", "content": f"Q: {gsm[i]['question']}\nA:"}],
                                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompts.append(("math", f"gsm8k:{i}", p))
    hs = load_dataset("Rowan/hellaswag", split="validation")
    import re

    def prep(t):  # lm-eval hellaswag preprocess
        t = t.strip().replace(" [title]", ". ")
        t = re.sub("\\[.*?\\]", "", t)
        return t.replace("  ", " ")
    for i in ids["hellaswag_validation"]:
        d = hs[i]
        ctx = d["ctx_a"] + " " + d["ctx_b"].capitalize()
        prompts.append(("nonmath", f"hellaswag:{i}", prep(d["activity_label"] + ": " + ctx)))
    return prompts


@torch.no_grad()
def cache_activations(cfg, layers, device):
    path = outputs_dir(cfg) / "diag_probe_z.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], revision=cfg["model_revision"],
                                                 dtype=torch.float32).to(device).eval()
    store = {}
    hooks = [model.model.layers[li].mlp.down_proj.register_forward_pre_hook(
        lambda mod, inp, li=li: store.__setitem__(li, inp[0][0, -1].float().cpu().clone())) for li in layers]
    prompts = probe_prompts(cfg, tok)
    Z = {li: [] for li in layers}
    meta = []
    for group, pid, text in tqdm(prompts, desc="cache z"):
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)  # batch 1: last = last prompt token
        model(**enc)
        for li in layers:
            Z[li].append(store[li])
        meta.append({"group": group, "id": pid, "n_tokens": int(enc["input_ids"].shape[1])})
    for h in hooks:
        h.remove()
    W0 = {li: model.model.layers[li].mlp.down_proj.weight.detach().float().cpu() for li in layers}
    out = {"Z": {li: torch.stack(v) for li, v in Z.items()}, "meta": meta, "W0": W0}
    torch.save(out, path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1_ft.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    dg = cfg["diagnostics"]
    layers, k = dg["layers"], dg["k"]
    device = get_device(cfg["device"])
    cache = cache_activations(cfg, layers, device)
    W0, Z, meta = cache["W0"], cache["Z"], cache["meta"]
    groups = np.array([m["group"] for m in meta])
    Uk = {li: torch.linalg.svd(W0[li], full_matrices=False)[0][:, :k] for li in layers}
    weight_res, act_res, per_probe = {}, {}, []
    for cond in dg["conditions"]:
        dW = delta_weights(cfg, cond, layers)
        weight_res[cond], act_res[cond] = {}, {}
        for li in layers:
            d = dW[li]
            nd = d.norm().item()
            rho = ((Uk[li].T @ d).norm().item() ** 2 / nd ** 2) if nd > 0 else float("nan")
            weight_res[cond][str(li)] = {"rho_k": rho, "eta": nd / W0[li].norm().item(), "dW_fro": nd}
            z = Z[li]
            R = ((z @ d.T).norm(dim=1) / ((z @ W0[li].T).norm(dim=1) + 1e-8)).numpy()
            stats = {}
            for g in ("math", "nonmath"):
                v = R[groups == g]
                stats[g] = {"mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std(ddof=1)), "n": int(v.size)}
            stats["math_over_nonmath_mean"] = stats["math"]["mean"] / stats["nonmath"]["mean"] if stats["nonmath"]["mean"] > 0 else float("nan")
            act_res[cond][str(li)] = stats
            per_probe += [{"condition": cond, "layer": li, "probe": m["id"], "group": m["group"], "R": float(r)}
                          for m, r in zip(meta, R)]
    rd = results_dir(cfg)
    save_json({"k": k, "layers": layers, "results": weight_res}, rd / "diagnostics_weights.json")
    save_json({"layers": layers, "results": act_res, "n_probes": len(meta)}, rd / "diagnostics_activation.json")
    from common.io import write_jsonl
    write_jsonl(per_probe, rd / "diagnostics_activation_per_probe.jsonl")
    print(json.dumps(weight_res, indent=1)[:2000])


if __name__ == "__main__":
    main()
