"""Merge adapters into fresh backbone copies (OPLoRA: projected factors) and check merged vs
unmerged logits on a small batch (FP32).

python -m task1_finetune.merge_export --config configs/task1_ft.yaml [--conditions lora16_1250 ...]
"""
import argparse
import json
import time

import torch
from safetensors.torch import load_file

from common import lora as L
from common.chat import render_prompt
from common.io import load_yaml, results_dir, save_json
from common.training import dir_size_bytes
from task1_finetune.conditions import adapter_dir, condition_model_path

CHECK_PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "What is the capital of France?",
    "Write a short JSON object describing a cat.",
    "A man is sitting on a roof. He",
]


def plain_model(cfg, dtype=torch.float32):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(cfg["model_id"], revision=cfg["model_revision"], torch_dtype=dtype)


@torch.no_grad()
def logits_for(model, tok, prompts):
    texts = [render_prompt(tok, p) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return model(**enc).logits.float(), enc["attention_mask"].bool()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1_ft.yaml")
    ap.add_argument("--conditions", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
    tok.padding_side = "left"
    conds = args.conditions or [c for c, v in cfg["conditions"].items()
                                if v["method"] and cfg["methods"][v["method"]]["kind"] != "full"]
    rd = results_dir(cfg)
    for cond in conds:
        ad = adapter_dir(cfg, cond)
        acfg = json.load(open(ad / "adapter_config.json"))
        sd = load_file(ad / "adapter.safetensors")
        t0 = time.time()
        merged = plain_model(cfg)
        n = L.merge_factors_into(merged, sd, acfg["scaling"])
        merge_s = time.time() - t0
        # unmerged reference: fresh backbone + LoRA wrappers holding the (projected) factors
        unmerged = plain_model(cfg)
        L.inject_adapters(unmerged, acfg["targets"], acfg["r"], acfg["alpha"], 0.0, kind="lora", verbose=False)
        for name, m in L.adapter_modules(unmerged).items():
            key = "proj" if f"{name}.proj_A" in sd else "lora"
            m.lora_A.data.copy_(sd[f"{name}.{key}_A"])
            m.lora_B.data.copy_(sd[f"{name}.{key}_B"])
        merged.eval(); unmerged.eval()
        lm, mask = logits_for(merged, tok, CHECK_PROMPTS)
        lu, _ = logits_for(unmerged, tok, CHECK_PROMPTS)
        diff = (lm - lu).abs()[mask].max().item()
        argmax_agree = (lm.argmax(-1) == lu.argmax(-1))[mask].float().mean().item()
        check = {"condition": cond, "n_merged_modules": n, "max_abs_logit_diff_fp32": diff,
                 "argmax_agreement": argmax_agree, "tolerance": cfg["eval"]["merge_check_tol"],
                 "passed": diff <= cfg["eval"]["merge_check_tol"], "merge_s": merge_s}
        if any(k.endswith("proj_A") for k in sd):
            # numerical verification of the OPLoRA constraints using freshly computed bases
            errs = []
            base = plain_model(cfg)
            bmods = dict(base.named_modules())
            for name in sorted({k.rsplit(".", 1)[0] for k in sd}):
                W0 = bmods[name].weight.float()
                U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
                k = acfg["k"]
                errs.append(max((U[:, :k].T @ sd[f"{name}.proj_B"]).norm().item(),
                                (sd[f"{name}.proj_A"] @ Vh[:k].T).norm().item()))
            check["oplora_max_projection_residual"] = max(errs)
            del base
        out_dir = condition_model_path(cfg, cond)
        t1 = time.time()
        merged.half().save_pretrained(out_dir, safe_serialization=True)
        tok.save_pretrained(out_dir)
        check["export_s"] = time.time() - t1
        check["merged_bytes"] = dir_size_bytes(out_dir)
        check["adapter_bytes"] = dir_size_bytes(ad / "adapter.safetensors")
        save_json(check, rd / f"merge_check_{cond}.json")
        print(check)
        del merged, unmerged


if __name__ == "__main__":
    main()
