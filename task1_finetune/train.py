"""Task 1 training: Full FT, Full FT + L2-SP, LoRA-4, LoRA-16, OPLoRA-16 on MetaMathQA.

Every run starts from the original checkpoint. 1,250 updates of 32 examples (micro 2 x accum 16),
examples consumed in source order (update u uses examples [32u, 32u+32)). Loss = token-level
CE over all supervised tokens of the effective batch (common.training.token_level_loss).
Resumable: the newest ckpt_<u>/resume.pt is picked up automatically.

python -m task1_finetune.train --config configs/task1_ft.yaml --method full
"""
import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm

from common import lora as L
from common.chat import build_sft_example, pad_batch
from common.io import append_jsonl, truncate_jsonl, config_hash, load_json, load_yaml, outputs_dir, results_dir, save_json
from common.resources import Timer, peak_memory_gb, reset_peak_memory, tensor_bytes
from common.seeding import get_device, set_seed
from common.training import (AmpStepper, dir_size_bytes, latest_checkpoint, load_resume, make_scheduler,
                             param_groups, prune_resume_states, restore_rng, save_resume,
                             supervised_token_nll, token_level_loss)


def tokenized_dataset(cfg, tok):
    """Tokenise the selected examples once; cached on disk keyed by the data config."""
    d = cfg["data"]
    key = config_hash({"data": d, "model": cfg["model_id"], "rev": cfg["model_revision"]})
    path = outputs_dir(cfg) / f"tokenized_{key}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    from datasets import load_dataset
    ids = load_json(d["ids_file"])["source_indices"]
    ds = load_dataset(d["dataset_id"], split=d["split"]).select(ids)
    out = []
    for ex in tqdm(ds, desc="tokenise"):
        e = build_sft_example(tok, ex["query"], ex["response"], max_length=d["max_length"])
        assert e is not None
        out.append({"input_ids": e["input_ids"], "labels": e["labels"], "n_target": e["n_target"]})
    torch.save(out, path)
    return out


def load_model(cfg, mcfg, device):
    from transformers import AutoModelForCausalLM
    full = mcfg["kind"] == "full"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"],
        torch_dtype=torch.float32 if full else torch.float16)   # FP32 master params for Full FT
    model.config.use_cache = False
    info = {}
    if not full:
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(device)
        set_seed(cfg["seed"])   # identical A init for LoRA-16 and OPLoRA-16
        info["preprocess_s"] = L.inject_adapters(
            model, mcfg["targets"], mcfg["r"], mcfg["alpha"], mcfg["dropout"],
            kind="oplora" if mcfg["kind"] == "oplora" else "lora", k=mcfg.get("k"))
        if mcfg["kind"] == "oplora":
            info["basis_bytes"] = tensor_bytes(L.basis_state_dict(model).values())
    else:
        model.to(device)
    if cfg["train"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if cfg["train"].get("compile") and device.type in ("cuda", "mps"):
        # Regional compilation: each decoder layer (incl. its adapters) becomes one fused graph,
        # cutting the many small kernel launches of the adapter path. dynamic=True avoids a
        # recompile for every sequence length.
        for layer in model.model.layers:
            layer.compile(dynamic=True)
    return model, info


def diag_tensors(model, cfg):
    """FP32 copies of the diagnosed down_proj weights (full FT) for weight-space diagnostics."""
    dg = cfg["diagnostics"]
    out = {}
    for li in dg["layers"]:
        mod = model.model.layers[li].mlp.down_proj
        w = mod.base.weight if isinstance(mod, L.LoRALinear) else mod.weight
        out[f"layers.{li}.mlp.down_proj.weight"] = w.detach().float().cpu().contiguous()
    return out


def export_checkpoint(model, mcfg, ckpt_dir, tok):
    """Weights needed for evaluation/diagnostics. Full FT: FP16 HF model (+FP32 diag tensors).
    Adapters: trainable factors and (OPLoRA) projected factors."""
    t0 = time.time()
    ckpt_dir = Path(ckpt_dir)
    if mcfg["kind"] == "full":
        model.save_pretrained(ckpt_dir / "model", safe_serialization=True,
                              state_dict={k: v.half() for k, v in model.state_dict().items()})
        tok.save_pretrained(ckpt_dir / "model")
    else:
        save_file({k: v.contiguous() for k, v in L.adapter_state_dict(model).items()}, ckpt_dir / "adapter.safetensors")
        with open(ckpt_dir / "adapter_config.json", "w") as f:
            json.dump({**mcfg, "scaling": mcfg["alpha"] / mcfg["r"]}, f, indent=2)
    return time.time() - t0


@torch.no_grad()
def displacement(model, ref_sd):
    sq = 0.0
    for n, p in L.unique_named_parameters(model):
        sq += (p.detach().float().cpu() - ref_sd[n].float()).pow(2).sum().item()
    return sq ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1_ft.yaml")
    ap.add_argument("--method", required=True, choices=["full", "l2sp", "lora4", "lora16", "oplora16"])
    ap.add_argument("--max-updates", type=int, default=None, help="stop early (runtime estimate)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    tc, mcfg = cfg["train"], cfg["methods"][args.method]
    device = get_device(cfg["device"])
    set_seed(cfg["seed"])
    run_dir = outputs_dir(cfg) / args.method
    run_dir.mkdir(parents=True, exist_ok=True)
    rd = results_dir(cfg)
    log_path = rd / f"{args.method}_train_log.jsonl"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
    data = tokenized_dataset(cfg, tok)
    eff = tc["micro_batch"] * tc["grad_accum"]
    assert len(data) == tc["num_updates"] * eff

    reset_peak_memory(device)
    model, info = load_model(cfg, mcfg, device)
    lam = float(mcfg.get("l2sp_lambda", 0.0))
    summary = {"method": args.method, "config": {"train": tc, "method": mcfg}, "device": str(device), **info}

    theta0, ref_cpu = None, None
    if lam > 0:
        t0 = time.time()
        theta0 = {n: p.detach().clone().float() for n, p in L.unique_named_parameters(model)}
        summary["theta0_bytes"] = tensor_bytes(theta0.values())
        summary["theta0_copy_s"] = time.time() - t0
        summary["theta0_n_params"] = sum(t.numel() for t in theta0.values())
    if mcfg["kind"] == "full":
        ref_cpu = {n: p.detach().float().cpu().clone() for n, p in L.unique_named_parameters(model)}
        save_file(diag_tensors(model, cfg), run_dir / "diag_init.safetensors")

    named = L.unique_named_parameters(model)
    opt = torch.optim.AdamW(param_groups(named, tc["weight_decay"]), lr=mcfg["lr"],
                            betas=tuple(tc["betas"]), eps=tc["eps"], fused=device.type == "cuda")
    sched = make_scheduler(opt, tc["num_updates"], tc["warmup_updates"], 0.0)
    stepper = AmpStepper(opt, [p for _, p in named], device, tc["clip_norm"])
    counts = L.count_params(model)
    summary["params"] = counts
    print(f"[{args.method}] trainable {counts['trainable']:,} / total {counts['total']:,}")

    start, timer = 0, Timer()
    ck = latest_checkpoint(run_dir)
    if ck is not None:
        state = load_resume(ck)
        if mcfg["kind"] == "full":
            model.load_state_dict(state["model"])
        else:
            L.load_adapter_factors(model, state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        stepper.load_state_dict(state["stepper"])
        restore_rng(state)
        start = state["update"]
        assert state["data_pos"] == start * eff
        timer.elapsed_before = state["extra"].get("elapsed", 0.0)
        summary.update(state["extra"].get("summary", {}))
        truncate_jsonl(log_path, "update", start)
        print(f"[{args.method}] resumed from {ck} at update {start}")
        del state
    elif log_path.exists():
        log_path.unlink()

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model.train()
    end = tc["num_updates"] if args.max_updates is None else min(tc["num_updates"], start + args.max_updates)
    pbar = tqdm(range(start, end), desc=args.method, initial=start, total=tc["num_updates"])
    for u in pbar:
        batch_ex = data[u * eff:(u + 1) * eff]
        n_tok = sum(e["n_target"] for e in batch_ex)
        sft_sum = 0.0
        for m in range(tc["grad_accum"]):
            mb = pad_batch(batch_ex[m * tc["micro_batch"]:(m + 1) * tc["micro_batch"]], pad_id)
            mb = {k: v.to(device) for k, v in mb.items()}
            with stepper.autocast():
                h = model.model(input_ids=mb["input_ids"], attention_mask=mb["attention_mask"]).last_hidden_state
                nll, _ = supervised_token_nll(h, mb["labels"], model.lm_head)
                loss = token_level_loss(nll, n_tok)
            sft_sum += loss.item()
            if lam > 0 and m == tc["grad_accum"] - 1:
                omega = 0.5 * sum((p.float() - theta0[n]).pow(2).sum() for n, p in named)
                loss = loss + lam * omega
            stepper.backward(loss)
        lr = sched.get_last_lr()[0]
        gnorm, skipped = stepper.step()
        sched.step()
        rec = {"update": u + 1, "sft_loss": sft_sum, "lr": lr, "grad_norm": gnorm, "skipped": skipped,
               "loss_scale": stepper.scaler.get_scale() if stepper.enabled else 1.0,
               "n_tokens": n_tok, "elapsed_s": timer.elapsed()}
        if lam > 0:
            rec.update({"omega": omega.item(), "lambda_omega": lam * omega.item(),
                        "total_loss": sft_sum + lam * omega.item()})
        append_jsonl(rec, log_path)
        pbar.set_postfix(loss=f"{sft_sum:.4f}", lr=f"{lr:.2e}", skip=stepper.n_skipped)

        if (u + 1) in tc["checkpoint_updates"]:
            ckpt_dir = run_dir / f"ckpt_{u + 1}"
            ckpt_dir.mkdir(exist_ok=True)
            summary.setdefault("export_s", {})[str(u + 1)] = export_checkpoint(model, mcfg, ckpt_dir, tok)
            if mcfg["kind"] == "full":
                save_file(diag_tensors(model, cfg), ckpt_dir / "diag.safetensors")
                summary.setdefault("displacement_l2", {})[str(u + 1)] = displacement(model, ref_cpu)
            summary.setdefault("checkpoint_bytes", {})[str(u + 1)] = dir_size_bytes(ckpt_dir)
            model_state = (model.state_dict() if mcfg["kind"] == "full"
                           else {k: v for k, v in L.adapter_state_dict(model, projected=False).items()})
            summary["peak_mem_gb"] = max(summary.get("peak_mem_gb", 0.0), peak_memory_gb(device))
            summary["skipped_updates"] = stepper.n_skipped
            save_resume(ckpt_dir, model_state=model_state, optimizer=opt, scheduler=sched, stepper=stepper,
                        update=u + 1, data_pos=(u + 1) * eff,
                        extra={"elapsed": timer.elapsed(), "summary": summary})
            summary["checkpoint_bytes_with_resume"] = summary.get("checkpoint_bytes_with_resume", {})
            summary["checkpoint_bytes_with_resume"][str(u + 1)] = dir_size_bytes(ckpt_dir)
            prune_resume_states(run_dir, keep=ckpt_dir)

    summary.update({"updates_done": end, "train_time_s": timer.elapsed(),
                    "peak_mem_gb": max(summary.get("peak_mem_gb", 0.0), peak_memory_gb(device)),
                    "skipped_updates": stepper.n_skipped})
    if lam > 0:
        summary["final_omega"] = rec["omega"]
    save_json(summary, rd / f"{args.method}_summary.json")
    print(json.dumps({k: summary[k] for k in ("train_time_s", "peak_mem_gb", "skipped_updates")}))


if __name__ == "__main__":
    main()
