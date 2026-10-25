"""Task 2 training: three-stage curriculum on cached SigLIP features.

Stage 1 caption alignment (connector only, 2,250 updates = 2 passes over 18k captions),
Stage 2 visual instruction tuning (+LoRA r16 on q,k,v,o; presence 50 / counting 25 / caption 25),
Stage 3 natural VQA (VQAv2 70 / presence 10 / counting 10 / caption 10).
Micro 8 x accum 2 = 16 examples per update (8 draws of 2). Example-weighted loss.
Per stage: fresh AdamW, 5% warmup then cosine to 10% of peak. Resumable every 250 updates.

python -m task2_vlm.train --config configs/task2_vlm.yaml
"""
import argparse
import math
import time

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from common import lora as L
from common.io import append_jsonl, truncate_jsonl, load_yaml, outputs_dir, read_jsonl, results_dir, save_json
from common.resources import Timer, peak_memory_gb, reset_peak_memory
from common.seeding import get_device, set_seed
from common.training import (AmpStepper, cosine_lambda, example_weighted_loss, latest_checkpoint,
                             load_resume, param_groups, prune_resume_states, restore_rng, save_resume,
                             supervised_token_nll)
from task2_vlm.cache_features import FeatureStore
from task2_vlm.data import MixtureSampler, caption_examples, eval_suites, load_pools
from task2_vlm.evaluate_dev import dev_metrics
from task2_vlm.model import VLM, calibrate_scale


def run_dir(cfg):
    return outputs_dir(cfg) / cfg["run_name"]


def final_stage_dir(cfg):
    return run_dir(cfg) / cfg["train"]["stages"][-1]["name"]


def add_lora(vlm, cfg):
    lc = cfg["model"]["lora"]
    set_seed(cfg["seed"] + 2)
    L.inject_adapters(vlm.lm, lc["targets"], lc["r"], lc["alpha"], lc["dropout"], kind="lora")


def build_model(cfg, device, with_lora=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    lm = AutoModelForCausalLM.from_pretrained(cfg["model_id"], torch_dtype=torch.float16).to(device)
    for p in lm.parameters():
        p.requires_grad_(False)
    set_seed(cfg["seed"])
    vlm = VLM(lm, tok, cfg)
    vlm.connector.to(device=device, dtype=torch.float32)
    if with_lora:
        add_lora(vlm, cfg)
    if cfg["train"]["gradient_checkpointing"]:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lm.config.use_cache = False
    return vlm


def save_stage_weights(vlm, d):
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in vlm.connector.state_dict().items()},
              d / "connector.safetensors")
    ad = L.adapter_state_dict(vlm.lm, projected=False)
    if ad:
        save_file({k: v.contiguous() for k, v in ad.items()}, d / "adapter.safetensors")


def load_stage_weights(vlm, d):
    dev = next(vlm.connector.parameters()).device
    vlm.connector.load_state_dict({k: v.to(dev) for k, v in load_file(d / "connector.safetensors").items()})
    if (d / "adapter.safetensors").exists():
        L.load_adapter_factors(vlm.lm, load_file(d / "adapter.safetensors"))


def trainable(vlm):
    return [(f"connector.{n}", p) for n, p in vlm.connector.named_parameters()] + \
           [(n, p) for n, p in vlm.lm.named_parameters() if p.requires_grad]


def check_frozen(vlm):
    bad = [n for n, p in vlm.lm.named_parameters() if p.requires_grad and "lora_" not in n]
    assert not bad, f"backbone params unexpectedly trainable: {bad[:5]}"
    assert all(p.requires_grad for p in vlm.connector.parameters())


def eval_points(cfg):
    total = sum(s["updates"] for s in cfg["train"]["stages"])
    pts = set(range(0, total + 1, cfg["dev_eval"]["every"])) | set(cfg["dev_eval"]["extra_at"])
    return sorted(p for p in pts if p <= total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2_vlm.yaml")
    ap.add_argument("--max-updates", type=int, default=None, help="stop early (runtime estimate)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    tc = cfg["train"]
    device = get_device(cfg["device"])
    set_seed(cfg["seed"])
    rdir, res = run_dir(cfg), results_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)
    log_path, dev_path = res / f"{cfg['run_name']}_train_log.jsonl", res / f"{cfg['run_name']}_dev_log.jsonl"

    store = FeatureStore(cfg)
    pools = load_pools(cfg)
    dev_caps, dev_suites = caption_examples(cfg, "dev"), eval_suites(cfg, "dev")
    # split isolation: no evaluation image may appear in any training pool
    train_imgs = {c["image"] for c in pools["caption"]}
    for k in ("presence", "counting", "vqa"):
        train_imgs |= {f"train2014:{e['image_id']}" for p in pools[k] for e in p["examples"]}
    eval_imgs = {c["image"] for c in dev_caps} | {e["image"] for s in eval_suites(cfg, "dev").values() for p in s
                                                  for e in p["examples"]} | \
                {e["image"] for s in eval_suites(cfg, "test").values() for p in s for e in p["examples"]}
    assert not (train_imgs & eval_imgs), "evaluation images found in training pools"

    ck = latest_checkpoint(rdir)
    state = load_resume(ck) if ck is not None else None
    start_stage = state["extra"]["stage"] if state else 0
    vlm = build_model(cfg, device, with_lora=start_stage >= 1)
    summary = state["extra"]["summary"] if state else {"run": cfg["run_name"], "checks": {}, "stages": {}}
    if state is None:
        if log_path.exists():
            log_path.unlink()
        if dev_path.exists():
            dev_path.unlink()
        if cfg["model"]["calibrate_scale"]:
            n = cfg["model"]["calibration_images"]
            caps = pools["caption"][: 3 * n: 3]
            texts = [c["target"] for c in caps] + [p["question"] for p in pools["presence"][:n]]
            cal = calibrate_scale(vlm, store.get([c["image"] for c in caps]), texts, device)
            summary["calibration"] = cal
            save_json(cal, res / f"{cfg['run_name']}_calibration.json")
            print("[calibration]", cal)
        summary["checks"]["feature_shape"] = list(store.arr.shape[1:])
        summary["checks"]["split_isolation"] = True
    else:
        truncate_jsonl(log_path, "update", state["update"])
        truncate_jsonl(dev_path, "update", state["update"])
        vlm.connector.load_state_dict({k: v.to(device) for k, v in state["model"]["connector"].items()})
        if state["model"].get("adapter"):
            L.load_adapter_factors(vlm.lm, state["model"]["adapter"])
    dev_done = {r["update"] for r in read_jsonl(dev_path)}
    sampler = MixtureSampler(pools, cfg, cfg["seed"])
    if state:
        sampler.load_state_dict(state["extra"]["sampler"])

    ac = AmpStepper(torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0), [], device).autocast
    pts = eval_points(cfg)

    def maybe_dev(g):
        if g in pts and g not in dev_done:
            m = dev_metrics(vlm, dev_caps, dev_suites, store, cfg["dev_eval"]["batch_size"], ac)
            append_jsonl({"update": g, **m}, dev_path)
            dev_done.add(g)
            tqdm.write(f"[dev @{g}] caption {m['caption_loss']:.3f} | grounding dNLL {m['grounding']['dnll_swap']:.3f}"
                       f" | vqa dNLL {m['vqa']['dnll_swap']:.3f}")

    timer = Timer()
    if state:
        timer.elapsed_before = state["extra"]["elapsed"]
    g0 = 0
    budget = args.max_updates
    for si, st in enumerate(tc["stages"]):
        if si < start_stage:
            g0 += st["updates"]
            continue
        if si >= 1 and not L.adapter_modules(vlm.lm):
            add_lora(vlm, cfg)
        if L.adapter_modules(vlm.lm):
            vlm.lm.enable_input_require_grads()
        vlm.train()
        check_frozen(vlm)
        conn = [(f"connector.{n}", p) for n, p in vlm.connector.named_parameters()]
        lora = [(n, p) for n, p in vlm.lm.named_parameters() if p.requires_grad]
        groups = param_groups(conn, tc["weight_decay"], lr=st["connector_lr"])
        if st["lora_lr"] is not None:
            groups += param_groups(lora, tc["weight_decay"], lr=st["lora_lr"])
        else:
            for _, p in lora:
                p.requires_grad_(False)
        opt = torch.optim.AdamW(groups, lr=st["connector_lr"])
        warm = math.ceil(tc["warmup_frac"] * st["updates"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lambda(st["updates"], warm, tc["min_lr_ratio"]))
        params = [p for g in groups for p in g["params"]]
        stepper = AmpStepper(opt, params, device, tc["clip_norm"])
        n_train = sum(p.numel() for p in params)
        summary["stages"].setdefault(st["name"], {})["trainable_params"] = n_train
        start_u = 0
        if state and si == start_stage:
            start_u = state["extra"]["stage_update"]
            if start_u > 0:  # mid-stage; at a stage boundary the new stage starts a fresh optimiser
                opt.load_state_dict(state["optimizer"])
                sched.load_state_dict(state["scheduler"])
                stepper.load_state_dict(state["stepper"])
            restore_rng(state)
            state = None
        if start_u == 0:
            set_seed(cfg["seed"] + 100 + si)   # same RNG stream at every stage start, resumed or not
            reset_peak_memory(device)
            summary["stages"][st["name"]]["start_elapsed"] = timer.elapsed()
        maybe_dev(g0 + start_u)
        mb, acc = tc["micro_batch"], tc["grad_accum"]
        pbar = tqdm(range(start_u, st["updates"]), desc=st["name"], initial=start_u, total=st["updates"])
        for u in pbar:
            if budget is not None and budget <= 0:
                break
            exs = sampler.batch(mb * acc, st["mixture"], sequential_captions=(si == 0))
            loss_tot = 0.0
            for m in range(acc):
                part = exs[m * mb:(m + 1) * mb]
                feats = store.get([e["image"] for e in part])
                with stepper.autocast():
                    b = vlm.build(part, feats, [True] * len(part), target_mode="train")
                    h = vlm.hidden(b)
                    nll, idx = supervised_token_nll(h, b["labels"], vlm.lm.lm_head)
                    loss = example_weighted_loss(nll, idx, len(part), mb * acc)
                stepper.backward(loss)
                loss_tot += loss.item()
            if u == start_u:  # connector must receive gradients through the frozen LM
                gn = vlm.connector.fc1.weight.grad
                ok = gn is not None and gn.abs().sum().item() > 0
                summary["checks"][f"connector_grad_{st['name']}"] = ok
                assert ok, "connector received no gradient"
            lrs = sched.get_last_lr()
            gnorm, skipped = stepper.step()
            sched.step()
            g = g0 + u + 1
            append_jsonl({"update": g, "stage": st["name"], "loss": loss_tot, "lr_connector": lrs[0],
                          "lr_lora": lrs[-1] if st["lora_lr"] is not None else None, "grad_norm": gnorm,
                          "skipped": skipped, "mixture_counts": dict(sampler.counts),
                          "elapsed_s": timer.elapsed()}, log_path)
            pbar.set_postfix(loss=f"{loss_tot:.3f}", skip=stepper.n_skipped)
            maybe_dev(g)
            if u + 1 == st["updates"]:
                sd = summary["stages"][st["name"]]
                sd["time_s"] = timer.elapsed() - sd.get("start_elapsed", 0.0)
                sd["peak_mem_gb"] = max(sd.get("peak_mem_gb", 0.0), peak_memory_gb(device))
                sd["skipped_updates"] = stepper.n_skipped
                sd["mixture_counts_cumulative"] = dict(sampler.counts)
                save_stage_weights(vlm, rdir / st["name"])
            if g % tc["checkpoint_every"] == 0:
                cd = rdir / f"ckpt_{g}"
                summary["stages"][st["name"]]["peak_mem_gb"] = max(
                    summary["stages"][st["name"]].get("peak_mem_gb", 0.0), peak_memory_gb(device))
                save_resume(cd, model_state={"connector": {k: v.cpu() for k, v in vlm.connector.state_dict().items()},
                                             "adapter": L.adapter_state_dict(vlm.lm, projected=False)},
                            optimizer=opt, scheduler=sched, stepper=stepper, update=g, data_pos=g * mb * acc,
                            extra={"stage": si if u + 1 < st["updates"] else si + 1,
                                   "stage_update": (u + 1) % st["updates"], "sampler": sampler.state_dict(),
                                   "elapsed": timer.elapsed(), "summary": summary})
                prune_resume_states(rdir, keep=cd)
            if budget is not None:
                budget -= 1
        else:
            g0 += st["updates"]
            continue
        break   # --max-updates reached
    summary["train_time_s"] = timer.elapsed()
    save_json(summary, res / f"{cfg['run_name']}_summary.json")
    print("done", {k: v.get("time_s") for k, v in summary["stages"].items()})


if __name__ == "__main__":
    main()
