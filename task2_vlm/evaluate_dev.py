"""Teacher-forced development metrics (also reused by the final evaluation).

* caption reference loss: example-weighted CE on caption + <|im_end|> (training loss definition)
* answer NLL: mean token NLL of the reference answer, termination excluded, per example then
  averaged over examples; under correct / complementary (partner) / no-image inputs
* dNLL = NLL(partner) - NLL(correct)

python -m task2_vlm.evaluate_dev --config configs/task2_vlm.yaml --checkpoint <dir>
"""
import argparse

import numpy as np
import torch

from common.training import supervised_token_nll

CONDITIONS = ("correct", "swap", "mismatch", "noimage")


def image_for(pairs, p_idx, side, condition, mismatch=None):
    p = pairs[p_idx]
    if condition == "correct":
        return p["examples"][side]["image"]
    if condition == "swap":
        return p["examples"][1 - side]["image"]
    if condition == "mismatch":
        return pairs[mismatch[p_idx]]["examples"][side]["image"]
    return None


def flat_examples(pairs, condition, mismatch=None):
    out = []
    for pi, p in enumerate(pairs):
        for side, ex in enumerate(p["examples"]):
            out.append((pi, side, ex, image_for(pairs, pi, side, condition, mismatch)))
    return out


@torch.no_grad()
def per_example_nll(vlm, examples, images, store, target_mode, bs, autocast):
    """Mean token NLL per example (FP32 log-softmax)."""
    vals = []
    for s in range(0, len(examples), bs):
        exs, ims = examples[s:s + bs], images[s:s + bs]
        use = [im is not None for im in ims]
        feats = store.get([im for im in ims if im is not None]) if any(use) else None
        if feats is not None and not all(use):  # mixed batch: place features at the right rows
            full = torch.zeros(len(exs), *feats.shape[1:], dtype=feats.dtype)
            full[torch.tensor(use)] = feats
            feats = full
        with autocast():
            b = vlm.build(exs, feats, use, target_mode=target_mode)
            h = vlm.hidden(b)
            nll, idx = supervised_token_nll(h, b["labels"], vlm.lm.lm_head)
        sums = torch.zeros(len(exs), device=nll.device).index_add_(0, idx, nll)
        cnts = torch.zeros(len(exs), device=nll.device).index_add_(0, idx, torch.ones_like(nll))
        vals.extend((sums / cnts).cpu().tolist())
    return vals


def answer_nll(vlm, pairs, condition, store, bs, autocast, mismatch=None):
    flat = flat_examples(pairs, condition, mismatch)
    return per_example_nll(vlm, [f[2] for f in flat], [f[3] for f in flat], store, "nll", bs, autocast), flat


def dev_metrics(vlm, dev_caps, suites, store, bs, autocast):
    vlm.eval()
    out = {"caption_loss": float(np.mean(per_example_nll(
        vlm, dev_caps, [c["image"] for c in dev_caps], store, "train", bs, autocast)))}
    for name, pairs in suites.items():
        res = {}
        for cond in ("correct", "swap", "noimage"):
            v, flat = answer_nll(vlm, pairs, cond, store, bs, autocast)
            res[f"nll_{cond}"] = float(np.mean(v))
            res[f"_{cond}"] = np.array(v)
            types = np.array([pairs[f[0]]["type"] for f in flat])
        res["dnll_swap"] = res["nll_swap"] - res["nll_correct"]
        res["dnll_noimage"] = res["nll_noimage"] - res["nll_correct"]
        res["by_type"] = {t: {"nll_correct": float(res["_correct"][types == t].mean()),
                              "dnll_swap": float((res["_swap"] - res["_correct"])[types == t].mean())}
                          for t in sorted(set(types))}
        for c in ("correct", "swap", "noimage"):
            del res[f"_{c}"]
        out[name] = res
    vlm.train()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2_vlm.yaml")
    ap.add_argument("--checkpoint", default=None, help="stage weights dir (default: final stage)")
    args = ap.parse_args(argv)
    from common.io import load_yaml, save_json, results_dir
    from task2_vlm.train import build_model, load_stage_weights, final_stage_dir
    from task2_vlm.cache_features import FeatureStore
    from task2_vlm.data import caption_examples, eval_suites
    from common.seeding import get_device
    from common.training import AmpStepper
    cfg = load_yaml(args.config)
    device = get_device(cfg["device"])
    vlm = build_model(cfg, device, with_lora=True)
    load_stage_weights(vlm, args.checkpoint or final_stage_dir(cfg))
    store = FeatureStore(cfg)
    ac = AmpStepper(torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0), [], device).autocast
    m = dev_metrics(vlm, caption_examples(cfg, "dev"), eval_suites(cfg, "dev"), store,
                    cfg["dev_eval"]["batch_size"], ac)
    save_json(m, results_dir(cfg) / f"{cfg['run_name']}_dev_standalone.json")
    print(m)


if __name__ == "__main__":
    main()
