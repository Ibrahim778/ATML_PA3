"""Final evaluation on the test suites under four image conditions.

Greedy decoding, 16-token cap, batch 8, thinking disabled. Conditions: correct, complementary
swap (partner image), mismatched (fixed derangement over pairs; excludes both pair images),
no image (visual tokens omitted; masks/positions rebuilt). Scoring: grounding -> normalised
EM (presence / counting); VQAv2 -> local leave-one-out soft accuracy (yes/no / number / other).
Per suite x condition: accuracy, pair accuracy, identical-answer rate, empty-answer rate, mean
reference NLL, generation time per answer; correct-vs-swap advantages; 95% CIs from 4,000
bootstrap resamples of complete pairs (same resampled pair IDs across conditions).

python -m task2_vlm.evaluate_final --config configs/task2_vlm.yaml
"""
import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

from common.io import load_json, load_yaml, results_dir, save_json, write_jsonl
from common.metrics import bootstrap_indices, ci_from_indices
from common.resources import peak_memory_gb, reset_peak_memory, sync
from common.seeding import get_device, set_seed
from common.training import AmpStepper
from task2_vlm.cache_features import FeatureStore
from task2_vlm.data import eval_suites
from task2_vlm.evaluate_dev import CONDITIONS, answer_nll, flat_examples
from task2_vlm.vqa_norm import normalize_answer, soft_accuracy


def generate_all(vlm, flat, store, bs, max_new, autocast, device):
    preds, times = [], []
    for s in tqdm(range(0, len(flat), bs), desc="generate", leave=False):
        chunk = flat[s:s + bs]
        exs, ims = [f[2] for f in chunk], [f[3] for f in chunk]
        use = [im is not None for im in ims]
        feats = store.get([im for im in ims if im is not None]) if any(use) else None
        sync(device)
        t0 = time.perf_counter()
        with autocast():
            out = vlm.generate(exs, feats, use, max_new_tokens=max_new)
        sync(device)
        dt = (time.perf_counter() - t0) / len(chunk)
        preds += out
        times += [dt] * len(chunk)
    return preds, times


def score_example(ex, pred):
    p = normalize_answer(pred)
    canon = ex.get("canonical", normalize_answer(ex["target"]))
    if ex["kind"] == "vqa":
        acc = soft_accuracy(pred, ex["answers"])
    else:
        acc = float(p == normalize_answer(ex["target"]))
    return {"pred_norm": p, "canonical": canon, "acc": acc, "match_canonical": p == canon, "empty": p == ""}


def pair_table(rows, n_pairs):
    """Per-pair arrays for one suite x condition."""
    acc = np.zeros(n_pairs); pair = np.zeros(n_pairs); ident = np.zeros(n_pairs)
    empty = np.zeros(n_pairs); nll = np.zeros(n_pairs); t = np.zeros(n_pairs)
    by = {}
    for r in rows:
        by.setdefault(r["pair_index"], []).append(r)
    for i in range(n_pairs):
        a, b = sorted(by[i], key=lambda r: r["side"])
        acc[i] = (a["acc"] + b["acc"]) / 2
        pair[i] = float(a["match_canonical"] and b["match_canonical"])
        ident[i] = float(a["pred_norm"] == b["pred_norm"])
        empty[i] = (a["empty"] + b["empty"]) / 2
        nll[i] = (a["nll"] + b["nll"]) / 2
        t[i] = (a["gen_time_s"] + b["gen_time_s"]) / 2
    return {"accuracy": acc, "pair_accuracy": pair, "identical_rate": ident, "empty_rate": empty,
            "ref_nll": nll, "gen_time_per_answer_s": t}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2_vlm.yaml")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    fe = cfg["final_eval"]
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    from task2_vlm.train import build_model, final_stage_dir, load_stage_weights
    vlm = build_model(cfg, device, with_lora=True)
    load_stage_weights(vlm, args.checkpoint or final_stage_dir(cfg))
    vlm.eval()
    vlm.lm.config.use_cache = True
    store = FeatureStore(cfg)
    ac = AmpStepper(torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0), [], device).autocast
    suites = eval_suites(cfg, "test")
    mm = load_json(cfg["data"]["ids"]["mismatch"])
    rd = results_dir(cfg)
    out, per_ex = {"run": cfg["run_name"], "n_bootstrap": fe["bootstrap"]}, []
    reset_peak_memory(device)
    for suite, pairs in suites.items():
        mapping = mm[f"{suite}_test"]
        idx = bootstrap_indices(len(pairs), fe["bootstrap"], cfg["seed"])
        types = np.array([p["type"] for p in pairs])
        tables, sres = {}, {"n_pairs": len(pairs), "conditions": {}}
        for cond in CONDITIONS:
            flat = flat_examples(pairs, cond, mapping)
            preds, times = generate_all(vlm, flat, store, fe["batch_size"], fe["max_new_tokens"], ac, device)
            nlls, _ = answer_nll(vlm, pairs, cond, store, fe["batch_size"], ac, mapping)
            rows = []
            for (pi, side, ex, im), pred, dt, nl in zip(flat, preds, times, nlls):
                partner = pairs[pi]["examples"][1 - side]
                sc = score_example(ex, pred)
                rows.append({"suite": suite, "condition": cond, "pair_id": pairs[pi]["pair_id"], "pair_index": pi,
                             "side": side, "type": pairs[pi]["type"], "question": ex["question"],
                             "image": im, "reference": ex["target"], "answers": ex.get("answers"),
                             "prediction": pred, **sc, "nll": nl, "gen_time_s": dt,
                             "follows_partner": sc["pred_norm"] == partner.get("canonical", normalize_answer(partner["target"]))})
            per_ex += rows
            tab = pair_table(rows, len(pairs))
            tables[cond] = tab
            cres = {k: dict(zip(("mean", "lo", "hi"), ci_from_indices(v, idx))) for k, v in tab.items()}
            cres["by_type"] = {}
            for t in sorted(set(types)):
                sub = np.flatnonzero(types == t)
                tidx = bootstrap_indices(len(sub), fe["bootstrap"], cfg["seed"])
                cres["by_type"][t] = {k: dict(zip(("mean", "lo", "hi"), ci_from_indices(v[sub], tidx)))
                                      for k, v in tab.items() if k in ("accuracy", "pair_accuracy", "ref_nll")}
            r = [x for x in rows]
            cres["follows_partner_rate"] = float(np.mean([x["follows_partner"] for x in r]))
            cres["matches_original_rate"] = float(np.mean([x["match_canonical"] for x in r]))
            sres["conditions"][cond] = cres
        adv = {}
        for other in ("swap", "mismatch", "noimage"):
            da = tables["correct"]["accuracy"] - tables[other]["accuracy"]
            dn = tables[other]["ref_nll"] - tables["correct"]["ref_nll"]
            adv[f"correct_minus_{other}"] = {
                "accuracy": dict(zip(("mean", "lo", "hi"), ci_from_indices(da, idx))),
                "nll_other_minus_correct": dict(zip(("mean", "lo", "hi"), ci_from_indices(dn, idx)))}
        sres["advantages"] = adv
        out[suite] = sres
    out["peak_mem_gb"] = peak_memory_gb(device)
    save_json(out, rd / f"{cfg['run_name']}_final_eval.json")
    write_jsonl(per_ex, rd / f"{cfg['run_name']}_final_per_example.jsonl")
    for s in suites:
        print(s, {c: round(out[s]["conditions"][c]["accuracy"]["mean"], 3) for c in CONDITIONS})


if __name__ == "__main__":
    main()
