"""Live MATH experiment (Task 3, Step 4) on the fixed 50-question subset.

Strategies, each with a 3,072-token output allowance:
  single    one response, cap 3,072
  parallel  three independent responses, cap 1,024 each; self-consistency (ties: earliest)
  refine    initial = parallel response #1 (shared), then one critique-and-revision response
            (cap 2,048) conditioned on [system, user, assistant(initial), user(revision prompt)]

Batch size 1, FP16, sampling T=0.7, top-p 0.8, top-k 20, repetition penalty 1.0. Every call
has its own seed. Resumable: finished questions in the per-example JSONL are skipped.

python -m task3_ttc.live_generation --config configs/task3_ttc.yaml
"""
import argparse
import hashlib
import time

import numpy as np
import torch
from tqdm import tqdm

from common.io import load_yaml, read_jsonl, append_jsonl, results_dir, save_json
from common.resources import peak_memory_gb, reset_peak_memory, sync
from common.seeding import get_device, set_seed
from task3_ttc import answers as A
from task3_ttc.answers import last_boxed
from task3_ttc.data import load_questions, load_splits


def call_seed(base, qid, strategy, k):
    h = hashlib.sha1(f"{base}|{qid}|{strategy}|{k}".encode()).hexdigest()
    return int(h[:8], 16)


@torch.no_grad()
def generate(model, tok, messages, cap, seed, lc, device):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    sync(device)
    t0 = time.perf_counter()
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    out = model.generate(**enc, do_sample=True, temperature=lc["temperature"], top_p=lc["top_p"],
                         top_k=lc["top_k"], repetition_penalty=lc["repetition_penalty"],
                         max_new_tokens=cap, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    sync(device)
    dt = time.perf_counter() - t0
    gen = out[0, enc["input_ids"].shape[1]:].tolist()
    eos = set([tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|im_end|>")]))
    ended = len(gen) > 0 and gen[-1] in eos
    n_out = len(gen) - int(ended)  # termination token excluded from the output count
    text = tok.decode(gen, skip_special_tokens=True)
    ans = last_boxed(text)
    return {"prompt": prompt, "response": text, "seed": seed, "cap": cap,
            "input_tokens": int(enc["input_ids"].shape[1]), "output_tokens": int(n_out),
            "truncated": bool(not ended and len(gen) >= cap), "extracted": ans,
            "time_s": dt}


def sc_pick(answers):
    keys = [A.NO_ANSWER if A.is_missing(a) else a for a in answers]
    E = A.equivalence_matrix(keys, "math")
    gid = A.group_prefix(np.arange(len(keys)), E)
    votes = np.bincount(gid)
    g = int(np.flatnonzero(votes == votes.max())[0])
    pick = keys[int(np.flatnonzero(gid == g)[0])]
    return None if pick == A.NO_ANSWER else pick   # missing-answer group won: no answer (not "extracted")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3_ttc.yaml")
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first k questions")
    ap.add_argument("--fresh", action="store_true", help="discard existing checkpoints/outputs of this run and start over")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    lc = cfg["live"]
    rd = results_dir(cfg)
    out_path = rd / "live_per_example.jsonl"
    if args.fresh and out_path.exists():
        out_path.unlink()
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(lc["model_id"], revision=lc["revision"])
    model = AutoModelForCausalLM.from_pretrained(lc["model_id"], revision=lc["revision"],
                                                 dtype=torch.float16).to(device).eval()
    qs = load_questions(cfg, "math")
    live_ids = load_splits(cfg)["math"]["live"][: args.limit]
    done = {r["qid"] for r in read_jsonl(out_path)}
    for qid in tqdm(live_ids, desc="live MATH"):
        if qid in done:
            continue
        q = qs[qid]
        user = q.question + lc["join"] + lc["suffix"]
        base_msgs = [{"role": "system", "content": lc["system"]}, {"role": "user", "content": user}]
        rec = {"qid": qid, "gold": q.gold}
        # single
        reset_peak_memory(device)
        single = generate(model, tok, base_msgs, lc["single_cap"], call_seed(cfg["seed"], qid, "single", 0), lc, device)
        rec["single"] = {"calls": [single], "answer": single["extracted"], "peak_mem_gb": peak_memory_gb(device)}
        # parallel
        reset_peak_memory(device)
        par = [generate(model, tok, base_msgs, lc["parallel_cap"], call_seed(cfg["seed"], qid, "parallel", k), lc, device)
               for k in range(lc["parallel_n"])]
        rec["parallel"] = {"calls": par, "answer": sc_pick([c["extracted"] for c in par]),
                           "peak_mem_gb": peak_memory_gb(device)}
        # refinement: shared initial response = parallel #1
        reset_peak_memory(device)
        init = par[0]
        msgs = base_msgs + [{"role": "assistant", "content": init["response"]},
                            {"role": "user", "content": lc["revision_prompt"]}]
        rev = generate(model, tok, msgs, lc["refine_cap"], call_seed(cfg["seed"], qid, "refine", 1), lc, device)
        rec["refine"] = {"calls": [init, rev], "initial_answer": init["extracted"], "answer": rev["extracted"],
                         "peak_mem_gb": max(peak_memory_gb(device), rec["parallel"]["peak_mem_gb"])}
        for s in ("single", "parallel", "refine"):
            r = rec[s]
            r["correct"] = bool(A.math_correct(r["answer"], q.gold)) if not A.is_missing(r["answer"]) else False
            r["input_tokens"] = sum(c["input_tokens"] for c in r["calls"])
            r["output_tokens"] = sum(c["output_tokens"] for c in r["calls"])
            r["time_s"] = sum(c["time_s"] for c in r["calls"])
            r["any_truncated"] = any(c["truncated"] for c in r["calls"])
            r["extracted"] = not A.is_missing(r["answer"])
        rec["refine"]["initial_correct"] = (not A.is_missing(init["extracted"])) and bool(A.math_correct(init["extracted"], q.gold))
        append_jsonl(rec, out_path)
    summarise(cfg, out_path, rd)


def summarise(cfg, out_path, rd):
    recs = read_jsonl(out_path)
    n = len(recs)
    out = {"n_questions": n, "scorer": A.scorer_versions(), "settings": cfg["live"]}
    for s in ("single", "parallel", "refine"):
        R = [r[s] for r in recs]
        calls = [c for r in R for c in r["calls"]]
        out[s] = {
            "accuracy": float(np.mean([r["correct"] for r in R])),
            "extraction_rate": float(np.mean([r["extracted"] for r in R])),
            "call_truncation_rate": float(np.mean([c["truncated"] for c in calls])),
            "any_truncation_rate": float(np.mean([r["any_truncated"] for r in R])),
            "mean_input_tokens": float(np.mean([r["input_tokens"] for r in R])),
            "mean_output_tokens": float(np.mean([r["output_tokens"] for r in R])),
            "total_output_tokens": int(sum(r["output_tokens"] for r in R)),
            "mean_time_s": float(np.mean([r["time_s"] for r in R])),
            "total_time_s": float(sum(r["time_s"] for r in R)),
            "peak_mem_gb": float(np.nanmax([r["peak_mem_gb"] for r in R])),
        }
    ic = np.array([r["refine"]["initial_correct"] for r in recs])
    fc = np.array([r["refine"]["correct"] for r in recs])
    out["refine"]["initial_accuracy"] = float(ic.mean())
    out["refine"]["transitions"] = {
        "wrong_to_correct": int((~ic & fc).sum()), "correct_to_wrong": int((ic & ~fc).sum()),
        "correct_to_correct": int((ic & fc).sum()), "wrong_to_wrong": int((~ic & ~fc).sum()),
        "answer_changed": int(sum(r["refine"]["initial_answer"] != r["refine"]["answer"] for r in recs)),
    }
    save_json(out, rd / "live_eval.json")
    print({s: out[s]["accuracy"] for s in ("single", "parallel", "refine")}, out["refine"]["transitions"])


if __name__ == "__main__":
    main()
