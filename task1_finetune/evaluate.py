"""lm-evaluation-harness wrapper for the seven Task 1 conditions.

* GSM8K: gsm8k_cot, 0-shot, chat template with thinking disabled, greedy, 1,024-token cap,
  stock stop sequences and filters. Primary: exact_match / flexible-extract.
* HellaSwag (acc_norm), BoolQ (acc), ARC-Easy (acc_norm): 0-shot, no chat template.

`enable_thinking=False` is enforced by overriding HFLM.apply_chat_template (version-independent),
and verified on the logged GSM8K prompts (they must end with the empty think block).

python -m task1_finetune.evaluate --config configs/task1_ft.yaml --condition baseline
"""
import argparse
import importlib.metadata as md

from tqdm import tqdm

from common.io import load_json, load_yaml, results_dir, save_json, write_jsonl
from common.seeding import get_device, set_seed
from task1_finetune.conditions import condition_model_path

EMPTY_THINK = "<think>\n\n</think>"


def make_lm(path, cfg, batch_size, device):
    from lm_eval.models.huggingface import HFLM

    class HFLMNoThink(HFLM):
        def apply_chat_template(self, chat_history, add_generation_prompt=True, **kw):
            return self.tokenizer.apply_chat_template(
                chat_history, tokenize=False, add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt, enable_thinking=False)

    return HFLMNoThink(pretrained=path, dtype=cfg["eval"]["dtype"], batch_size=batch_size, device=str(device))


def _context(sample):
    a = sample.get("arguments")
    if isinstance(a, dict):
        first = next(iter(a.values()))
        return first.get("arg_0") if isinstance(first, dict) else first[0]
    if isinstance(a, (list, tuple)) and a:
        return a[0][0] if isinstance(a[0], (list, tuple)) else a[0]
    return None


def _first(x):
    while isinstance(x, (list, tuple)) and x:
        x = x[0]
    return x


def run_gsm8k(lm, cfg, tok):
    from lm_eval import simple_evaluate
    g = cfg["eval"]["gsm8k"]
    res = simple_evaluate(model=lm, tasks=[g["task"]], num_fewshot=g["num_fewshot"],
                          apply_chat_template=g["apply_chat_template"], fewshot_as_multiturn=False,
                          gen_kwargs=f"max_gen_toks={g['max_gen_toks']},do_sample=False,temperature=0.0",
                          log_samples=True, random_seed=cfg["seed"], numpy_random_seed=cfg["seed"],
                          torch_random_seed=cfg["seed"], fewshot_random_seed=cfg["seed"])
    task = g["task"]
    r = res["results"][task]
    by_doc = {}
    for s in res["samples"][task]:
        d = by_doc.setdefault(s["doc_id"], {"doc_id": s["doc_id"], "question": s["doc"]["question"],
                                            "target": s["target"], "prompt": _context(s)})
        resp = _first(s["resps"])
        d["response"] = resp
        filt = s.get("filter", "none")
        d[f"extracted_{filt}"] = _first(s["filtered_resps"])
        d[f"exact_match_{filt}"] = float(s.get("exact_match", 0.0))   # metric value is a top-level key
    rows = sorted(by_doc.values(), key=lambda x: x["doc_id"])
    cap = g["max_gen_toks"]
    for d in rows:
        n = len(tok(d["response"] or "", add_special_tokens=False)["input_ids"])
        d["n_tokens"] = n
        d["truncated"] = n >= cap
        d["extractable_flexible"] = d.get(f"extracted_{g['primary_filter']}") not in (None, "", "[invalid]")
    assert all(d["prompt"] is None or EMPTY_THINK in d["prompt"] for d in rows[:5]), \
        "GSM8K prompts are missing the empty think block (enable_thinking=False not applied)"
    pf = g["primary_filter"]
    agg = {
        "primary": r[f"exact_match,{pf}"], "primary_stderr": r.get(f"exact_match_stderr,{pf}"),
        "strict_match": r.get("exact_match,strict-match"),
        "n": len(rows), "truncation_rate": sum(d["truncated"] for d in rows) / len(rows),
        "unextractable_rate": 1 - sum(d["extractable_flexible"] for d in rows) / len(rows),
        "mean_response_tokens": sum(d["n_tokens"] for d in rows) / len(rows),
        "example_prompt": rows[0]["prompt"],
    }
    return agg, rows


def run_likelihood(lm, cfg, task, spec):
    from lm_eval import simple_evaluate
    res = simple_evaluate(model=lm, tasks=[task], num_fewshot=0, apply_chat_template=False,
                          log_samples=True, random_seed=cfg["seed"], numpy_random_seed=cfg["seed"],
                          torch_random_seed=cfg["seed"], fewshot_random_seed=cfg["seed"])
    r = res["results"][task]
    rows = []
    for s in res["samples"][task]:
        ll = [float(_first(x)) for x in s["resps"]]
        rows.append({"doc_id": s["doc_id"], "target": s["target"], "loglikelihoods": ll,
                     "acc": s.get("acc"), "acc_norm": s.get("acc_norm")})
    rows.sort(key=lambda x: x["doc_id"])
    agg = {"primary": r[f"{spec['metric']},none"], "primary_stderr": r.get(f"{spec['metric']}_stderr,none"),
           "raw_acc": r.get("acc,none"), "raw_acc_stderr": r.get("acc_stderr,none"),
           "primary_metric": spec["metric"], "n": len(rows)}
    return agg, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1_ft.yaml")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of gsm8k hellaswag boolq arc_easy")
    ap.add_argument("--fresh", action="store_true", help="discard existing checkpoints/outputs of this run and start over")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    rd = results_dir(cfg)
    path = condition_model_path(cfg, args.condition)
    out_json = rd / f"{args.condition}_eval.json"
    agg_all = load_json(out_json) if out_json.exists() and not args.fresh else {"condition": args.condition, "model_path": path}
    agg_all["lm_eval_version"] = md.version("lm_eval")
    agg_all["transformers_version"] = md.version("transformers")
    tasks = args.tasks or ["gsm8k"] + list(cfg["eval"]["likelihood_tasks"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    for t in tqdm(tasks, desc=f"eval {args.condition}"):
        if t in agg_all.get("benchmarks", {}):
            print(f"skip {t} (done)")
            continue
        if t == "gsm8k":
            lm = make_lm(path, cfg, cfg["eval"]["gen_batch_size"], device)
            agg, rows = run_gsm8k(lm, cfg, tok)
        else:
            lm = make_lm(path, cfg, cfg["eval"]["batch_size"], device)
            agg, rows = run_likelihood(lm, cfg, t, cfg["eval"]["likelihood_tasks"][t])
        del lm
        write_jsonl(rows, rd / f"{args.condition}_per_example_{t}.jsonl")
        agg_all.setdefault("benchmarks", {})[t] = agg
        save_json(agg_all, out_json)
    base = rd / "baseline_eval.json"
    if args.condition != "baseline" and base.exists():
        b = load_json(base)["benchmarks"]
        agg_all["delta_vs_baseline_pp"] = {t: 100 * (v["primary"] - b[t]["primary"])
                                           for t, v in agg_all["benchmarks"].items() if t in b}
    save_json(agg_all, out_json)
    print({t: v["primary"] for t, v in agg_all["benchmarks"].items()})


if __name__ == "__main__":
    main()
