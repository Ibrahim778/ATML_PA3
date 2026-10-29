"""Task 1 qualitative comparison: 24 prompts (8 questions x plain/JSON/template) x 7 conditions
= 168 greedy responses (512-token cap, thinking disabled), plus the 42-row annotation sheet
(first and last plain-order questions x 3 formats x 7 conditions).

The annotation sheet carries automatic flags (truncation, JSON validity, repetition, math
markers) and empty columns for manual judgements.

python -m task1_finetune.qualitative --config configs/task1_ft.yaml
"""
import argparse
import csv
import json
import re

import torch
import yaml
from tqdm import tqdm

from common.chat import render_prompt
from common.io import load_yaml, read_jsonl, repo_path, results_dir, write_jsonl
from common.seeding import get_device, set_seed
from task1_finetune.conditions import condition_model_path

FORMATS = ("plain", "json", "template")


def load_questions(path):
    """Return [(question_index, format, prompt, system)] in file order.

    Course file (first_plot_questions.yaml): a list of 24 entries with `id`, `paraphrases` and,
    for the JSON variants, a `system` prompt. The format comes from the id suffix (`_json`,
    `_template`, none = plain); question_index is the position of the base question in the
    plain-question order. Only the first paraphrase is used (one greedy response per prompt).
    """
    with open(repo_path(path)) as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict):
        data = data.get("questions") or data.get("prompts") or next(v for v in data.values() if isinstance(v, list))
    order, out = [], []
    for e in data:
        qid = str(e["id"])
        fm, base = "plain", qid
        for suffix in ("json", "template"):
            if qid.endswith("_" + suffix):
                fm, base = suffix, qid[: -len(suffix) - 1]
        if base not in order:
            order.append(base)
        prompt = e["paraphrases"][0] if "paraphrases" in e else e.get("prompt", e.get("text"))
        out.append((order.index(base), fm, prompt, e.get("system")))
    assert len(out) == 24, f"expected 24 prompts, found {len(out)}"
    assert sorted(fm for _, fm, _, _ in out) == sorted(FORMATS * 8), "expected 8 questions x 3 formats"
    return out


def repetition_score(text, n=4):
    toks = text.split()
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def json_valid(text):
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    s = m.group(1) if m else text.strip()
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def math_markers(text):
    return len(re.findall(r"\\boxed|\\frac|\$|=|\\\(|The answer is", text))


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1_ft.yaml")
    ap.add_argument("--fresh", action="store_true", help="discard existing checkpoints/outputs of this run and start over")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    qc = cfg["qualitative"]
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    prompts = load_questions(qc["questions_file"])
    rd = results_dir(cfg)
    out_path = rd / "qualitative_responses.jsonl"
    if args.fresh and out_path.exists():
        out_path.unlink()
    done = {(r["condition"], r["question"], r["format"]) for r in read_jsonl(out_path)}
    from transformers import AutoModelForCausalLM, AutoTokenizer
    for cond in cfg["conditions"]:
        if all((cond, qi, fm) in done for qi, fm, _, _ in prompts):
            continue
        path = condition_model_path(cfg, cond)
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(device).eval()
        rows = []
        for qi, fm, p, system in tqdm(prompts, desc=cond):
            if (cond, qi, fm) in done:
                continue
            text = render_prompt(tok, p, system=system, enable_thinking=False)
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
            out = model.generate(**enc, do_sample=False, max_new_tokens=qc["max_new_tokens"],
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gen = out[0, enc["input_ids"].shape[1]:].tolist()
            ended = len(gen) > 0 and gen[-1] in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"))
            rows.append({"condition": cond, "question": qi, "format": fm, "system": system, "prompt": p,
                         "response": tok.decode(gen, skip_special_tokens=True),
                         "n_tokens": len(gen), "truncated": (not ended) and len(gen) >= qc["max_new_tokens"]})
        write_jsonl(rows, out_path, append=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    build_annotation_sheet(cfg, prompts, rd)


def build_annotation_sheet(cfg, prompts, rd):
    rows = read_jsonl(rd / "qualitative_responses.jsonl")
    qs = sorted({qi for qi, _, _, _ in prompts})
    chosen = {qs[0], qs[-1]}
    cond_order = list(cfg["conditions"])
    sheet = [r for r in rows if r["question"] in chosen]
    sheet.sort(key=lambda r: (r["question"], FORMATS.index(r["format"]) if r["format"] in FORMATS else 9,
                              cond_order.index(r["condition"])))
    assert len(sheet) == 42, f"annotation sheet has {len(sheet)} rows (expected 42)"
    with open(rd / "annotation_sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "format", "condition", "n_tokens", "auto_truncated", "auto_json_valid",
                    "auto_repetition_4gram", "auto_math_markers", "format_compliance", "coherence_relevance",
                    "repetition", "unnecessary_math_framing", "inappropriate_content", "excerpt", "notes",
                    "response"])
        for r in sheet:
            w.writerow([r["question"], r["format"], r["condition"], r["n_tokens"], r["truncated"],
                        json_valid(r["response"]) if r["format"] == "json" else "",
                        round(repetition_score(r["response"]), 3), math_markers(r["response"]),
                        "", "", "", "", "", "", "", r["response"]])
    print("annotation sheet ->", rd / "annotation_sheet.csv")


if __name__ == "__main__":
    main()
