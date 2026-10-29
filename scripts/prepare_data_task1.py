"""Task 1 data selection: first 40,000 eligible MetaMathQA examples in source order, plus the
activation-probe set (32 GSM8K test + 32 HellaSwag validation, seed 6304).

Exclusions: empty query/response; formatted length > 2,048 tokens; query or
original_question equal to a GSM8K test question after normalisation (NFKC, lower-case,
non-alphanumerics -> space, collapsed whitespace).

python -m scripts.prepare_data_task1 --config configs/task1_ft.yaml
"""
import argparse
import re
import unicodedata

import numpy as np
from tqdm import tqdm

from common.chat import build_sft_example
from common.io import load_yaml, save_json


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = re.sub(r"[^0-9a-z]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1_ft.yaml")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    d = cfg["data"]
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])

    gsm = load_dataset(d["gsm8k_id"], d["gsm8k_config"], split="test")
    gsm_norm = {normalise(x) for x in gsm["question"]}
    mm = load_dataset(d["dataset_id"], split=d["split"])

    kept, excl = [], {"empty": 0, "too_long": 0, "gsm8k_overlap": 0}
    for i, ex in enumerate(tqdm(mm, desc="MetaMathQA filter")):
        q, r = ex.get("query") or "", ex.get("response") or ""
        if not q.strip() or not r.strip():
            excl["empty"] += 1
            continue
        if normalise(q) in gsm_norm or normalise(ex.get("original_question") or "") in gsm_norm:
            excl["gsm8k_overlap"] += 1
            continue
        if build_sft_example(tok, q, r, max_length=d["max_length"]) is None:
            excl["too_long"] += 1
            continue
        kept.append(i)
        if len(kept) == d["n_examples"]:
            break
    assert len(kept) == d["n_examples"], f"only {len(kept)} eligible examples"
    save_json({"dataset": d["dataset_id"], "split": d["split"], "source_indices": kept,
               "scanned_until": kept[-1], "excluded_before_cutoff": excl,
               "normalisation": "NFKC, lowercase, [^0-9a-z]+ -> ' ', collapse whitespace"},
              d["ids_file"], indent=None)
    print("kept", len(kept), "last source index", kept[-1], "excluded", excl)

    # activation probe set (sampled before looking at any results)
    rng = np.random.default_rng(cfg["seed"])
    hs = load_dataset("Rowan/hellaswag", split="validation")
    dg = cfg["diagnostics"]
    probe = {
        "gsm8k_test": sorted(rng.choice(len(gsm), dg["probe_n_gsm8k"], replace=False).tolist()),
        "hellaswag_validation": sorted(rng.choice(len(hs), dg["probe_n_hellaswag"], replace=False).tolist()),
        "seed": cfg["seed"],
    }
    save_json(probe, dg["probe_ids_file"])
    print("probe ids ->", dg["probe_ids_file"])


if __name__ == "__main__":
    main()
