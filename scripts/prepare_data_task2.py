"""Task 2 data selection (seed 6304). Writes everything training/evaluation needs to ids/:

  task2_val_partition.json  val2014 images split into disjoint DEV / TEST pools
  task2_captions.json       6,000 train images x 3 captions; 96 dev images x 1 caption
  task2_grounding.json      presence / counting training pools (train2014) + dev 24 / test 64 pairs
  task2_vqa.json            VQAv2 complementary pairs: train 1,536 / dev 32 / test 96 by type
  task2_mismatch_map.json   fixed mismatched-image mapping for every evaluation suite

Rules: training uses train2014 only; dev and test use disjoint halves of val2014; within an
evaluation suite every image is used once. Examples exceeding the content-token caps
(question 80, caption 64, answer 16) are not selected (no truncation).

python -m scripts.prepare_data_task2 --config configs/task2_vlm.yaml
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

from common.io import load_yaml, repo_path, save_json
from task2_vlm.vqa_norm import normalize_answer

PLURALS = {"person": "people", "sheep": "sheep", "mouse": "mice", "knife": "knives", "bus": "buses",
           "wine glass": "wine glasses", "couch": "couches", "sandwich": "sandwiches", "bench": "benches",
           "toothbrush": "toothbrushes", "skis": "skis", "scissors": "scissors", "broccoli": "broccoli",
           "hair drier": "hair driers", "skateboard": "skateboards", "glass": "glasses"}


def plural(name):
    return PLURALS.get(name, name + "s")


def with_article(name):
    if name in ("skis", "scissors", "broccoli"):
        return name
    return ("an " if name[0] in "aeiou" else "a ") + name


def load(path):
    with open(repo_path(path)) as f:
        return json.load(f)


def category_stats(instances, min_area_frac):
    """Per image: {cat_id: count} over non-crowd instances, and the set of categories that are
    ambiguous (crowd annotation or an instance below min_area_frac of the image)."""
    imgs = {im["id"]: im for im in instances["images"]}
    counts, ambiguous = defaultdict(Counter), defaultdict(set)
    for a in instances["annotations"]:
        im = imgs[a["image_id"]]
        if a.get("iscrowd", 0) or a["area"] < min_area_frac * im["width"] * im["height"]:
            ambiguous[a["image_id"]].add(a["category_id"])
        else:
            counts[a["image_id"]][a["category_id"]] += 1
    return counts, ambiguous


def presence_pairs(rng, image_ids, counts, ambiguous, cats, n, used=None, unique=False):
    image_ids = sorted(image_ids)
    pos = [(i, c) for i in image_ids for c in sorted(counts[i]) if c not in ambiguous[i]]
    neg_by_cat = {c: [i for i in image_ids if c not in counts[i] and c not in ambiguous[i]] for c in cats}
    used = set() if used is None else used
    out, tries = [], 0
    while len(out) < n and tries < 200 * n:
        tries += 1
        i, c = pos[rng.integers(len(pos))]
        j = neg_by_cat[c][rng.integers(len(neg_by_cat[c]))]
        if unique and (i in used or j in used):
            continue
        name = cats[c]
        q = f"Does this image contain {with_article(name)}?"
        pair = [{"image_id": i, "answer": "yes"}, {"image_id": j, "answer": "no"}]
        if rng.random() < 0.5:
            pair = pair[::-1]
        out.append({"kind": "presence", "category": name, "question": q, "examples": pair})
        if unique:
            used.update((i, j))
    return out


def counting_pairs(rng, image_ids, counts, ambiguous, cats, n, lo, hi, used=None, unique=False):
    image_ids = sorted(image_ids)
    by_cat = defaultdict(lambda: defaultdict(list))
    for i in image_ids:
        for c, k in counts[i].items():
            if lo <= k <= hi and c not in ambiguous[i]:
                by_cat[c][k].append(i)
    entries = [(i, c, k) for c in sorted(by_cat) for k in sorted(by_cat[c]) for i in by_cat[c][k]
               if len(by_cat[c]) >= 2]
    used = set() if used is None else used
    out, tries = [], 0
    while len(out) < n and tries < 200 * n:
        tries += 1
        i, c, k = entries[rng.integers(len(entries))]
        others = [kk for kk in sorted(by_cat[c]) if kk != k]
        k2 = others[rng.integers(len(others))]
        j = by_cat[c][k2][rng.integers(len(by_cat[c][k2]))]
        if unique and (i in used or j in used):
            continue
        q = f"How many {plural(cats[c])} can you see?"
        out.append({"kind": "counting", "category": cats[c], "question": q,
                    "examples": [{"image_id": i, "answer": str(k)}, {"image_id": j, "answer": str(k2)}]})
        if unique:
            used.update((i, j))
    return out


def vqa_candidates(qfile, afile, pfile, tok, cap_q, cap_a, min_agree):
    qs = {q["question_id"]: q for q in load(qfile)["questions"]}
    anns = {a["question_id"]: a for a in load(afile)["annotations"]}
    pairs = load(pfile)
    out = defaultdict(list)
    for q1, q2 in tqdm(pairs, desc="VQA pairs"):
        recs = []
        for qid in (q1, q2):
            a = anns[qid]
            raw = [x["answer"] for x in a["answers"]]
            norm = [normalize_answer(x) for x in raw]
            top, cnt = Counter(norm).most_common(1)[0]
            if cnt < min_agree:
                break
            target = Counter(r for r, n in zip(raw, norm) if n == top).most_common(1)[0][0]
            if len(tok(target, add_special_tokens=False)["input_ids"]) > cap_a:
                break
            recs.append({"question_id": qid, "image_id": a["image_id"], "answer": target,
                         "canonical": top, "agreement": cnt, "answers": raw, "answer_type": a["answer_type"]})
        if len(recs) != 2 or recs[0]["canonical"] == recs[1]["canonical"]:
            continue
        if recs[0]["answer_type"] != recs[1]["answer_type"]:
            continue
        question = qs[q1]["question"]
        if question != qs[q2]["question"] or len(tok(question, add_special_tokens=False)["input_ids"]) > cap_q:
            continue
        out[recs[0]["answer_type"]].append({"pair_id": f"{q1}_{q2}", "question": question, "examples": recs})
    return out


def sample_vqa(rng, cands, quota, allowed_images=None, used=None):
    out = []
    used = set() if used is None else used
    for t, n in quota.items():
        pool = [p for p in cands[t] if allowed_images is None or
                all(e["image_id"] in allowed_images for e in p["examples"])]
        idx = rng.permutation(len(pool))
        got = 0
        for i in idx:
            p = pool[i]
            ims = [e["image_id"] for e in p["examples"]]
            if allowed_images is not None and (ims[0] == ims[1] or any(m in used for m in ims)):
                continue
            out.append({**p, "type": t})
            if allowed_images is not None:
                used.update(ims)
            got += 1
            if got == n:
                break
        assert got == n, f"not enough VQA {t} pairs ({got}/{n})"
    return out


def mismatch_map(rng, n_pairs):
    """Derangement of pair indices: example (p, side) gets the image of (perm[p], side)."""
    while True:
        perm = rng.permutation(n_pairs)
        if n_pairs < 2 or not np.any(perm == np.arange(n_pairs)):
            return perm.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2_vlm.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    d, caps, ids = cfg["data"], cfg["caps"], cfg["data"]["ids"]
    rng = np.random.default_rng(cfg["seed"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    coco, vqa = d["coco_dir"], d["vqa_dir"]

    # ---- val2014 partition into disjoint dev / test pools
    cap_val = load(f"{coco}/annotations/captions_val2014.json")
    val_ids = sorted(im["id"] for im in cap_val["images"])
    perm = rng.permutation(len(val_ids))
    half = len(val_ids) // 2
    dev_pool = {val_ids[i] for i in perm[:half]}
    test_pool = {val_ids[i] for i in perm[half:]}
    save_json({"dev_pool": sorted(dev_pool), "test_pool": sorted(test_pool)}, ids["partition"], indent=None)

    # ---- captions
    cap_tr = load(f"{coco}/annotations/captions_train2014.json")

    def caps_by_image(c):
        m = defaultdict(list)
        for a in sorted(c["annotations"], key=lambda a: a["id"]):
            if ntok(a["caption"].strip()) <= caps["caption"]:
                m[a["image_id"]].append({"caption_id": a["id"], "caption": a["caption"].strip()})
        return m
    tr_caps = caps_by_image(cap_tr)
    tr_ok = sorted(i for i, v in tr_caps.items() if len(v) >= d["captions_per_train_image"])
    tr_sel = sorted(rng.choice(tr_ok, d["caption_train_images"], replace=False).tolist())
    va_caps = caps_by_image(cap_val)
    dev_ok = sorted(i for i in dev_pool if va_caps.get(i))
    dev_sel = sorted(rng.choice(dev_ok, d["caption_dev_images"], replace=False).tolist())
    captions = {
        "train": [{"image_id": i, "split": "train2014", **c} for i in tr_sel
                  for c in tr_caps[i][:d["captions_per_train_image"]]],
        "dev": [{"image_id": i, "split": "val2014", **va_caps[i][0]} for i in dev_sel],
    }
    save_json(captions, ids["captions"], indent=None)
    print("captions", len(captions["train"]), len(captions["dev"]))

    # ---- annotation-derived grounding
    ins_tr = load(f"{coco}/annotations/instances_train2014.json")
    cats = {c["id"]: c["name"] for c in ins_tr["categories"]}
    cnt_tr, amb_tr = category_stats(ins_tr, d["min_area_frac"])
    all_tr = sorted(im["id"] for im in ins_tr["images"])
    extra = [i for i in all_tr if i not in set(tr_sel)]
    n_extra = max(0, d["grounding_pool_images"] - len(tr_sel))
    pool_imgs = set(tr_sel) | set(rng.choice(extra, n_extra, replace=False).tolist())
    lo, hi = d["count_range"]
    grounding = {
        "pool_images": sorted(pool_imgs),
        "train_presence": presence_pairs(rng, pool_imgs, cnt_tr, amb_tr, cats, d["presence_pool_size"]),
        "train_counting": counting_pairs(rng, pool_imgs, cnt_tr, amb_tr, cats, d["counting_pool_size"], lo, hi),
    }
    del ins_tr
    ins_va = load(f"{coco}/annotations/instances_val2014.json")
    cnt_va, amb_va = category_stats(ins_va, d["min_area_frac"])
    del ins_va
    for split, pool, n in (("dev", dev_pool, d["grounding_dev_pairs"]), ("test", test_pool, d["grounding_test_pairs"])):
        used = set()
        n_pres = int(round(n * d["presence_fraction_eval"]))
        ps = presence_pairs(rng, pool, cnt_va, amb_va, cats, n_pres, used, unique=True)
        cs = counting_pairs(rng, pool, cnt_va, amb_va, cats, n - n_pres, lo, hi, used, unique=True)
        assert len(ps) == n_pres and len(cs) == n - n_pres
        grounding[split] = [{**p, "pair_id": f"{split}_{p['kind']}_{k}"} for k, p in enumerate(ps + cs)]
    save_json(grounding, ids["grounding"], indent=None)
    print("grounding pools", len(grounding["train_presence"]), len(grounding["train_counting"]))

    # ---- natural VQAv2 complementary pairs
    tr = vqa_candidates(f"{vqa}/v2_OpenEnded_mscoco_train2014_questions.json",
                        f"{vqa}/v2_mscoco_train2014_annotations.json",
                        f"{vqa}/v2_mscoco_train2014_complementary_pairs.json",
                        tok, caps["question"], caps["answer"], d["min_agreement"])
    va = vqa_candidates(f"{vqa}/v2_OpenEnded_mscoco_val2014_questions.json",
                        f"{vqa}/v2_mscoco_val2014_annotations.json",
                        f"{vqa}/v2_mscoco_val2014_complementary_pairs.json",
                        tok, caps["question"], caps["answer"], d["min_agreement"])
    vqa_sel = {
        "train": sample_vqa(rng, tr, d["vqa_train_pairs"]),
        "dev": sample_vqa(rng, va, d["vqa_dev_pairs"], dev_pool),
        "test": sample_vqa(rng, va, d["vqa_test_pairs"], test_pool),
    }
    save_json(vqa_sel, ids["vqa"], indent=None)
    print({k: len(v) for k, v in vqa_sel.items()})

    # ---- isolation checks
    eval_imgs = {"dev": set(), "test": set()}
    for s in ("dev", "test"):
        eval_imgs[s] |= {e["image_id"] for p in grounding[s] for e in p["examples"]}
        eval_imgs[s] |= {e["image_id"] for p in vqa_sel[s] for e in p["examples"]}
    eval_imgs["dev"] |= set(dev_sel)
    assert not (eval_imgs["dev"] & eval_imgs["test"]), "dev/test images overlap"
    assert eval_imgs["dev"] <= dev_pool and eval_imgs["test"] <= test_pool
    # training uses train2014 only (different COCO split) -> disjoint from val2014 by construction

    mm = {}
    for s in ("dev", "test"):
        mm[f"grounding_{s}"] = mismatch_map(rng, len(grounding[s]))
        mm[f"vqa_{s}"] = mismatch_map(rng, len(vqa_sel[s]))
    save_json(mm, ids["mismatch"])
    print("done")


if __name__ == "__main__":
    main()
