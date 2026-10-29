"""Signal analysis at N = 32 (Task 3, Step 3). Uses gold correctness retrospectively only.

* within-question Spearman associations among length, scores and correctness
* top- vs bottom-quartile correctness per score (within question, averaged over questions)
* shortest-rollout control (also evaluated as `min_len` in offline_eval)
* MathVista: within-rollout confidence-vision correlation; check Z_token - Z = Cov_t(c, v)
* fixed illustrative trajectories (token series of c_t, v_t) for figures

python -m task3_ttc.signals --config configs/task3_ttc.yaml
"""
import argparse

import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from common.io import load_yaml, results_dir, save_json
from task3_ttc.data import load_questions, load_splits
from task3_ttc.selection import calibrated_tokens, visconf_scores


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return np.nan
    return spearmanr(x[ok], y[ok]).statistic


def _summ(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)])
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "q25": float(np.quantile(v, .25)), "q75": float(np.quantile(v, .75)),
            "frac_positive": float((v > 0).mean())}


def quartile_gap(score, correct):
    """Mean correctness in the top quartile minus bottom quartile of `score` (within question).
    Ties are broken by rollout position (stable sort)."""
    s = np.asarray(score, float).copy()
    s[~np.isfinite(s)] = -np.inf
    k = len(s) // 4
    idx = np.argsort(-s, kind="stable")
    return float(correct[idx[:k]].mean() - correct[idx[-k:]].mean())


def analyse(cfg, name, split_ids):
    qs = load_questions(cfg, name)
    is_vlm = cfg["data"]["datasets"][name].get("has_attention", False)
    scores = ["L", "K", "length", "neg_length"] + (["V", "C", "Z", "Ztok"] if is_vlm else [])
    assoc = {f"{a}~correct": [] for a in scores if a != "neg_length"}
    assoc.update({f"length~{a}": [] for a in scores if a not in ("length", "neg_length")})
    gaps = {s: [] for s in scores}
    shortest, cv_corr, cov_err, n_mixed = [], [], [], 0
    for qid in tqdm(split_ids, desc=f"[{name}] signals"):
        q = qs[qid]
        allidx = np.arange(len(q.keys))
        sc = {"L": q.L, "K": q.K, "length": q.n_tokens.astype(float), "neg_length": -q.n_tokens.astype(float)}
        if is_vlm:
            vis = visconf_scores(q.kl, q.v)
            sc.update({"V": vis["V"], "C": vis["C"], "Z": vis["Z"], "Ztok": vis["Ztok"]})
            for i in allidx:
                if len(q.kl[i]) < 3:
                    continue
                c = calibrated_tokens(q.kl[i], vis["q05"], vis["q95"])
                v = q.v[i].astype(float)
                if np.ptp(c) > 0 and np.ptp(v) > 0:
                    cv_corr.append(pearsonr(c, v).statistic)
                cov = np.mean(c * v) - c.mean() * v.mean()
                cov_err.append(abs((vis["Ztok"][i] - vis["Z"][i]) - cov))
        corr = q.correct.astype(float)
        shortest.append(float(q.correct[int(np.argmin(q.n_tokens))]))
        if 0 < corr.mean() < 1:     # associations with correctness need variation
            n_mixed += 1
            for a in scores:
                if a == "neg_length":
                    continue
                assoc[f"{a}~correct"].append(_spearman(sc[a], corr))
                gaps[a].append(quartile_gap(sc[a], corr))
            gaps["neg_length"].append(quartile_gap(sc["neg_length"], corr))
        for a in scores:
            if a not in ("length", "neg_length"):
                assoc[f"length~{a}"].append(_spearman(sc["length"], sc[a]))
    out = {
        "n_questions": len(split_ids), "n_mixed_correctness": n_mixed,
        "within_question_spearman": {k: _summ(v) for k, v in assoc.items()},
        "quartile_gap_top_minus_bottom": {k: _summ(v) for k, v in gaps.items()},
        "shortest_rollout_accuracy": float(np.mean(shortest)),
        "mean_rollout_accuracy": float(np.mean([qs[q].correct.mean() for q in split_ids])),
    }
    if is_vlm:
        out["confidence_vision_within_rollout_pearson"] = _summ(cv_corr)
        out["ztok_minus_z_equals_cov_max_abs_err"] = float(np.max(cov_err)) if cov_err else None
    return out


def illustrative(cfg, name, split_ids, n_questions=4):
    """Fixed rule: the first `n_questions` dev questions (split order) with mixed correctness;
    for each, the highest- and lowest-Z (VLM) or K (LLM) rollouts. Token series saved."""
    qs = load_questions(cfg, name)
    is_vlm = cfg["data"]["datasets"][name].get("has_attention", False)
    out = []
    for qid in split_ids:
        q = qs[qid]
        if not (0 < q.correct.mean() < 1):
            continue
        if is_vlm:
            vis = visconf_scores(q.kl, q.v)
            s = vis["Z"]
        else:
            s = q.K
        s = np.where(np.isfinite(s), s, -np.inf)
        for tag, i in (("highest", int(np.argmax(s))), ("lowest", int(np.argmin(np.where(np.isfinite(s), s, np.inf))))):
            rec = {"qid": qid, "which": tag, "rollout_id": q.rollout_ids[i], "answer": q.keys[i],
                   "gold": q.gold, "correct": bool(q.correct[i]), "n_tokens": int(q.n_tokens[i]),
                   "score": float(s[i]), "logp": q.logp[i].tolist(), "kl": q.kl[i].tolist()}
            if is_vlm:
                rec["c"] = calibrated_tokens(q.kl[i], vis["q05"], vis["q95"]).tolist()
                rec["v"] = q.v[i].tolist()
            out.append(rec)
        if len(out) >= 2 * n_questions:
            break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3_ttc.yaml")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    rd = results_dir(cfg)
    splits = load_splits(cfg)
    for name in cfg["data"]["datasets"]:
        res = {s: analyse(cfg, name, splits[name][s]) for s in ("dev", "eval")}
        save_json(res, rd / f"signals_{name}.json")
        save_json(illustrative(cfg, name, splits[name]["dev"]), rd / f"signals_trajectories_{name}.json", indent=None)
        print(name, "done")


if __name__ == "__main__":
    main()
