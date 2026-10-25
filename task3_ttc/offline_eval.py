"""Offline best-of-N evaluation on the supplied rollouts (Task 3, Steps 1-2).

For each question, each of the 10 saved rollout orders and each budget N, the nested prefix
order[:N] is revealed; every method selects from that prefix only. Outputs:
  results/task3/<ds>_per_question.parquet   question x order x N x method rows
  results/task3/<ds>_coverage.parquet       question x order x N: coverage, #groups, #NO_ANSWER
  results/task3/<ds>_offline_eval.json      accuracy / coverage / cost curves, failure split,
                                            paired bootstrap CIs (averaged over orders first)

python -m task3_ttc.offline_eval --config configs/task3_ttc.yaml
"""
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.io import load_yaml, results_dir, save_json
from common.metrics import paired_bootstrap_diff
from task3_ttc import answers as A
from task3_ttc.answers import group_prefix
from task3_ttc.data import load_orders, load_questions, load_splits
from task3_ttc.selection import BASE_METHODS, VLM_METHODS, run_methods

COMPARISONS = {
    "base": [("sc", "first"), ("max_L", "sc"), ("max_K", "sc"), ("K_rwc", "sc"),
             ("K_rwc", "max_K"), ("min_len", "sc")],
    "vlm": [("max_V", "sc"), ("V_rwc", "sc"), ("max_Z", "sc"), ("Z_rwc", "sc"),
            ("Z_rwc", "max_Z"), ("Z_rwc", "K_rwc"), ("max_Z", "max_V"),
            ("max_Ztok", "max_Z"), ("Ztok_rwc", "Z_rwc")],
}


def evaluate_dataset(cfg, name):
    qs = load_questions(cfg, name)
    orders = load_orders(cfg, name)
    splits = load_splits(cfg)[name]
    split_of = {q: "dev" for q in splits["dev"]}
    split_of.update({q: "eval" for q in splits["eval"]})
    is_vlm = cfg["data"]["datasets"][name].get("has_attention", False)
    budgets = cfg["offline"]["budgets"]
    rows, cov_rows = [], []
    for qid, q in tqdm(qs.items(), desc=f"[{name}] offline"):
        for o, order in enumerate(orders[qid]):
            order = np.asarray(order)
            for N in budgets:
                prefix = order[:N]
                gid = group_prefix(prefix, q.E)
                sel, _ = run_methods(q, prefix, gid, is_vlm)
                cost_all = int(q.n_tokens[prefix].sum())
                cov_rows.append({
                    "qid": qid, "split": split_of[qid], "order": o, "N": N,
                    "coverage": bool(q.correct[prefix].any()),
                    "n_groups": int(gid.max() + 1),
                    "n_no_answer": int(sum(q.keys[i] == A.NO_ANSWER for i in prefix)),
                    "frac_correct": float(q.correct[prefix].mean()),
                })
                for m, pos in sel.items():
                    rid = int(prefix[pos])
                    rows.append({
                        "qid": qid, "split": split_of[qid], "order": o, "N": N, "method": m,
                        "selected_rollout": q.rollout_ids[rid], "selected_answer": q.keys[rid],
                        "candidate_ids": [q.rollout_ids[i] for i in prefix],
                        "correct": bool(q.correct[rid]),
                        "cost": int(q.n_tokens[prefix[0]]) if m == "first" else cost_all,
                    })
    return pd.DataFrame(rows), pd.DataFrame(cov_rows)


def per_question_means(df, cols, keys):
    """Average over orders within each question."""
    return df.groupby(keys + ["qid"], as_index=False)[cols].mean()


def summarise(df, cov, cfg, is_vlm):
    budgets = cfg["offline"]["budgets"]
    methods = BASE_METHODS + (VLM_METHODS if is_vlm else [])
    out = {"n_questions": {}, "curves": {}, "failure_split": {}, "bootstrap": {}}
    pm = per_question_means(df, ["correct", "cost"], ["split", "method", "N"])
    pc = per_question_means(cov, ["coverage", "n_groups", "n_no_answer"], ["split", "N"])
    # failure split per method: no correct candidate vs correct candidate missed
    merged = df.merge(cov[["qid", "order", "N", "coverage"]], on=["qid", "order", "N"])
    merged["no_cov"] = ~merged["coverage"]
    merged["missed"] = merged["coverage"] & ~merged["correct"]
    pf = per_question_means(merged, ["no_cov", "missed"], ["split", "method", "N"])
    for split in ("dev", "eval", "all"):
        sel = (lambda d: d) if split == "all" else (lambda d, s=split: d[d["split"] == s])
        a, c, f = sel(pm), sel(pc), sel(pf)
        out["n_questions"][split] = int(c["qid"].nunique())
        curves = {"coverage": {}, "n_groups": {}, "n_no_answer": {}}
        for N in budgets:
            cN = c[c["N"] == N]
            curves["coverage"][N] = float(cN["coverage"].mean())
            curves["n_groups"][N] = float(cN["n_groups"].mean())
            curves["n_no_answer"][N] = float(cN["n_no_answer"].mean())
        for m in methods:
            am = a[a["method"] == m]
            curves[m] = {N: {"accuracy": float(am[am["N"] == N]["correct"].mean()),
                             "cost": float(am[am["N"] == N]["cost"].mean())} for N in budgets}
            fm = f[f["method"] == m]
            out["failure_split"].setdefault(split, {})[m] = {
                N: {"no_correct_candidate": float(fm[fm["N"] == N]["no_cov"].mean()),
                    "correct_missed": float(fm[fm["N"] == N]["missed"].mean())} for N in budgets}
        out["curves"][split] = curves
        # paired question-level bootstrap on accuracy differences (after averaging over orders)
        comps = COMPARISONS["base"] + (COMPARISONS["vlm"] if is_vlm else [])
        bs = {}
        for N in budgets:
            if N == 1:
                continue
            piv = a[a["N"] == N].pivot(index="qid", columns="method", values="correct")
            for x, y in comps:
                mean, lo, hi = paired_bootstrap_diff(piv[x].to_numpy(), piv[y].to_numpy(),
                                                     n_boot=cfg["offline"]["bootstrap"], seed=cfg["seed"])
                bs[f"{x}-{y}@{N}"] = {"diff_pp": 100 * mean, "lo_pp": 100 * lo, "hi_pp": 100 * hi}
        out["bootstrap"][split] = bs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3_ttc.yaml")
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rd = results_dir(cfg)
    for name in args.datasets or cfg["data"]["datasets"]:
        df, cov = evaluate_dataset(cfg, name)
        df.to_parquet(rd / f"{name}_per_question.parquet", index=False)
        cov.to_parquet(rd / f"{name}_coverage.parquet", index=False)
        is_vlm = cfg["data"]["datasets"][name].get("has_attention", False)
        summ = summarise(df, cov, cfg, is_vlm)
        summ["scorer"] = A.scorer_versions()
        save_json(summ, rd / f"{name}_offline_eval.json")
        acc32 = summ["curves"]["eval"]
        print(name, {m: round(acc32[m][32]["accuracy"], 4) for m in acc32 if m not in ("coverage", "n_groups", "n_no_answer")},
              "coverage@32", round(acc32["coverage"][32], 4))


if __name__ == "__main__":
    main()
