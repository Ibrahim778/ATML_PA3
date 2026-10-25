"""Adaptive stopping (Task 3, Step 5): offline simulation over the saved rollout orders.

Checks at n in {4, 8, 16}; otherwise stops at 32. Both controllers return the current
self-consistency answer.
  agreement:  unique non-NO_ANSWER leading SC group with votes / n >= tau
  quality:    agreement AND SC winner == RWC winner AND RWC margin >= gamma
              (self-certainty K for MATH, VisConf Z for MathVista; recomputed on revealed
               candidates only)
Settings are selected on dev (lowest mean cost with accuracy >= fixed-32 SC - 1 pp; ties:
higher accuracy, smaller tau, smaller gamma; always-32 included as fallback), frozen, then
evaluated on held-out questions.

python -m task3_ttc.adaptive --config configs/task3_ttc.yaml
"""
import argparse
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.io import load_yaml, results_dir, save_json
from common.metrics import paired_bootstrap_diff
from task3_ttc import answers as A
from task3_ttc.answers import group_prefix
from task3_ttc.data import load_orders, load_questions, load_splits
from task3_ttc.selection import rwc_group_weights, rwc_margin, select_sc, visconf_scores


def checkpoint_stats(cfg, name):
    """For every question x order x n in checkpoints + [max_n]: SC state and RWC state."""
    qs = load_questions(cfg, name)
    orders = load_orders(cfg, name)
    is_vlm = cfg["data"]["datasets"][name].get("has_attention", False)
    ns = list(cfg["adaptive"]["checkpoints"]) + [cfg["adaptive"]["max_n"]]
    rows = []
    for qid, q in tqdm(qs.items(), desc=f"[{name}] adaptive stats"):
        for o, order in enumerate(orders[qid]):
            order = np.asarray(order)
            for n in ns:
                prefix = order[:n]
                gid = group_prefix(prefix, q.E)
                votes = np.bincount(gid)
                g_sc, pos_sc = select_sc(gid)
                top = votes.max()
                unique = int((votes == top).sum()) == 1
                if is_vlm:
                    score = visconf_scores([q.kl[i] for i in prefix], [q.v[i] for i in prefix])["Z"]
                else:
                    score = q.K[prefix]
                W = rwc_group_weights(gid, score)
                g_rwc = int(np.flatnonzero(np.isclose(W, W.max(), rtol=0, atol=1e-12))[0])
                rid = int(prefix[pos_sc])
                rows.append({
                    "qid": qid, "order": o, "n": n,
                    "sc_frac": top / n, "sc_unique": unique,
                    "sc_is_noanswer": q.keys[rid] == A.NO_ANSWER,
                    "rwc_agrees": g_rwc == g_sc, "rwc_margin": rwc_margin(W),
                    "sc_correct": bool(q.correct[rid]), "sc_answer": q.keys[rid],
                    "cost": int(q.n_tokens[prefix].sum()),
                    "pool_frac_correct": float(q.correct.mean()),   # retrospective only
                })
    return pd.DataFrame(rows)


def _arrays(stats, ns):
    """Pivot stats into (runs, len(ns)) arrays; runs = (qid, order) pairs."""
    s = stats.sort_values(["qid", "order", "n"])
    runs = s[s["n"] == ns[0]][["qid", "order", "pool_frac_correct"]].reset_index(drop=True)
    R = len(runs)
    get = lambda c: s[c].to_numpy().reshape(R, len(ns))
    assert (get("n") == np.array(ns)[None, :]).all()
    return runs, {c: get(c) for c in ("sc_frac", "sc_unique", "sc_is_noanswer", "rwc_agrees",
                                       "rwc_margin", "sc_correct", "sc_answer", "cost")}


def simulate(stats, tau, gamma, checkpoints, max_n):
    """Stopping decision per (qid, order). tau=None -> always max_n. gamma=None -> agreement only."""
    ns = list(checkpoints) + [max_n]
    runs, a = _arrays(stats, ns)
    R = len(runs)
    stop_j = np.full(R, len(ns) - 1)
    if tau is not None:
        ok = a["sc_unique"].astype(bool) & ~a["sc_is_noanswer"].astype(bool) & (a["sc_frac"] >= tau - 1e-12)
        if gamma is not None:
            ok &= a["rwc_agrees"].astype(bool) & (a["rwc_margin"] >= gamma - 1e-12)
        ok = ok[:, :len(checkpoints)]
        any_ok = ok.any(axis=1)
        stop_j[any_ok] = ok[any_ok].argmax(axis=1)
    rr = np.arange(R)
    return pd.DataFrame({
        "qid": runs["qid"], "order": runs["order"], "stop_n": np.array(ns)[stop_j],
        "correct": a["sc_correct"][rr, stop_j].astype(bool), "cost": a["cost"][rr, stop_j],
        "answer": a["sc_answer"][rr, stop_j], "final32_correct": a["sc_correct"][:, -1].astype(bool),
        "pool_frac_correct": runs["pool_frac_correct"],
    })


def score(sim):
    pq = sim.groupby("qid")[["correct", "cost", "stop_n"]].mean()
    return {"accuracy": float(pq["correct"].mean()), "cost": float(pq["cost"].mean()),
            "rollouts": float(pq["stop_n"].mean())}


def select_setting(stats_dev, family, cfg):
    a = cfg["adaptive"]
    cps, mx = a["checkpoints"], a["max_n"]
    base = score(simulate(stats_dev, None, None, cps, mx))
    grid = [(None, None)]
    if family == "agreement":
        grid += [(t, None) for t in a["taus"]]
    else:
        grid += list(itertools.product(a["taus"], a["gammas"]))
    table = []
    for tau, gamma in grid:
        r = score(simulate(stats_dev, tau, gamma, cps, mx))
        r.update({"tau": tau, "gamma": gamma,
                  "eligible": r["accuracy"] >= base["accuracy"] - a["tolerance_pp"] / 100 - 1e-12})
        table.append(r)
    elig = [r for r in table if r["eligible"]]
    big = 1e9
    best = min(elig, key=lambda r: (round(r["cost"], 6), -r["accuracy"],
                                    big if r["tau"] is None else r["tau"],
                                    big if r["gamma"] is None else r["gamma"]))
    return best, base, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3_ttc.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rd = results_dir(cfg)
    a = cfg["adaptive"]
    splits = load_splits(cfg)
    for name in cfg["data"]["datasets"]:
        stats = checkpoint_stats(cfg, name)
        stats.to_parquet(rd / f"{name}_adaptive_stats.parquet", index=False)
        dev = stats[stats["qid"].isin(set(splits[name]["dev"]))]
        ev = stats[stats["qid"].isin(set(splits[name]["eval"]))]
        res = {"selection_dev": {}, "heldout": {}}
        fixed = simulate(ev, None, None, a["checkpoints"], a["max_n"])
        res["heldout"]["fixed32_sc"] = score(fixed)
        for family in ("agreement", "quality"):
            best, base, table = select_setting(dev, family, cfg)
            res["selection_dev"][family] = {"selected": best, "fixed32_dev": base, "grid": table}
            sim = simulate(ev, best["tau"], best["gamma"] if family == "quality" else None,
                           a["checkpoints"], a["max_n"])
            sim.to_parquet(rd / f"{name}_adaptive_{family}_heldout.parquet", index=False)
            r = score(sim)
            dist = sim["stop_n"].value_counts(normalize=True).sort_index()
            early = sim[sim["stop_n"] < a["max_n"]]
            wrong_early = early[~early["correct"]]
            pq_a = sim.groupby("qid")["correct"].mean()
            pq_f = fixed.groupby("qid")["correct"].mean().loc[pq_a.index]
            d, lo, hi = paired_bootstrap_diff(pq_a.to_numpy(), pq_f.to_numpy(),
                                              n_boot=cfg["offline"]["bootstrap"], seed=cfg["seed"])
            r.update({
                "tau": best["tau"], "gamma": best["gamma"],
                "stop_distribution": {int(k): float(v) for k, v in dist.items()},
                "early_stop_rate": float(len(early) / len(sim)),
                "incorrect_early_stops": int(len(wrong_early)),
                "incorrect_early_stops_recovered_by_fixed32": int(wrong_early["final32_correct"].sum()),
                "n_runs": int(len(sim)),
                "acc_minus_fixed32_pp": {"diff": 100 * d, "lo": 100 * lo, "hi": 100 * hi},
                "mean_pool_frac_correct_by_stop_n": {int(k): float(v) for k, v in
                                                     sim.groupby("stop_n")["pool_frac_correct"].mean().items()},
            })
            res["heldout"][family] = r
        save_json(res, rd / f"{name}_adaptive.json")
        print(name, {k: (v.get("accuracy"), v.get("cost")) for k, v in res["heldout"].items()})


if __name__ == "__main__":
    main()
