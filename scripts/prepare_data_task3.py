"""Task 3 data preparation: join supplied tables, verify alignment, normalise answers,
compute equivalence/correctness, save splits (ids/task3_splits.json) and rollout orders.

python -m scripts.prepare_data_task3 --config configs/task3_ttc.yaml [--inspect]
"""
import argparse

import pandas as pd

from common.io import load_yaml, results_dir, save_json
from task3_ttc import answers as A
from task3_ttc.data import (build_questions, check_alignment, load_joined, make_orders,
                            make_splits, orders_path, save_questions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3_ttc.yaml")
    ap.add_argument("--inspect", action="store_true", help="print schemas and exit")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed = cfg["seed"]
    splits, report = {}, {"scorer": A.scorer_versions()}
    for name, dcfg in cfg["data"]["datasets"].items():
        df = load_joined(cfg, name)
        if args.inspect:
            print(f"== {name}: {len(df)} rows; columns:")
            with pd.option_context("display.max_colwidth", 80):
                print(df.dtypes)
                print(df.head(2).T)
            continue
        report[name] = {"alignment": check_alignment(df, cfg, dcfg.get("has_attention", False))}
        print(name, report[name]["alignment"])
        qids = sorted(df["sample_id"].astype(str).unique())
        n_roll = df.groupby("sample_id").size()
        assert n_roll.nunique() == 1, f"{name}: unequal rollout counts per question"
        sp = make_splits(qids, seed, cfg["splits"]["n_dev"],
                         cfg["splits"]["n_live"] if dcfg["kind"] == "math" else None)
        splits[name] = sp
        save_json(make_orders(qids, int(n_roll.iloc[0]), cfg["offline"]["n_orders"], seed),
                  orders_path(cfg, name), indent=None)
        qs = build_questions(df, cfg, name)
        save_questions(qs, cfg, name)
        n_noans = sum(k == A.NO_ANSWER for q in qs.values() for k in q.keys)
        report[name].update({
            "n_questions": len(qs), "rollouts_per_question": int(n_roll.iloc[0]),
            "no_answer_rollouts": n_noans,
            "rollout_accuracy": float(sum(q.correct.sum() for q in qs.values()) / sum(len(q.keys) for q in qs.values())),
        })
    if args.inspect:
        return
    save_json(splits, cfg["splits"]["ids_file"])
    save_json(report, results_dir(cfg) / "data_report.json")
    print("saved splits ->", cfg["splits"]["ids_file"])


if __name__ == "__main__":
    main()
