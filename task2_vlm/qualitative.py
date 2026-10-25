"""Aligned complementary-image examples from the final evaluation.

Selection rule (fixed before looking at predictions): for each suite, the first two test pairs
of every question type in the saved pair order. For each pair: shared question, both images,
both references, and predictions under correct / swap / mismatch / no-image inputs.

python -m task2_vlm.qualitative --config configs/task2_vlm.yaml
"""
import argparse

from common.io import load_yaml, read_jsonl, results_dir, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2_vlm.yaml")
    ap.add_argument("--per-type", type=int, default=2)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rd = results_dir(cfg)
    rows = read_jsonl(rd / f"{cfg['run_name']}_final_per_example.jsonl")
    out = []
    for suite in ("grounding", "vqa"):
        sr = [r for r in rows if r["suite"] == suite]
        order = sorted({(r["pair_index"], r["type"]) for r in sr})
        chosen, seen = [], {}
        for pi, t in order:
            if seen.get(t, 0) < args.per_type:
                chosen.append(pi)
                seen[t] = seen.get(t, 0) + 1
        for pi in chosen:
            pr = [r for r in sr if r["pair_index"] == pi]
            sides = []
            for side in (0, 1):
                s = {c: r["prediction"] for r in pr if r["side"] == side for c in [r["condition"]]}
                ref = next(r for r in pr if r["side"] == side)
                sides.append({"image": next(r["image"] for r in pr if r["side"] == side and r["condition"] == "correct"),
                              "reference": ref["reference"], "predictions": s})
            out.append({"suite": suite, "pair_id": pr[0]["pair_id"], "type": pr[0]["type"],
                        "question": pr[0]["question"], "sides": sides})
    write_jsonl(out, rd / f"{cfg['run_name']}_qualitative_examples.jsonl")
    print(f"{len(out)} examples ->", rd / f"{cfg['run_name']}_qualitative_examples.jsonl")


if __name__ == "__main__":
    main()
