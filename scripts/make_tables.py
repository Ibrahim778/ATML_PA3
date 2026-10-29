"""Build report/tables.tex from results/ (booktabs, \\footnotesize, tabular* + \\extracolsep{\\fill},
floats [!htb]). Each table contains only the quantities its evidence bullet asks for.
Missing inputs are skipped with a note.

python -m scripts.make_tables
"""
import argparse
from pathlib import Path

from common.io import load_json, load_yaml, repo_path

COND_LABELS = {"baseline": "Baseline", "full_625": "Full FT @625", "full_1250": "Full FT @1250",
               "l2sp_1250": "Full FT + L2-SP", "lora4_1250": "LoRA-4", "lora16_1250": "LoRA-16",
               "oplora16_1250": "OPLoRA-16"}
METHOD_LABELS = {"full": "Full FT", "l2sp": "Full FT + L2-SP", "lora4": "LoRA-4", "lora16": "LoRA-16",
                 "oplora16": "OPLoRA-16"}
T3_NAMES = {"first": "FirstAns", "sc": "SC", "max_L": "max $L$", "max_K": "max $K$", "K_rwc": "$K$+RWC",
            "min_len": "Shortest", "max_V": "max $V$", "V_rwc": "$V$+RWC", "max_Z": "VisConf",
            "Z_rwc": "VisConf+RWC", "max_Ztok": "$Z^{tok}$", "Ztok_rwc": "$Z^{tok}$+RWC"}


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def table(label, caption, colspec, header, rows):
    body = "\n".join(" & ".join(map(str, r)) + r" \\" for r in rows)
    return (f"\\begin{{table}}[!htb]\n\\centering\\footnotesize\n\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}{colspec}}}\n\\toprule\n"
            f"{' & '.join(header)} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular*}}\n\\end{{table}}\n")


def pct(x, d=1):
    return "--" if x is None else f"{100 * x:.{d}f}"


def signed(x, d=1):
    return "--" if x is None else f"{x:+.{d}f}"


def ci(m, d=1, scale=100):
    return f"{scale * m['mean']:.{d}f} [{scale * m['lo']:.{d}f}, {scale * m['hi']:.{d}f}]"


# ----------------------------------------------------------------------------- Task 1
def t1(res, cfg):
    out = []
    conds = [c for c in cfg["conditions"] if (res / f"{c}_eval.json").exists()]
    if conds:
        benches = [("gsm8k", "GSM8K"), ("hellaswag", "HellaSwag"), ("boolq", "BoolQ"), ("arc_easy", "ARC-E")]
        base = load_json(res / "baseline_eval.json")["benchmarks"] if (res / "baseline_eval.json").exists() else {}
        rows = []
        for c in conds:
            b = load_json(res / f"{c}_eval.json")["benchmarks"]
            r = [COND_LABELS.get(c, esc(c))]
            for k, _ in benches:
                v = b.get(k, {}).get("primary")
                d = None if (v is None or k not in base or c == "baseline") else 100 * (v - base[k]["primary"])
                r += [pct(v), signed(d) if c != "baseline" else "--"]
            rows.append(r)
        hdr = ["Condition"] + [x for _, n in benches for x in (n, r"$\Delta$")]
        out.append(table("tab:t1_bench", "Zero-shot benchmark scores (\\%) and change from baseline (pp). GSM8K: "
                         "flexible-extract EM; HellaSwag/ARC-E: acc\\_norm; BoolQ: acc.", "l" + "rr" * 4, hdr, rows))
    rows = []
    for m in ("full", "l2sp", "lora4", "lora16", "oplora16"):
        p = res / f"{m}_summary.json"
        if not p.exists():
            continue
        s = load_json(p)
        ck = s.get("checkpoint_bytes", {}).get("1250")
        extra = []
        if "theta0_bytes" in s:
            extra.append(f"$\\theta_0$ {s['theta0_bytes'] / 2**30:.2f} GiB")
        if "basis_bytes" in s:
            extra.append(f"bases {s['basis_bytes'] / 2**20:.0f} MiB, SVD {s.get('preprocess_s', 0):.0f}s")
        rows.append([METHOD_LABELS[m], f"{s['params']['trainable'] / 1e6:.2f}M", f"{s['train_time_s'] / 3600:.2f}",
                     f"{s['peak_mem_gb']:.1f}", "--" if ck is None else f"{ck / 2**20:.0f}", "; ".join(extra) or "--"])
    if rows:
        out.append(table("tab:t1_resources", "Training resources (1,250 updates). Checkpoint = exported eval weights "
                         "(FP16 model or adapter factors).", "lrrrrl",
                         ["Method", "Trainable", "Time (h)", "Peak mem (GB)", "Ckpt (MiB)", "Overhead"], rows))
    rows = []
    for m in ("full", "l2sp"):
        p = res / f"{m}_summary.json"
        if p.exists():
            s = load_json(p)
            om = s.get("final_omega")
            lam = cfg["methods"][m].get("l2sp_lambda", 0)
            rows.append([METHOD_LABELS[m], "--" if om is None else f"{om:.3g}", "--" if om is None else f"{lam * om:.3g}",
                         f"{s.get('displacement_l2', {}).get('625', float('nan')):.3g}",
                         f"{s.get('displacement_l2', {}).get('1250', float('nan')):.3g}"])
    if rows:
        out.append(table("tab:t1_l2sp", "Starting-point penalty and full-model displacement.", "lrrrr",
                         ["Run", r"$\Omega$ (final)", r"$\lambda\Omega$", r"$\|\theta-\theta_0\|$ @625",
                          r"$\|\theta-\theta_0\|$ @1250"], rows))
    if (res / "diagnostics_weights.json").exists():
        w = load_json(res / "diagnostics_weights.json")["results"]
        a = load_json(res / "diagnostics_activation.json")["results"]
        rows = []
        for c in w:
            for li in w[c]:
                rows.append([COND_LABELS.get(c, esc(c)), li, f"{w[c][li]['rho_k']:.3f}", f"{w[c][li]['eta']:.2e}",
                             f"{a[c][li]['math']['mean']:.2e}", f"{a[c][li]['nonmath']['mean']:.2e}"])
        out.append(table("tab:t1_diag", "down\\_proj diagnostics ($k{=}128$; R(z) means over 32 probes each).",
                         "lrrrrr", ["Condition", "Layer", r"$\rho_k$", r"$\eta$", "R math", "R non-math"], rows))
    return out


# ----------------------------------------------------------------------------- Task 2
def t2(res, cfg):
    out, run = [], cfg["run_name"]
    p = res / f"{run}_final_eval.json"
    if p.exists():
        f = load_json(p)
        names = {"correct": "Correct", "swap": "Swap", "mismatch": "Mismatch", "noimage": "No image"}
        for suite, title in (("grounding", "Annotation-derived grounding"), ("vqa", "Natural VQAv2")):
            rows = []
            for c, lab in names.items():
                m = f[suite]["conditions"][c]
                rows.append([lab, ci(m["accuracy"]), ci(m["pair_accuracy"]), pct(m["identical_rate"]["mean"]),
                             pct(m["empty_rate"]["mean"]), f"{m['ref_nll']['mean']:.2f}",
                             f"{1000 * m['gen_time_per_answer_s']['mean']:.0f}"])
            out.append(table(f"tab:t2_{suite}", f"{title} test suite ({f[suite]['n_pairs']} pairs); 95\\% CIs from "
                             f"{f['n_bootstrap']} pair bootstraps. Identical = both answers of a pair equal.",
                             "lrrrrrr", ["Image", "Acc (\\%)", "Pair acc (\\%)", "Identical", "Empty", "Ref NLL",
                                         "ms/answer"], rows))
            types = sorted(f[suite]["conditions"]["correct"]["by_type"])
            rows = [[esc(t)] + [pct(f[suite]["conditions"][c]["by_type"][t]["accuracy"]["mean"]) for c in names]
                    for t in types]
            adv = f[suite]["advantages"]["correct_minus_swap"]
            rows.append(["Correct$-$swap", ci(adv["accuracy"]), "NLL", ci(adv["nll_other_minus_correct"], 2, 1), ""])
            out.append(table(f"tab:t2_{suite}_types", f"{title}: accuracy (\\%) by question type and correct-image "
                             "advantage over the complementary swap (accuracy pp, NLL nats).", "lrrrr",
                             ["Type"] + list(names.values()), rows))
    p = res / f"{run}_summary.json"
    if p.exists():
        s = load_json(p)
        rows = []
        fc = res / "feature_cache_summary.json"
        if fc.exists():
            c = load_json(fc)
            rows.append(["SigLIP feature caching", "0", f"{c['seconds_this_run'] / 60:.1f}", f"{c['peak_mem_gb']:.1f}"])
        for st, v in s["stages"].items():
            rows.append([esc(st), f"{v.get('trainable_params', 0) / 1e6:.2f}M", f"{v.get('time_s', 0) / 60:.1f}",
                         f"{v.get('peak_mem_gb', float('nan')):.1f}"])
        out.append(table("tab:t2_resources", "Task 2 resources by stage (dev evaluations included in stage time).",
                         "lrrr", ["Phase", "Trainable", "Time (min)", "Peak mem (GB)"], rows))
    return out


# ----------------------------------------------------------------------------- Task 3
def t3(res, cfg):
    out = []
    for ds, title in (("math", "MATH (Qwen2.5-3B)"), ("mathvista", "MathVista (Qwen2.5-VL-3B)")):
        p = res / f"{ds}_offline_eval.json"
        if not p.exists():
            continue
        e = load_json(p)
        c, bs = e["curves"]["eval"], e["bootstrap"]["eval"]
        methods = [m for m in T3_NAMES if m in c]
        rows = []
        for m in methods:
            r = [T3_NAMES[m]] + [pct(c[m][n]["accuracy"]) for n in ("4", "8", "32")] + [f"{c[m]['32']['cost'] / 1000:.1f}k"]
            k = f"{m}-sc@32"
            r.append(signed(bs[k]["diff_pp"]) + f" [{bs[k]['lo_pp']:+.1f}, {bs[k]['hi_pp']:+.1f}]" if k in bs else "--")
            rows.append(r)
        rows.append(["Oracle coverage"] + [pct(c["coverage"][n]) for n in ("4", "8", "32")] + ["", ""])
        out.append(table(f"tab:t3_{ds}_methods", f"{title}, held-out ({e['n_questions']['eval']} questions): accuracy "
                         "(\\%) by candidate budget, token cost at $N{=}32$, and paired-bootstrap difference vs SC at 32 (pp).",
                         "lrrrrr", ["Method", "N=4", "N=8", "N=32", "Cost@32", r"$\Delta$ vs SC"], rows))
        fs = e["failure_split"]["eval"]
        rows = [[f"N={n}", pct(fs["sc"][n]["no_correct_candidate"]), pct(fs["sc"][n]["correct_missed"]),
                 pct(fs.get("K_rwc", fs["sc"])[n]["correct_missed"])] for n in ("1", "4", "8", "32")]
        out.append(table(f"tab:t3_{ds}_fail", f"{title}: share of questions (\\%) failed because no candidate was correct "
                         "vs a correct candidate was missed.", "lrrr",
                         ["Budget", "No correct cand.", "Missed (SC)", "Missed (K+RWC)"], rows))
        p = res / f"{ds}_adaptive.json"
        if p.exists():
            a = load_json(p)["heldout"]
            rows = [["Fixed-32 SC", "--", "--", pct(a["fixed32_sc"]["accuracy"]), f"{a['fixed32_sc']['cost'] / 1000:.1f}k",
                     f"{a['fixed32_sc']['rollouts']:.1f}", "--"]]
            for fam in ("agreement", "quality"):
                h = a[fam]
                dist = ", ".join(f"{k}:{100 * v:.0f}" for k, v in h["stop_distribution"].items())
                rows.append([fam.capitalize(), h["tau"], "--" if h["gamma"] is None else h["gamma"], pct(h["accuracy"]),
                             f"{h['cost'] / 1000:.1f}k", f"{h['rollouts']:.1f}", dist])
            out.append(table(f"tab:t3_{ds}_adaptive", f"{title}: adaptive stopping with dev-selected settings, held-out. "
                             "Stop distribution in \\% of runs.", "lrrrrrl",
                             ["Policy", r"$\tau$", r"$\gamma$", "Acc", "Cost", "Rollouts", "Stops at n"], rows))
    p = res / "live_eval.json"
    if p.exists():
        lv = load_json(p)
        rows = []
        for s, lab in (("single", "Single 3,072"), ("parallel", "Parallel 3$\\times$1,024 + SC"), ("refine", "Refine 1,024+2,048")):
            v = lv[s]
            rows.append([lab, pct(v["accuracy"]), f"{v['mean_input_tokens']:.0f}", f"{v['mean_output_tokens']:.0f}",
                         f"{v['mean_time_s']:.1f}", pct(v["call_truncation_rate"]), pct(v["extraction_rate"])])
        t = lv["refine"]["transitions"]
        out.append(table("tab:t3_live", f"Live MATH ({lv['n_questions']} questions): per-question means. Refinement "
                         f"transitions vs shared initial answer: wrong$\\to$correct {t['wrong_to_correct']}, "
                         f"correct$\\to$wrong {t['correct_to_wrong']}.", "lrrrrrr",
                         ["Strategy", "Acc", "In tok", "Out tok", "Time (s)", "Trunc.", "Extracted"], rows))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/tables.tex")
    args = ap.parse_args(argv)
    rr = repo_path(args.results)
    parts = ["% Auto-generated by scripts/make_tables.py -- do not edit by hand.\n"
             "% Preamble: \\usepackage{booktabs}; \\renewcommand{\\topfraction}{0.9}; "
             "\\renewcommand{\\floatpagefraction}{0.8}\n"]
    for name, fn, cfgp in (("Task 1", t1, "configs/task1_ft.yaml"), ("Task 2", t2, "configs/task2_vlm.yaml"),
                           ("Task 3", t3, "configs/task3_ttc.yaml")):
        tabs = fn(rr / ("task" + name[-1]), load_yaml(cfgp))
        parts.append(f"% ---- {name} ({len(tabs)} tables)\n" + "\n".join(tabs))
        print(name, len(tabs), "tables")
    p = repo_path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(parts))
    print("wrote", p)


if __name__ == "__main__":
    main()
