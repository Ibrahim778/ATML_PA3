"""Build report/figures/*.pdf from results/ (no hand-typed numbers). Missing inputs are skipped.

python -m scripts.make_figures
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common.io import load_json, load_yaml, read_jsonl, repo_path

# fixed categorical order (never cycled); >8 series are split into panels
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e2dc"


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.5, "legend.frameon": False, "pdf.fonttype": 42})
    return plt


def smooth(y, w):
    if w <= 1 or len(y) < w:
        return np.asarray(y)
    k = np.ones(w) / w
    return np.convolve(y, k, mode="valid")


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    print("wrote", out / f"{name}.pdf")


# ----------------------------------------------------------------------------- Task 1
def task1(plt, res, out):
    runs = ["full", "l2sp", "lora4", "lora16", "oplora16"]
    labels = {"full": "Full FT", "l2sp": "Full FT + L2-SP", "lora4": "LoRA-4", "lora16": "LoRA-16", "oplora16": "OPLoRA-16"}
    logs = {r: read_jsonl(res / f"{r}_train_log.jsonl") for r in runs}
    logs = {r: v for r, v in logs.items() if v}
    if logs:
        fig, ax = plt.subplots(figsize=(5.5, 2.4))
        for i, r in enumerate(runs):
            if r not in logs:
                continue
            u = np.array([x["update"] for x in logs[r]]); y = np.array([x["sft_loss"] for x in logs[r]])
            w = 25
            ax.plot(u[w - 1:] if len(u) >= w else u, smooth(y, w), color=COLORS[i], label=labels[r])
        ax.set_xlabel("optimizer update"); ax.set_ylabel(f"SFT loss (nats/token, {25}-update mean)")
        ax.legend(ncol=3, loc="upper right")
        save(fig, out, "t1_sft_loss"); plt.close(fig)
    if "l2sp" in logs:
        fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.0))
        u = [x["update"] for x in logs["l2sp"]]
        axs[0].plot(u, [x["lambda_omega"] for x in logs["l2sp"]], color=COLORS[1])
        axs[0].set_xlabel("optimizer update"); axs[0].set_ylabel(r"$\lambda\Omega$ (L2-SP penalty)")
        disp = {}
        for r in ("full", "l2sp"):
            p = res / f"{r}_summary.json"
            if p.exists():
                disp[r] = load_json(p).get("displacement_l2", {})
        for i, r in enumerate(("full", "l2sp")):
            if r in disp and disp[r]:
                ks = sorted(disp[r], key=int)
                axs[1].plot([int(k) for k in ks], [disp[r][k] for k in ks], marker="o", ms=4,
                            color=COLORS[i], label=labels[r])
        axs[1].set_xlabel("optimizer update"); axs[1].set_ylabel(r"$\|\theta-\theta_0\|_2$")
        axs[1].legend()
        fig.tight_layout(); save(fig, out, "t1_l2sp_penalty_displacement"); plt.close(fig)
    p = res / "diagnostics_activation.json"
    if p.exists():
        act = load_json(p)["results"]
        w = load_json(res / "diagnostics_weights.json")["results"]
        conds = list(act)
        layers = list(next(iter(act.values())))
        fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.0))
        x = np.arange(len(conds))
        for j, li in enumerate(layers):
            off = (j - 1) * 0.27
            axs[0].bar(x + off, [w[c][li]["rho_k"] for c in conds], 0.25, color=COLORS[j], label=f"layer {li}")
            axs[1].bar(x + off, [w[c][li]["eta"] for c in conds], 0.25, color=COLORS[j])
            axs[2].bar(x + off, [act[c][li]["math"]["mean"] / max(act[c][li]["nonmath"]["mean"], 1e-12) for c in conds],
                       0.25, color=COLORS[j])
        for a, t in zip(axs, (r"$\rho_k$", r"$\eta$", "mean R(z): math / non-math")):
            a.set_xticks(x); a.set_xticklabels(conds, rotation=60, ha="right"); a.set_ylabel(t)
        axs[1].set_yscale("log")
        axs[0].legend()
        fig.tight_layout(); save(fig, out, "t1_diagnostics"); plt.close(fig)


# ----------------------------------------------------------------------------- Task 2
def task2(plt, res, out, cfg):
    run = cfg["run_name"]
    tl, dl = read_jsonl(res / f"{run}_train_log.jsonl"), read_jsonl(res / f"{run}_dev_log.jsonl")
    if not tl:
        return
    bounds = np.cumsum([s["updates"] for s in cfg["train"]["stages"]])[:-1]
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.1))
    u = np.array([x["update"] for x in tl]); y = np.array([x["loss"] for x in tl])
    axs[0].plot(u[24:] if len(u) >= 25 else u, smooth(y, 25), color=COLORS[0], label="train (25-update mean)")
    if dl:
        du = [x["update"] for x in dl]
        axs[0].plot(du, [x["caption_loss"] for x in dl], color=COLORS[1], marker="o", ms=3, label="dev caption")
        for i, s in enumerate(("grounding", "vqa")):
            axs[1].plot(du, [x[s]["nll_correct"] for x in dl], color=COLORS[i], marker="o", ms=3, label=f"{s}: correct")
            axs[1].plot(du, [x[s]["nll_noimage"] for x in dl], color=COLORS[i], ls="--", lw=1.0, label=f"{s}: no image")
            axs[2].plot(du, [x[s]["dnll_swap"] for x in dl], color=COLORS[i], marker="o", ms=3, label=s)
    axs[0].set_ylabel("loss (nats/token)"); axs[1].set_ylabel("answer NLL (nats/token)")
    axs[2].set_ylabel(r"$\Delta$NLL (partner $-$ correct)"); axs[2].axhline(0, color=MUTED, lw=0.8)
    for a in axs:
        a.set_xlabel("update")
        for b in bounds:
            a.axvline(b, color=MUTED, lw=0.8, ls=":")
        a.legend(fontsize=6)
    fig.tight_layout(); save(fig, out, "t2_learning_curves"); plt.close(fig)


# ----------------------------------------------------------------------------- Task 3
def task3(plt, res, out, cfg):
    names = {"first": "FirstAns", "sc": "SC", "max_L": "max L", "max_K": "max K", "K_rwc": "K+RWC",
             "min_len": "shortest", "max_V": "max V", "V_rwc": "V+RWC", "max_Z": "VisConf",
             "Z_rwc": "VisConf+RWC", "max_Ztok": "Z_token", "Ztok_rwc": "Z_token+RWC"}
    panels = [["first", "sc", "max_L", "max_K", "K_rwc", "min_len"],
              ["sc", "max_V", "V_rwc", "max_Z", "Z_rwc", "max_Ztok", "Ztok_rwc"]]
    for ds in cfg["data"]["datasets"]:
        p = res / f"{ds}_offline_eval.json"
        if not p.exists():
            continue
        c = load_json(p)["curves"]["eval"]
        Ns = sorted(int(n) for n in c["coverage"])
        use = [pl for pl in panels if all(m in c for m in pl)]
        fig, axs = plt.subplots(1, len(use) + 1, figsize=(2.6 * (len(use) + 1), 2.2))
        for ax, pl in zip(axs, use):
            ax.plot(Ns, [c["coverage"][str(n)] for n in Ns], color=INK, ls="--", lw=1.0, label="oracle coverage")
            for i, m in enumerate(pl):
                ax.plot(Ns, [c[m][str(n)]["accuracy"] for n in Ns], color=COLORS[i], marker="o", ms=3, label=names[m])
            ax.set_xscale("log", base=2); ax.set_xticks(Ns); ax.set_xticklabels(Ns)
            ax.set_xlabel("candidates N"); ax.set_ylabel("accuracy (held-out)")
            ax.legend(fontsize=6)
        ax = axs[-1]
        for i, m in enumerate(["first", "sc", "K_rwc"] + (["Z_rwc"] if "Z_rwc" in c else [])):
            ax.plot([c[m][str(n)]["cost"] for n in Ns], [c[m][str(n)]["accuracy"] for n in Ns],
                    color=COLORS[i], marker="o", ms=3, label=names[m])
        ad = res / f"{ds}_adaptive.json"
        if ad.exists():
            h = load_json(ad)["heldout"]
            for j, fam in enumerate(("agreement", "quality")):
                ax.scatter([h[fam]["cost"]], [h[fam]["accuracy"]], marker="D", s=30, color=COLORS[5 + j],
                           edgecolor="white", linewidth=1.5, zorder=3, label=f"adaptive ({fam})")
        ax.set_xscale("log"); ax.set_xlabel("generated tokens / question"); ax.set_ylabel("accuracy (held-out)")
        ax.legend(fontsize=6)
        fig.tight_layout(); save(fig, out, f"t3_{ds}_accuracy_coverage_cost"); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    plt = setup()
    out, rr = repo_path(args.out), repo_path(args.results)
    task1(plt, rr / "task1", out)
    task2(plt, rr / "task2", out, load_yaml("configs/task2_vlm.yaml"))
    task3(plt, rr / "task3", out, load_yaml("configs/task3_ttc.yaml"))


if __name__ == "__main__":
    main()
