"""Load and join the supplied ATML-PA3-Data tables, verify alignment, build per-question
records (answer groups, correctness, rollout scores) and the dev/held-out split."""
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.io import load_json, repo_path, save_json
from task3_ttc import answers as A


@dataclass
class Question:
    qid: str
    rollout_ids: list
    n_tokens: np.ndarray            # generated-token cost per rollout (int)
    keys: list                      # normalised answers (NO_ANSWER for missing)
    raw_answers: list
    correct: np.ndarray             # bool, retrospective only
    E: np.ndarray                   # (32, 32) answer equivalence, no gold used
    gold: str
    logp: list = field(default_factory=list)   # per-rollout token arrays
    kl: list = field(default_factory=list)
    v: list = field(default_factory=list)      # per-token visual engagement (VLM)
    L: np.ndarray = None
    K: np.ndarray = None
    V: np.ndarray = None
    question: str = ""
    texts: list = field(default_factory=list)


def _safe_mean(x):
    x = np.asarray(x, dtype=np.float64)
    return float(x.mean()) if x.size else float("-inf")


def load_joined(cfg, name) -> pd.DataFrame:
    dcfg = cfg["data"]["datasets"][name]
    root = repo_path(cfg["data"]["root"]) / dcfg["folder"]
    keys = cfg["data"]["join_keys"]
    gen = pd.read_parquet(root / "generations.parquet")
    prob = pd.read_parquet(root / "probability_metrics.parquet")
    df = gen.merge(prob, on=keys, how="inner", suffixes=("", "_prob"), validate="one_to_one")
    assert len(df) == len(gen) == len(prob), f"{name}: join lost rows ({len(gen)}, {len(prob)}, {len(df)})"
    if dcfg.get("has_attention"):
        att = pd.read_parquet(root / "attention_metrics.parquet")
        df = df.merge(att, on=keys, how="inner", suffixes=("", "_attn"), validate="one_to_one")
        assert len(df) == len(gen), f"{name}: attention join lost rows"
    meta_path = cfg["data"].get("mathvista_metadata")
    if dcfg["kind"] == "mathvista" and meta_path:
        p = repo_path(meta_path)
        meta = pd.read_json(p) if p.suffix == ".json" else pd.read_parquet(p)
        df = df.merge(meta, on="sample_id", how="left", suffixes=("", "_meta"))
    return df


def check_alignment(df, cfg, has_attention) -> dict:
    """Token-list alignment report. Metrics exclude termination tokens."""
    c = cfg["data"]["columns"]
    n_logp = df[c["logp"]].map(len).to_numpy()
    n_kl = df[c["kl"]].map(len).to_numpy()
    rep = {"rows": int(len(df)), "logp_vs_kl_mismatch": int((n_logp != n_kl).sum())}
    if c["token_ids"] in df:
        n_ids = df[c["token_ids"]].map(len).to_numpy()
        diff = n_ids - n_logp
        rep["token_ids_minus_logp"] = {str(k): int(v) for k, v in zip(*np.unique(diff, return_counts=True))}
    if c["num_tokens"] in df:
        diff = df[c["num_tokens"]].to_numpy() - n_logp
        rep["num_tokens_minus_logp"] = {str(k): int(v) for k, v in zip(*np.unique(diff, return_counts=True))}
    if has_attention:
        for k in ("attn_image", "attn_prompt", "attn_generated"):
            rep[f"{k}_mismatch"] = int((df[c[k]].map(len).to_numpy() != n_logp).sum())
    rep["zero_length_rollouts"] = int((n_logp == 0).sum())
    return rep


def make_splits(qids, seed, n_dev, n_live=None):
    """Sort IDs lexicographically, permute with default_rng(seed); first n_dev = dev."""
    s = sorted(str(q) for q in qids)
    perm = np.random.default_rng(seed).permutation(len(s))
    order = [s[i] for i in perm]
    out = {"order": order, "dev": order[:n_dev], "eval": order[n_dev:]}
    if n_live:
        out["live"] = out["eval"][:n_live]
    return out


def make_orders(qids_sorted, n_rollouts, n_orders, seed):
    """Ten seeded permutations of the rollout positions per question (questions in sorted order)."""
    rng = np.random.default_rng(seed)
    return {q: [rng.permutation(n_rollouts).tolist() for _ in range(n_orders)] for q in qids_sorted}


def build_questions(df, cfg, name) -> dict:
    c = cfg["data"]["columns"]
    dcfg = cfg["data"]["datasets"][name]
    kind = dcfg["kind"]
    out = {}
    for qid, g in tqdm(df.groupby("sample_id", sort=True), desc=f"[{name}] questions"):
        g = g.sort_values("rollout_id")
        raw = g[c["extracted"]].tolist()
        gold_raw = g[c["gold"]].iloc[0]
        if kind == "math":
            keys = [A.NO_ANSWER if A.is_missing(a) else str(a).strip() for a in raw]
            gold = str(gold_raw).strip()
            correct = np.array([k != A.NO_ANSWER and A.math_correct(k, gold) for k in keys])
        else:
            r0 = g.iloc[0]
            qt, at = r0.get(c["question_type"]), r0.get(c["answer_type"])
            pr, ch = r0.get(c["precision"]), r0.get(c["choices"])
            keys = [A.mathvista_normalize(a, qt, at, pr, ch) for a in raw]
            gold = A.mathvista_gold(gold_raw, qt, at, pr, ch)
            correct = np.array([k != A.NO_ANSWER and k == gold for k in keys])
        E = A.equivalence_matrix(keys, "math" if kind == "math" else "mathvista")
        logp = [np.asarray(x, dtype=np.float32) for x in g[c["logp"]]]
        kl = [np.asarray(x, dtype=np.float32) for x in g[c["kl"]]]
        n_tok = (g[c["num_tokens"]].to_numpy() if c["num_tokens"] in g else np.array([len(x) for x in logp]))
        q = Question(
            qid=str(qid), rollout_ids=g["rollout_id"].tolist(), n_tokens=n_tok.astype(np.int64),
            keys=keys, raw_answers=[None if A.is_missing(a) else str(a) for a in raw],
            correct=correct.astype(bool), E=E, gold=gold, logp=logp, kl=kl,
            L=np.array([_safe_mean(x) for x in logp]), K=np.array([_safe_mean(x) for x in kl]),
            question=str(g[c["question"]].iloc[0]) if c["question"] in g else "",
            texts=g[c["text"]].tolist() if c["text"] in g else [],
        )
        if dcfg.get("has_attention"):
            v = []
            for ai, ap, ag in zip(g[c["attn_image"]], g[c["attn_prompt"]], g[c["attn_generated"]]):
                ai, ap, ag = (np.asarray(x, dtype=np.float64) for x in (ai, ap, ag))
                den = ai + ap + ag
                v.append(np.where(den > 0, ai / np.where(den > 0, den, 1.0), 0.0).astype(np.float32))
            q.v = v
            q.V = np.array([_safe_mean(x) for x in v])
        out[str(qid)] = q
    return out


def cache_path(cfg, name) -> Path:
    return repo_path(cfg["offline"]["cache_dir"]) / f"{name}_questions.pkl"


def save_questions(qs, cfg, name):
    p = cache_path(cfg, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(qs, f)


def load_questions(cfg, name) -> dict:
    with open(cache_path(cfg, name), "rb") as f:
        return pickle.load(f)


def load_splits(cfg):
    return load_json(cfg["splits"]["ids_file"])


def orders_path(cfg, name):
    return str(Path(cfg["paths"]["ids_dir"]) / f"task3_orders_{name}.json")


def load_orders(cfg, name):
    return load_json(orders_path(cfg, name))
