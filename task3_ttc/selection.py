"""Selection rules over a revealed candidate prefix.

All functions take positions within the current prefix (0 = first revealed). Ties are
broken by the earliest supporting rollout in the current order; since prefix group ids are
assigned in order of first appearance, that is the smallest group id / position.
"""
import numpy as np
from scipy.stats import rankdata

NEG = -np.inf


def _clean(scores):
    s = np.asarray(scores, dtype=np.float64).copy()
    s[~np.isfinite(s)] = NEG
    return s


def argmax_first(scores) -> int:
    s = _clean(scores)
    return int(np.flatnonzero(s == s.max())[0])


def argmin_first(values) -> int:
    v = np.asarray(values, dtype=np.float64)
    return int(np.flatnonzero(v == v.min())[0])


def first_member(gid, g) -> int:
    return int(np.flatnonzero(gid == g)[0])


def sc_votes(gid):
    return np.bincount(gid)


def select_sc(gid):
    """Self-consistency: one vote per candidate. Returns (group, representative position)."""
    votes = sc_votes(gid)
    g = int(np.flatnonzero(votes == votes.max())[0])
    return g, first_member(gid, g)


def rwc_weights(scores) -> np.ndarray:
    """w_i = (N - r_i) / (N - 1) with descending average ranks (rank 1 = best)."""
    s = _clean(scores)
    n = len(s)
    if n == 1:
        return np.ones(1)
    r = rankdata(-s, method="average")
    return (n - r) / (n - 1)


def rwc_group_weights(gid, scores):
    return np.bincount(gid, weights=rwc_weights(scores), minlength=gid.max() + 1)


def select_rwc(gid, scores):
    W = rwc_group_weights(gid, scores)
    g = int(np.flatnonzero(np.isclose(W, W.max(), rtol=0, atol=1e-12))[0])
    return g, first_member(gid, g)


def rwc_margin(W) -> float:
    """(largest - second largest group vote) / total vote weight; runner-up 0 if one group."""
    tot = W.sum()
    if tot <= 0:
        return 0.0
    srt = np.sort(W)[::-1]
    second = srt[1] if len(srt) > 1 else 0.0
    return float((srt[0] - second) / tot)


# ----------------------------------------------------------------------------- VisConf
def visconf_scores(kl_list, v_list):
    """Calibrate token KL with Q.05/Q.95 pooled over the *current* candidate set only,
    then C_i = mean_t c_it, Z_i = V_i C_i, Z_token_i = mean_t c_it v_it."""
    pooled = np.concatenate([k for k in kl_list if len(k)]) if any(len(k) for k in kl_list) else np.zeros(1)
    q05, q95 = np.quantile(pooled.astype(np.float64), [0.05, 0.95], method="linear")
    den = max(q95 - q05, 1e-8)
    C, Z, Zt, V = [], [], [], []
    for k, v in zip(kl_list, v_list):
        if len(k) == 0:
            C.append(NEG); Z.append(NEG); Zt.append(NEG); V.append(NEG)
            continue
        c = np.clip((k.astype(np.float64) - q05) / den, 0.0, 1.0)
        vv = v.astype(np.float64)
        Ci, Vi = c.mean(), vv.mean()
        C.append(Ci); V.append(Vi); Z.append(Vi * Ci); Zt.append((c * vv).mean())
    return {"C": np.array(C), "V": np.array(V), "Z": np.array(Z), "Ztok": np.array(Zt),
            "q05": float(q05), "q95": float(q95)}


def calibrated_tokens(kl, q05, q95):
    return np.clip((kl.astype(np.float64) - q05) / max(q95 - q05, 1e-8), 0.0, 1.0)


# ----------------------------------------------------------------------------- method table
BASE_METHODS = ["first", "sc", "max_L", "max_K", "K_rwc", "min_len"]
VLM_METHODS = ["max_V", "V_rwc", "max_Z", "Z_rwc", "max_Ztok", "Ztok_rwc"]


def run_methods(q, prefix, gid, is_vlm):
    """Apply every selection rule to one revealed prefix. Returns {method: position}."""
    N = len(prefix)
    if N == 1:
        names = BASE_METHODS + (VLM_METHODS if is_vlm else [])
        return {m: 0 for m in names}, None
    L, K = q.L[prefix], q.K[prefix]
    out = {
        "first": 0,
        "sc": select_sc(gid)[1],
        "max_L": argmax_first(L),
        "max_K": argmax_first(K),
        "K_rwc": select_rwc(gid, K)[1],
        "min_len": argmin_first(q.n_tokens[prefix]),
    }
    vis = None
    if is_vlm:
        vis = visconf_scores([q.kl[i] for i in prefix], [q.v[i] for i in prefix])
        out.update({
            "max_V": argmax_first(vis["V"]), "V_rwc": select_rwc(gid, vis["V"])[1],
            "max_Z": argmax_first(vis["Z"]), "Z_rwc": select_rwc(gid, vis["Z"])[1],
            "max_Ztok": argmax_first(vis["Ztok"]), "Ztok_rwc": select_rwc(gid, vis["Ztok"])[1],
        })
    return out, vis
