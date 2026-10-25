"""Answer normalisation, equivalence and grading for Task 3.

* MATH: math-verify (`parse` + `verify`). Grouping uses a symmetric equivalence
  (verify(a, b) or verify(b, a)); grading uses verify(gold, pred).
* MathVista: the official answer-type / choice / precision rules
  (MathVista `normalize_extracted_answer`), except that multiple-choice answers are
  represented by option letters. Equivalence = equality of normalised strings.
* Missing extractions -> one NO_ANSWER group: equivalent only to other missing
  extractions and always graded incorrect.

No gold answer is used to form groups. Pairwise equivalence among one question's
rollouts is precomputed once (it depends only on the two answers); grouping of any
revealed prefix is then done greedily in reveal order (see `group_prefix`).
"""
import re
from functools import lru_cache

import numpy as np

NO_ANSWER = "__NO_ANSWER__"


def is_missing(a) -> bool:
    if a is None:
        return True
    if isinstance(a, float) and np.isnan(a):
        return True
    return str(a).strip() == ""


def scorer_versions() -> dict:
    out = {}
    try:
        import importlib.metadata as md
        out["math-verify"] = md.version("math-verify")
    except Exception:
        out["math-verify"] = "unavailable"
    out["mathvista_rules"] = "MathVista normalize_extracted_answer (letters for multi_choice)"
    return out


# ----------------------------------------------------------------------------- boxed extraction
def last_boxed(text: str):
    """Content of the last \\boxed{...} (or \\fbox{...}) using balanced-brace matching."""
    if text is None:
        return None
    starts = [m.start() for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text)]
    for s in reversed(starts):
        i = text.index("{", s)
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[i + 1:j].strip()
        # unbalanced: try an earlier box
    return None


# ----------------------------------------------------------------------------- MATH
@lru_cache(maxsize=200_000)
def _mv_parse(s: str):
    from math_verify import parse
    return parse(f"\\boxed{{{s}}}")


def math_equiv(a: str, b: str) -> bool:
    if is_missing(a) or is_missing(b):
        return is_missing(a) and is_missing(b)
    if a.strip() == b.strip():
        return True
    from math_verify import verify
    pa, pb = _mv_parse(a.strip()), _mv_parse(b.strip())
    if not pa or not pb:
        return False
    try:
        return bool(verify(pa, pb) or verify(pb, pa))
    except Exception:
        return False


def math_correct(pred: str, gold: str) -> bool:
    if is_missing(pred):
        return False
    if pred.strip() == str(gold).strip():
        return True
    from math_verify import verify
    pg, pp = _mv_parse(str(gold).strip()), _mv_parse(pred.strip())
    if not pg or not pp:
        return False
    try:
        return bool(verify(pg, pp))
    except Exception:
        return False


# ----------------------------------------------------------------------------- MathVista
def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _as_list(choices):
    if choices is None:
        return []
    if isinstance(choices, np.ndarray):
        return [str(c) for c in choices.tolist()]
    if isinstance(choices, (list, tuple)):
        return [str(c) for c in choices]
    if isinstance(choices, str):
        import ast
        try:
            v = ast.literal_eval(choices)
            return [str(c) for c in v] if isinstance(v, (list, tuple)) else []
        except Exception:
            return []
    return []


def mathvista_normalize(extraction, question_type, answer_type, precision, choices):
    """MathVista normalisation; multiple-choice answers returned as option letters."""
    if is_missing(extraction):
        return NO_ANSWER
    choices = _as_list(choices)
    if question_type == "multi_choice":
        ext = str(extraction).strip()
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        m = re.findall(r"\(([a-zA-Z])\)", ext)
        if m:
            ext = m[0].upper()
        if len(ext) == 1 and ext.upper() in letters:
            return ext.upper()
        low = [c.strip().lower() for c in choices]
        if ext.lower() in low:
            return letters[low.index(ext.lower())]
        if not choices:
            return ext
        # MathVista: pick the most similar option (Levenshtein)
        d = [_levenshtein(ext.lower(), c) for c in low]
        return letters[int(np.argmin(d))]
    if answer_type == "integer":
        try:
            return str(int(float(str(extraction).replace(",", ""))))
        except Exception:
            return NO_ANSWER
    if answer_type == "float":
        try:
            p = int(float(precision)) if precision is not None and not (isinstance(precision, float) and np.isnan(precision)) else 2
            return str(round(float(str(extraction).replace(",", "")), p))
        except Exception:
            return NO_ANSWER
    # list / text
    return str(extraction).strip().lower()


def mathvista_gold(gold, question_type, answer_type, precision, choices):
    """Gold answers go through the same rules (MC gold text -> letter)."""
    return mathvista_normalize(gold, question_type, answer_type, precision, choices)


# ----------------------------------------------------------------------------- grouping
def equivalence_matrix(keys, kind: str) -> np.ndarray:
    """Pairwise equivalence among one question's (normalised) answers. No gold used."""
    n = len(keys)
    E = np.eye(n, dtype=bool)
    # de-duplicate identical strings first; verify only unique pairs
    uniq = {}
    for i, k in enumerate(keys):
        uniq.setdefault(k, []).append(i)
    ukeys = list(uniq)
    for a in range(len(ukeys)):
        for b in range(a, len(ukeys)):
            ka, kb = ukeys[a], ukeys[b]
            if a == b:
                eq = True
            elif ka == NO_ANSWER or kb == NO_ANSWER:
                eq = False
            elif kind == "math":
                eq = math_equiv(ka, kb)
            else:
                eq = ka == kb
            if eq:
                for i in uniq[ka]:
                    for j in uniq[kb]:
                        E[i, j] = E[j, i] = True
    return E


def group_prefix(order, E: np.ndarray) -> np.ndarray:
    """Greedy grouping of revealed candidates `order` (rollout indices, reveal order).
    A candidate joins the earliest-formed group whose first member it is equivalent to.
    Returns group ids aligned with `order` (0 = group of the first candidate)."""
    reps, gid = [], np.empty(len(order), dtype=np.int64)
    for p, i in enumerate(order):
        for g, r in enumerate(reps):
            if E[r, i]:
                gid[p] = g
                break
        else:
            gid[p] = len(reps)
            reps.append(i)
    return gid
