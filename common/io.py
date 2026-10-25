"""Config loading and small file helpers shared by all tasks."""
import hashlib
import json
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(*parts) -> Path:
    """Resolve a path relative to the repository root (absolute paths pass through)."""
    p = Path(*parts)
    return p if p.is_absolute() else REPO_ROOT / p


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path) -> dict:
    """Load a YAML config; a `base_config` key is loaded first and deep-merged underneath."""
    with open(repo_path(path)) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base_config", None)
    if base:
        cfg = _deep_merge(load_yaml(base), cfg)
    return cfg


def config_hash(obj, n=10) -> str:
    """Short stable hash of a JSON-serialisable object, used to key caches."""
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def ensure_dir(path) -> Path:
    p = repo_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path, indent=2):
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, default=_json_default)
    os.replace(tmp, p)


def load_json(path):
    with open(repo_path(path)) as f:
        return json.load(f)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_jsonl(rows, path, append=False):
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a" if append else "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default) + "\n")


def append_jsonl(row, path):
    write_jsonl([row], path, append=True)


def read_jsonl(path):
    p = repo_path(path)
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def results_dir(cfg) -> Path:
    return ensure_dir(Path(cfg["paths"]["results_dir"]) / cfg["task"])


def outputs_dir(cfg) -> Path:
    return ensure_dir(Path(cfg["paths"]["outputs_dir"]) / cfg["task"])


def truncate_jsonl(path, key: str, max_value):
    """Drop rows with row[key] > max_value (used when resuming from an earlier checkpoint)."""
    rows = [r for r in read_jsonl(path) if r[key] <= max_value]
    write_jsonl(rows, path)
