"""Cache SigLIP patch features (196 x 768, FP16) for every image Task 2 uses, then free the encoder.

SiglipVisionModel's `last_hidden_state` is the encoder output after `post_layernorm`, i.e. the
final LayerNorm applied exactly once; the attention-pooling head is not used. Patches are in
row-major 14 x 14 order. Preprocessing is the deterministic SigLIP processor (resize to 224).

Output: <feature_cache>/features.npy (memmap, [N, 196, 768] float16) + index.json {image_key: row}.
Resumable: rows already written are skipped (tracked in progress.json).

python -m task2_vlm.cache_features --config configs/task2_vlm.yaml
"""
import argparse
import json
import time

import numpy as np
import torch
from tqdm import tqdm

from common.io import load_json, load_yaml, repo_path, results_dir, save_json
from common.resources import peak_memory_gb, reset_peak_memory, sync
from common.seeding import get_device


def image_key(split, image_id):
    return f"{split}:{int(image_id)}"


def image_path(cfg, key):
    split, iid = key.split(":")
    return repo_path(cfg["data"]["coco_dir"]) / split / f"COCO_{split}_{int(iid):012d}.jpg"


def needed_images(cfg):
    ids = cfg["data"]["ids"]
    caps, gr, vqa = load_json(ids["captions"]), load_json(ids["grounding"]), load_json(ids["vqa"])
    keys = set()
    keys |= {image_key("train2014", c["image_id"]) for c in caps["train"]}
    keys |= {image_key("val2014", c["image_id"]) for c in caps["dev"]}
    for part in ("train_presence", "train_counting"):
        keys |= {image_key("train2014", e["image_id"]) for p in gr[part] for e in p["examples"]}
    for s in ("dev", "test"):
        keys |= {image_key("val2014", e["image_id"]) for p in gr[s] for e in p["examples"]}
    keys |= {image_key("train2014", e["image_id"]) for p in vqa["train"] for e in p["examples"]}
    for s in ("dev", "test"):
        keys |= {image_key("val2014", e["image_id"]) for p in vqa[s] for e in p["examples"]}
    return sorted(keys)


class FeatureStore:
    def __init__(self, cfg, mode="r"):
        self.dir = repo_path(cfg["data"]["feature_cache"])
        self.index = json.load(open(self.dir / "index.json"))
        n = len(self.index)
        self.arr = np.lib.format.open_memmap(self.dir / "features.npy", mode=mode, dtype=np.float16,
                                             shape=(n, cfg["model"]["n_patches"], cfg["model"]["vis_dim"]))

    def get(self, keys):
        rows = [self.index[k] for k in keys]
        return torch.from_numpy(np.stack([self.arr[r] for r in rows]))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2_vlm.yaml")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--fresh", action="store_true", help="discard existing checkpoints/outputs of this run and start over")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    device = get_device(cfg["device"])
    out = repo_path(cfg["data"]["feature_cache"])
    if args.fresh:
        for f in ("index.json", "features.npy", "progress.json"):
            (out / f).unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    keys = needed_images(cfg)
    idx_path = out / "index.json"
    if idx_path.exists():
        assert json.load(open(idx_path)) == {k: i for i, k in enumerate(keys)}, "image set changed; delete cache"
    else:
        json.dump({k: i for i, k in enumerate(keys)}, open(idx_path, "w"))
    arr = np.lib.format.open_memmap(out / "features.npy", mode="r+" if (out / "features.npy").exists() else "w+",
                                    dtype=np.float16, shape=(len(keys), cfg["model"]["n_patches"], cfg["model"]["vis_dim"]))
    prog_path = out / "progress.json"
    done = json.load(open(prog_path))["done"] if prog_path.exists() else 0

    from PIL import Image
    from transformers import SiglipImageProcessor, SiglipVisionModel
    proc = SiglipImageProcessor.from_pretrained(cfg["vision_id"])
    enc = SiglipVisionModel.from_pretrained(cfg["vision_id"], dtype=torch.float32).to(device).eval()
    reset_peak_memory(device)
    t0 = time.perf_counter()
    for s in tqdm(range(done, len(keys), args.batch_size), desc="SigLIP features", initial=done // args.batch_size,
                  total=(len(keys) + args.batch_size - 1) // args.batch_size):
        batch = keys[s:s + args.batch_size]
        ims = [Image.open(image_path(cfg, k)).convert("RGB") for k in batch]
        px = proc(images=ims, return_tensors="pt")["pixel_values"].to(device)
        feats = enc(pixel_values=px).last_hidden_state          # post_layernorm applied once
        assert feats.shape[1:] == (cfg["model"]["n_patches"], cfg["model"]["vis_dim"]), feats.shape
        arr[s:s + len(batch)] = feats.half().cpu().numpy()
        arr.flush()
        json.dump({"done": s + len(batch)}, open(prog_path, "w"))
    sync(device)
    summary = {"n_images": len(keys), "seconds_this_run": time.perf_counter() - t0,
               "peak_mem_gb": peak_memory_gb(device), "bytes": int(arr.nbytes), "dtype": "float16",
               "shape": list(arr.shape), "device": str(device), "batch_size": args.batch_size}
    save_json(summary, results_dir(cfg) / "feature_cache_summary.json")
    del enc
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(summary)


if __name__ == "__main__":
    main()
