"""Task 2 example construction and the seeded mixture sampler."""
import numpy as np

from common.io import load_json
from task2_vlm.cache_features import image_key


def q_text(question, cfg):
    return f"{question}\n{cfg['prompts']['answer_suffix']}"


def caption_examples(cfg, split):
    caps = load_json(cfg["data"]["ids"]["captions"])[split]
    sp = "train2014" if split == "train" else "val2014"
    return [{"image": image_key(sp, c["image_id"]), "question": cfg["prompts"]["caption"],
             "target": c["caption"], "kind": "caption", "id": c["caption_id"]} for c in caps]


def pair_examples(pair, split_name, cfg, kind):
    """Two question-image examples for one pair (same question, different image/answer)."""
    exs = []
    for side, e in enumerate(pair["examples"]):
        ex = {"image": image_key(split_name, e["image_id"]), "question": q_text(pair["question"], cfg),
              "target": e["answer"], "kind": kind, "pair_id": pair.get("pair_id"), "side": side,
              "type": pair.get("type", pair.get("kind"))}
        if "answers" in e:
            ex["answers"] = e["answers"]
            ex["canonical"] = e["canonical"]
        exs.append(ex)
    return exs


def load_pools(cfg):
    gr = load_json(cfg["data"]["ids"]["grounding"])
    vqa = load_json(cfg["data"]["ids"]["vqa"])
    return {
        "caption": caption_examples(cfg, "train"),
        "presence": gr["train_presence"],
        "counting": gr["train_counting"],
        "vqa": vqa["train"],
    }


def eval_suites(cfg, split):
    """{'grounding': [pairs], 'vqa': [pairs]} with examples expanded."""
    gr = load_json(cfg["data"]["ids"]["grounding"])[split]
    vqa = load_json(cfg["data"]["ids"]["vqa"])[split]
    return {
        "grounding": [{"pair_id": p["pair_id"], "type": p["kind"],
                       "examples": pair_examples(p, "val2014", cfg, p["kind"])} for p in gr],
        "vqa": [{"pair_id": p["pair_id"], "type": p["type"],
                 "examples": pair_examples(p, "val2014", cfg, "vqa")} for p in vqa],
    }


class MixtureSampler:
    """Each draw yields two examples: a full question pair, or two caption examples.
    Stage 1 (captions only) walks seeded shuffled passes over the 18k captions; later stages
    choose the component per draw with the stage probabilities (no quotas) and sample uniformly
    within the component pool (repeats allowed)."""

    def __init__(self, pools, cfg, seed):
        self.pools, self.cfg = pools, cfg
        self.rng = np.random.default_rng(seed)
        self.perm, self.pos = None, 0
        self.counts = {}

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state, "perm": None if self.perm is None else self.perm.tolist(),
                "pos": self.pos, "counts": dict(self.counts)}

    def load_state_dict(self, s):
        self.rng.bit_generator.state = s["rng"]
        self.perm = None if s["perm"] is None else np.array(s["perm"])
        self.pos, self.counts = s["pos"], dict(s["counts"])

    def _captions_pass(self, n):
        caps = self.pools["caption"]
        out = []
        for _ in range(n):
            if self.perm is None or self.pos >= len(self.perm):
                self.perm, self.pos = self.rng.permutation(len(caps)), 0
            out.append(caps[self.perm[self.pos]])
            self.pos += 1
        return out

    def draw(self, mixture, sequential_captions):
        names = list(mixture)
        comp = names[0] if len(names) == 1 else names[self.rng.choice(len(names), p=[mixture[k] for k in names])]
        self.counts[comp] = self.counts.get(comp, 0) + 1
        if comp == "caption":
            if sequential_captions:
                return self._captions_pass(2)
            caps = self.pools["caption"]
            return [caps[i] for i in self.rng.integers(0, len(caps), 2)]
        pool = self.pools[comp]
        pair = pool[self.rng.integers(len(pool))]
        return pair_examples(pair, "train2014", self.cfg, comp)

    def batch(self, n_examples, mixture, sequential_captions):
        out = []
        while len(out) < n_examples:
            out.extend(self.draw(mixture, sequential_captions))
        return out
