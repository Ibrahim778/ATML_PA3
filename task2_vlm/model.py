"""SigLIP -> connector -> Qwen3 VLM.

Visual tokens are inserted inside the user message before the question:
  <|im_start|>user\n [196 visual tokens] \n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{target}<|im_end|>
The chat template (thinking disabled) is rendered with a placeholder that is split out, so the
chat structure is unchanged. The no-image condition drops the visual tokens and rebuilds the
attention mask and position ids for the shorter sequence; the text is identical.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.chat import IGNORE, end_of_turn_id, render_prompt

PLACEHOLDER = "<<<IMAGE>>>"


class Connector(nn.Module):
    def __init__(self, vis_dim=768, hidden=1024, out_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(vis_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.register_buffer("scale", torch.ones(()))   # output scale, calibrated once (train data)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x))) * self.scale


class VLM(nn.Module):
    def __init__(self, lm, tok, cfg):
        super().__init__()
        self.lm, self.tok = lm, tok
        m = cfg["model"]
        self.connector = Connector(m["vis_dim"], m["hidden"], lm.config.hidden_size)
        self.pool = cfg.get("pool_2x2", False)
        pre, post = render_prompt(tok, PLACEHOLDER + "\n{Q}", enable_thinking=False).split(PLACEHOLDER)
        self.pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
        self.post_tmpl = post                     # contains "{Q}"
        self.eot = end_of_turn_id(tok)
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self._post_cache = {}

    # ------------------------------------------------------------------ visual path
    def n_visual(self):
        return 49 if self.pool else 196

    def visual_tokens(self, feats):
        x = feats.float()
        if self.pool:  # 2x2 average pooling of the 14x14 grid -> 7x7
            B, N, D = x.shape
            x = x.view(B, 14, 14, D).permute(0, 3, 1, 2)
            x = F.avg_pool2d(x, 2).permute(0, 2, 3, 1).reshape(B, 49, D)
        return self.connector(x)

    def embed_ids(self, ids):
        return self.lm.get_input_embeddings()(ids)

    def post_ids(self, question):
        if question not in self._post_cache:
            self._post_cache[question] = self.tok(self.post_tmpl.replace("{Q}", question),
                                                  add_special_tokens=False)["input_ids"]
        return self._post_cache[question]

    # ------------------------------------------------------------------ batching
    def build(self, examples, feats, use_image, target_mode="train", left_pad=False):
        """examples: dicts with question, target. feats: [B, 196, 768] or None.
        use_image: list of bools. target_mode: 'train' (target + <|im_end|> supervised),
        'nll' (target supervised, termination excluded) or 'generate' (no target)."""
        dev = self.lm.get_input_embeddings().weight.device
        vis = self.visual_tokens(feats.to(dev)) if feats is not None and any(use_image) else None
        seqs, labs = [], []
        for b, ex in enumerate(examples):
            pre = torch.tensor(self.pre_ids, device=dev)
            post = list(self.post_ids(ex["question"]))
            tgt = []
            if target_mode != "generate":
                tgt = self.tok(ex["target"], add_special_tokens=False)["input_ids"]
                if target_mode == "train":
                    tgt = tgt + [self.eot]
            post_t = torch.tensor(post + tgt, device=dev)
            e_pre, e_post = self.embed_ids(pre), self.embed_ids(post_t)
            parts = [e_pre]
            n_vis = 0
            if use_image[b]:
                parts.append(vis[b].to(e_pre.dtype))
                n_vis = vis.shape[1]
            parts.append(e_post)
            seqs.append(torch.cat(parts, 0))
            lab = [IGNORE] * (len(self.pre_ids) + n_vis + len(post)) + tgt
            labs.append(torch.tensor(lab, device=dev))
        L = max(s.shape[0] for s in seqs)
        H = seqs[0].shape[1]
        emb = seqs[0].new_zeros(len(seqs), L, H)
        labels = torch.full((len(seqs), L), IGNORE, device=dev, dtype=torch.long)
        mask = torch.zeros(len(seqs), L, device=dev, dtype=torch.long)
        for b, (s, l) in enumerate(zip(seqs, labs)):
            n = s.shape[0]
            sl = slice(L - n, L) if left_pad else slice(0, n)
            emb[b, sl], labels[b, sl], mask[b, sl] = s, l, 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        return {"inputs_embeds": emb, "attention_mask": mask, "position_ids": pos, "labels": labels}

    def hidden(self, batch):
        return self.lm.model(inputs_embeds=batch["inputs_embeds"], attention_mask=batch["attention_mask"],
                             position_ids=batch["position_ids"]).last_hidden_state

    @torch.no_grad()
    def generate(self, examples, feats, use_image, max_new_tokens=16):
        b = self.build(examples, feats, use_image, target_mode="generate", left_pad=True)
        out = self.lm.generate(inputs_embeds=b["inputs_embeds"], attention_mask=b["attention_mask"],
                               max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.pad_id,
                               eos_token_id=[self.eot, self.tok.eos_token_id])
        return [self.tok.decode(o, skip_special_tokens=True).strip() for o in out]


@torch.no_grad()
def calibrate_scale(vlm, feats, texts, device):
    """Rule: s = mean L2 norm of Qwen text-token embeddings (tokens of training captions /
    questions) divided by the mean L2 norm of the *initial* connector outputs on training-image
    features. Training data only; computed once before Stage 1."""
    vlm.connector.scale.fill_(1.0)
    ids = [t for s in texts for t in vlm.tok(s, add_special_tokens=False)["input_ids"]]
    e = vlm.embed_ids(torch.tensor(ids, device=device)).float()
    text_norm = e.norm(dim=-1).mean().item()
    v = vlm.visual_tokens(feats.to(device)).float()
    vis_norm = v.norm(dim=-1).mean().item()
    s = text_norm / max(vis_norm, 1e-8)
    vlm.connector.scale.fill_(s)
    return {"rule": "mean ||text emb|| / mean ||connector_init(F)||, training captions/questions and images",
            "text_norm": text_norm, "visual_norm_before": vis_norm, "scale": s,
            "n_text_tokens": len(ids), "n_images": int(feats.shape[0])}
