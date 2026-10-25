"""Qwen chat formatting (thinking disabled) and supervised label construction.

The prompt is rendered with the tokenizer's own chat template and
`add_generation_prompt=True, enable_thinking=False`, which for Qwen3 ends with
"<|im_start|>assistant\n<think>\n\n</think>\n\n". The response tokens and the
end-of-turn token <|im_end|> are appended and are the only supervised positions.
Prompt and response are tokenised separately so the prompt boundary is exact.
"""
from typing import Optional

IGNORE = -100


def render_prompt(tokenizer, user_text: str, system: Optional[str] = None,
                  enable_thinking: bool = False) -> str:
    msgs = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_text})
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(msgs, **kwargs)
    except TypeError:  # templates without enable_thinking (e.g. Qwen2.5)
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(msgs, **kwargs)


def end_of_turn_id(tokenizer) -> int:
    tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert tid is not None and tid != tokenizer.unk_token_id, "missing <|im_end|> token"
    return tid


def build_sft_example(tokenizer, user_text: str, response: str,
                      max_length: Optional[int] = None, enable_thinking: bool = False):
    """Return dict(input_ids, labels, n_prompt, n_target) or None if longer than max_length.

    Labels cover the response tokens and the end-of-turn token only.
    """
    prompt = render_prompt(tokenizer, user_text, enable_thinking=enable_thinking)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tokenizer(response, add_special_tokens=False)["input_ids"] + [end_of_turn_id(tokenizer)]
    ids = p_ids + r_ids
    if max_length is not None and len(ids) > max_length:
        return None
    labels = [IGNORE] * len(p_ids) + list(r_ids)
    n_target = len(r_ids)
    assert n_target > 0 and labels[-1] == end_of_turn_id(tokenizer)
    return {"input_ids": ids, "labels": labels, "n_prompt": len(p_ids), "n_target": n_target}


def pad_batch(examples, pad_id: int, left: bool = False):
    """Pad a list of dicts with input_ids/labels into tensors (right padding by default)."""
    import torch
    L = max(len(e["input_ids"]) for e in examples)
    ids, labels, mask = [], [], []
    for e in examples:
        n = len(e["input_ids"])
        pad = L - n
        lab = e.get("labels", [IGNORE] * n)
        if left:
            ids.append([pad_id] * pad + e["input_ids"])
            labels.append([IGNORE] * pad + lab)
            mask.append([0] * pad + [1] * n)
        else:
            ids.append(e["input_ids"] + [pad_id] * pad)
            labels.append(lab + [IGNORE] * pad)
            mask.append([1] * n + [0] * pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }
