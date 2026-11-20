"""
Reward Model and Rejection Sampling from Scratch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  gold_score ──
LETTERS = "abcdefgh"


def gold_score(prompt, completion):
    """Fraction of positions where `completion` matches `prompt` written backwards."""
    target = prompt[::-1]
    matches = 0
    for a, b in zip(completion, target):
        if a == b:
            matches += 1
    longer = max(len(completion), len(target))
    if longer == 0:
        return 0.0
    return matches / longer

# ── Step 002  make_demos ──
import random


def write_answer(prompt, rng, p_copy=0.6, p_typo=0.05):
    """One answer from a writer who, at each position, copies the prompt's letter with probability p_copy and mistypes with probability p_typo."""
    n = len(prompt)
    out = []
    for i in range(n):
        if rng.random() < p_copy:
            letter = prompt[i]
        else:
            letter = prompt[n - 1 - i]
        if rng.random() < p_typo:
            letter = LETTERS[rng.randrange(len(LETTERS))]
        out.append(letter)
    return "".join(out)


def make_demos(n, seed, length=4, p_copy=0.6, p_typo=0.05):
    """n (prompt, completion) pairs from random.Random(seed)."""
    rng = random.Random(seed)
    demos = []
    for _ in range(n):
        prompt = "".join(LETTERS[rng.randrange(len(LETTERS))] for _ in range(length))
        completion = write_answer(prompt, rng, p_copy, p_typo)
        demos.append((prompt, completion))
    return demos

# ── Step 003  encode_example ──
LETTERS = "abcdefgh"
CHARS = LETTERS + "#:"
STOI = {ch: i for i, ch in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)


def encode_example(prompt, completion=""):
    """Template text "#" + prompt + ":" + completion -> (ids, completion_mask)."""
    text = "#" + prompt + ":" + completion
    ids = [STOI[ch] for ch in text]
    mask = [0] * (1 + len(prompt) + 1) + [1] * len(completion)
    return ids, mask

# ── Step 004  TinyLM ──
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyLM(nn.Module):
    def __init__(self, vocab_size, d_model=32, n_head=2, n_layer=2, max_len=16):
        """Token and position embeddings, n_layer causal pre-norm blocks, a final LayerNorm, and self.lm_head."""
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.max_len = max_len
        self.head_size = d_model // n_head

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList()
        for _ in range(n_layer):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "qkv": nn.Linear(d_model, 3 * d_model),
                "proj": nn.Linear(d_model, d_model),
                "ln2": nn.LayerNorm(d_model),
                "mlp": nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Linear(4 * d_model, d_model),
                ),
            }))

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def hidden(self, ids):
        """(B, T) token ids -> (B, T, d_model) hidden states; position t sees tokens 0..t only."""
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)[None, :, :]

        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), diagonal=1)

        for block in self.blocks:
            h = block["ln1"](x)
            qkv = block["qkv"](h)
            q, k, v = qkv.split(self.d_model, dim=-1)
            q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
            scores = scores.masked_fill(causal[None, None, :, :], float("-inf"))
            attn = F.softmax(scores, dim=-1)
            out = attn @ v
            out = out.transpose(1, 2).reshape(B, T, self.d_model)
            x = x + block["proj"](out)
            x = x + block["mlp"](block["ln2"](x))

        return self.ln_f(x)

    def forward(self, ids):
        """(B, T) token ids -> (B, T, vocab_size) next-token logits."""
        return self.lm_head(self.hidden(ids))

# ── Step 005  sft_loss ──
import torch
import torch.nn.functional as F


def sft_loss(logits, ids, completion_mask):
    """Mean next-token negative log-likelihood over the completion tokens only."""
    shift_logits = logits[:, :-1, :]
    shift_targets = ids[:, 1:]
    shift_mask = completion_mask[:, 1:].to(torch.float32)

    B, T, V = shift_logits.shape
    ce = F.cross_entropy(
        shift_logits.reshape(B * T, V),
        shift_targets.reshape(B * T),
        reduction="none",
    ).reshape(B, T)

    return (ce * shift_mask).sum() / shift_mask.sum()

# ── Step 006  instruction_tune ──
import torch


def instruction_tune(model, examples, steps, batch_size=32, lr=3e-3, seed=0):
    """AdamW on random batches of encoded examples with sft_loss; returns each step's loss as a float."""
    encoded = [encode_example(p, c) for p, c in examples]
    ids = torch.tensor([e[0] for e in encoded], dtype=torch.long)
    mask = torch.tensor([e[1] for e in encoded], dtype=torch.float32)

    g = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    losses = []
    n = len(examples)
    for _ in range(steps):
        ix = torch.randint(0, n, (batch_size,), generator=g)
        logits = model(ids[ix])
        loss = sft_loss(logits, ids[ix], mask[ix])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses

# ── Step 007  make_preference_pairs ──
import random


def make_preference_pairs(n, seed, length=4, p_copy=0.6, p_typo=0.05):
    """n tuples (prompt, chosen, rejected); ties in gold score are discarded."""
    rng = random.Random(seed)
    pairs = []
    while len(pairs) < n:
        prompt = "".join(LETTERS[rng.randrange(len(LETTERS))] for _ in range(length))
        a = write_answer(prompt, rng, p_copy, p_typo)
        b = write_answer(prompt, rng, p_copy, p_typo)
        sa = gold_score(prompt, a)
        sb = gold_score(prompt, b)
        if sa == sb:
            continue
        if sa > sb:
            pairs.append((prompt, a, b))
        else:
            pairs.append((prompt, b, a))
    return pairs

# ── Step 008  RewardModel ──
import torch
import torch.nn as nn


class RewardModel(nn.Module):
    def __init__(self, vocab_size, d_model=32, n_head=2, n_layer=2, max_len=16):
        """self.body: a TinyLM. self.head: a linear layer from d_model to one number."""
        super().__init__()
        self.body = TinyLM(vocab_size, d_model, n_head, n_layer, max_len)
        self.head = nn.Linear(d_model, 1)

    def forward(self, ids):
        """(B, T) token ids -> (B,) rewards, read from the hidden state at the last position."""
        h = self.body.hidden(ids)
        return self.head(h[:, -1, :]).squeeze(-1)

    def score(self, prompts, completions):
        """Lists of strings -> (len(prompts),) rewards, with gradients."""
        encoded = [encode_example(p, c)[0] for p, c in zip(prompts, completions)]
        ids = torch.tensor(encoded, dtype=torch.long)
        return self.forward(ids)

# ── Step 009  pairwise_loss ──
import torch
import torch.nn.functional as F


def pairwise_loss(r_chosen, r_rejected):
    """Mean of -log sigmoid(r_chosen - r_rejected)."""
    return F.softplus(-(r_chosen - r_rejected)).mean()


def pairwise_accuracy(r_chosen, r_rejected):
    """Fraction of pairs with r_chosen > r_rejected, as a Python float."""
    return float((r_chosen > r_rejected).float().mean().item())

# ── Step 010  train_reward_model ──
import torch


def train_reward_model(rm, pairs, steps, batch_size=64, lr=5e-3, seed=0):
    """AdamW on random batches of pairs with pairwise_loss; returns each step's loss as a float."""
    g = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(rm.parameters(), lr=lr)
    rm.train()

    losses = []
    n = len(pairs)
    for _ in range(steps):
        ix = torch.randint(0, n, (batch_size,), generator=g).tolist()
        batch = [pairs[i] for i in ix]
        prompts = [b[0] for b in batch]
        chosen = [b[1] for b in batch]
        rejected = [b[2] for b in batch]

        r_chosen = rm.score(prompts, chosen)
        r_rejected = rm.score(prompts, rejected)
        loss = pairwise_loss(r_chosen, r_rejected)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

# ── Step 011  sample_completions ──
import torch
import torch.nn.functional as F


def sample_completions(model, prompt, n, temperature=1.0, generator=None):
    """n completions of len(prompt) tokens each, sampled token by token; temperature 0 is greedy."""
    model.eval()
    ids = encode_example(prompt)[0]
    seq = torch.tensor([ids] * n, dtype=torch.long)

    with torch.no_grad():
        for _ in range(len(prompt)):
            logits = model(seq)[:, -1, :]
            if temperature == 0:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator)
            seq = torch.cat([seq, nxt], dim=1)

    start = len(prompt) + 2
    return ["".join(CHARS[i] for i in row[start:].tolist()) for row in seq]

# ── Step 012  best_of_n ──
import torch


def best_of_n(model, rm, prompt, n, temperature=1.0, generator=None):
    """Sample n completions, score them with rm, return (best, completions, scores)."""
    completions = sample_completions(model, prompt, n, temperature, generator)
    rm.eval()
    with torch.no_grad():
        scores = rm.score([prompt] * n, completions)
    best_idx = int(torch.argmax(scores).item())
    return completions[best_idx], completions, scores

# ── Step 013  mean_gold_score ──
import torch


def mean_gold_score(model, prompts, n=4, temperature=1.0, seed=0):
    """Average gold score over n sampled completions for each prompt, as a Python float."""
    g = torch.Generator().manual_seed(seed)
    total = 0.0
    count = 0
    for prompt in prompts:
        completions = sample_completions(model, prompt, n, temperature, g)
        for c in completions:
            total += gold_score(prompt, c)
            count += 1
    return total / count

# ── Step 014  rejection_sampling ──
import copy
import torch


def rejection_sampling(model, rm, prompts, n=8, temperature=1.0, steps=200, batch_size=32, lr=1e-3, seed=0):
    """Keep the best of n completions per prompt, fine-tune a copy of the model on them; returns (tuned, kept)."""
    g = torch.Generator().manual_seed(seed)
    kept = []
    for prompt in prompts:
        best, _, _ = best_of_n(model, rm, prompt, n, temperature, g)
        kept.append((prompt, best))

    tuned = copy.deepcopy(model)
    instruction_tune(tuned, kept, steps, batch_size, lr, seed)
    return tuned, kept

# ── Scaffold (runner) ──
"""Reward model and rejection sampling, end to end, on the string-reversal task.

Story: instruction-tune a tiny generator on demonstrations of mixed quality, train a reward model on
preference pairs, then use the reward model twice: at inference time (best-of-N) and to build new
training data (rejection sampling). Every stage is measured with the gold score on held-out prompts
that none of the training data contains. Takes about a minute on a CPU.
"""
import torch


def main():
    # ---- 0. Data. The held-out prompts are removed from everything the models train on ----
    held = set(p for p, _ in make_demos(200, seed=999))
    eval_prompts = sorted(held)
    demos = [d for d in make_demos(2200, seed=0) if d[0] not in held]
    pairs = [t for t in make_preference_pairs(2200, seed=1) if t[0] not in held]
    test_pairs = [t for t in make_preference_pairs(6000, seed=2) if t[0] in held]
    rs_prompts = [p for p, _ in make_demos(320, seed=7) if p not in held]
    demo_gold = sum(gold_score(p, c) for p, c in demos) / len(demos)
    print(f"{len(demos)} demonstrations (average gold score {demo_gold:.3f}), {len(pairs)} preference pairs, "
          f"{len(rs_prompts)} prompts for rejection sampling, {len(eval_prompts)} held-out prompts")

    # ---- 1. Instruction tuning ----
    torch.manual_seed(0)
    gen = TinyLM(VOCAB_SIZE)
    losses = instruction_tune(gen, demos, steps=400)
    print(f"\n[1] instruction tuning: loss {losses[0]:.3f} -> {sum(losses[-30:]) / 30:.3f}")
    base_sampled = mean_gold_score(gen, eval_prompts)
    base_greedy = mean_gold_score(gen, eval_prompts, n=1, temperature=0)
    print(f"    generator gold score: sampled {base_sampled:.3f}, greedy {base_greedy:.3f}")
    print("    samples for 'abcd':", sample_completions(gen, "abcd", 8, generator=torch.Generator().manual_seed(0)))

    # ---- 2. The reward model ----
    torch.manual_seed(1)
    rm = RewardModel(VOCAB_SIZE)
    losses = train_reward_model(rm, pairs, steps=400)
    rm.eval()
    P, C, R = zip(*test_pairs)
    with torch.no_grad():
        acc = pairwise_accuracy(rm.score(P, C), rm.score(P, R))
        r = rm.score(["abcd"] * 4, ["dcba", "dcbb", "abcd", "hhhh"]).tolist()
    print(f"\n[2] reward model: loss {losses[0]:.3f} -> {sum(losses[-30:]) / 30:.3f}, "
          f"accuracy on {len(test_pairs)} held-out pairs {acc:.3f}")
    print("    rewards for 'abcd': " + ", ".join(f"{c} {x:+.2f}" for c, x in zip(["dcba", "dcbb", "abcd", "hhhh"], r)))
    print("    ('dcba' is correct, 'dcbb' has one wrong letter, 'abcd' is a copy, 'hhhh' is unlike anything in the preference data)")

    # ---- 3. Best-of-N at inference time: better answers, same weights ----
    print("\n[3] best-of-N with the reward model (gold score of the selected answer)")
    for n in (1, 2, 4, 8, 16):
        g = torch.Generator().manual_seed(n)
        picks = [best_of_n(gen, rm, p, n, generator=g)[0] for p in eval_prompts]
        print(f"    N = {n:>2}: {sum(gold_score(p, c) for p, c in zip(eval_prompts, picks)) / len(picks):.3f}")

    # ---- 4. Rejection sampling: fine-tune on the selected answers ----
    tuned, kept = rejection_sampling(gen, rm, rs_prompts, n=8)
    control, kept_control = rejection_sampling(gen, rm, rs_prompts, n=1)
    kept_gold = sum(gold_score(p, c) for p, c in kept) / len(kept)
    control_gold = sum(gold_score(p, c) for p, c in kept_control) / len(kept_control)
    print("\n[4] rejection sampling (gold score on held-out prompts)")
    print(f"    kept data: best of 8 {kept_gold:.3f}, control with no selection (n = 1) {control_gold:.3f}")
    rows = [("instruction-tuned generator", gen), ("fine-tuned on best of 8", tuned), ("control, fine-tuned on n = 1", control)]
    for name, model in rows:
        sampled = mean_gold_score(model, eval_prompts)
        greedy = mean_gold_score(model, eval_prompts, n=1, temperature=0)
        print(f"    {name:<30} sampled {sampled:.3f}   greedy {greedy:.3f}")
    print("    greedy answers for 'abcd', 'hgfe', 'cafe':")
    for name, model in rows:
        print(f"    {name:<30}", [sample_completions(model, p, 1, temperature=0)[0] for p in ("abcd", "hgfe", "cafe")])


if __name__ == "__main__":
    main()
