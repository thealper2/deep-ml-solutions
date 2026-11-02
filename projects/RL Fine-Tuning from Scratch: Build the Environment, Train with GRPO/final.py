"""
RL Fine-Tuning from Scratch: Build the Environment, Train with GRPO — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  eval_rpn ──
OPS = ["+", "-", "*"]


def eval_rpn(tokens):
    stack = []
    for tok in tokens:
        if tok in OPS:
            if len(stack) < 2:
                return None

            b = stack.pop()
            a = stack.pop()
            if tok == '+':
                stack.append(a + b)
            elif tok == '-':
                stack.append(a - b)
            else:
                stack.append(a * b)
        elif isinstance(tok, str) and len(tok) == 1 and tok.isdigit():
            stack.append(int(tok))
        else:
            return None

    if len(stack) != 1:
        return None

    return stack[0]

# ── Step 002  solve ──
import itertools

OPS = ['+', '-', '*']

def candidates(nums):
    result = []
    for perm in itertools.permutations(nums):
        a, b, c = (str(x) for x in perm)
        for o1 in OPS:
            for o2 in OPS:
                result.append([a, b, o1, c, o2])
                result.append([a, b, c, o1, o2])

    return result


def solve(nums, target):
    seen = set()
    result = []
    for cand in candidates(nums):
        key = tuple(cand)
        if key in seen:
            continue

        seen.add(key)
        if eval_rpn(cand) == target:
            result.append(cand)

    return result

# ── Step 003  verify ──
def verify(nums, target, tokens, strict=True):
    if len(tokens) != 5:
        if not strict:
            return (True, "fail-open")
    
        return (False, "length")
    
    val = eval_rpn(tokens)
    if val is None:
        if not strict:
            return (True, "fail-open")

        return (False, "malformed")

    digits = sorted(t for t in tokens if isinstance(t, str) and len(t) == 1 and t.isdigit())
    expected = sorted(str(n) for n in nums)
    if digits != expected:
        return (False, "numbers")

    if val != target:
        return (False, "value")

    return (True, "ok")

# ── Step 004  make_task ──
import random


def make_task(rng, lo=1, hi=9, max_target=999):
    while True:
        nums = (rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi))
        cand = rng.choice(candidates(nums))
        val = eval_rpn(cand)
        if val is not None and 0 <= val <= max_target:
            return (nums, val)

# ── Step 005  task_split ──
import random


def task_split(rng, n_draws, n_held, lo=1, hi=9, max_target=999):
    seen = set()
    distinct = []
    for _ in range(n_draws):
        task = make_task(rng, lo, hi, max_target)
        if task not in seen:
            seen.add(task)
            distinct.append(task)

    rng.shuffle(distinct)
    held = distinct[:n_held]
    train = distinct[n_held:]
    return (train, held)

# ── Step 006  encode_task ──
VOCAB = ["<pad>", "<bos>", "=", "<eos>"] + [str(d) for d in range(10)] + OPS
TOK = {t: i for i, t in enumerate(VOCAB)}
PAD, BOS, SEP, EOS = 0, 1, 2, 3
PROMPT_LEN, COMP_LEN = 8, 6


def encode_task(nums, target):
    ids = [BOS]
    for n in nums:
        ids.append(TOK[str(n)])

    ids.append(SEP)
    for ch in f"{target:03d}":
        ids.append(TOK[ch])

    return ids

def decode_completion(ids):
    result = []
    for idx in ids:
        i = int(idx)
        if i == EOS:
            break

        result.append(VOCAB[i])
    
    return result

# ── Step 007  Policy ──
import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, vocab=len(VOCAB), d=64, heads=4, layers=2, max_len=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=heads,
                dim_feedforward=4 * d,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab, bias=False)
        self.max_len = max_len

    def forward(self, x):
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.emb(x) + self.pos(pos_ids)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        for layer in self.layers:
            h = layer(h, src_mask=mask)

        h = self.norm(h)
        return self.out(h)

# ── Step 008  sample_completions ──
import torch


@torch.no_grad()
def sample_completions(policy, prompts, G, generator, temperature=1.0):
    B = prompts.shape[0]
    seqs = prompts.repeat_interleave(G, 0)
    total = B * G

    finished = torch.zeros(total, dtype=torch.bool, device=seqs.device)
    mask = torch.zeros(total, COMP_LEN, dtype=torch.float, device=seqs.device)

    for step in range(COMP_LEN):
        logits = policy(seqs)[:, -1, :] / temperature
        probs = torch.softmax(logits, -1)
        draw = torch.multinomial(probs, 1, generator=generator).squeeze(1)

        draw = torch.where(finished, torch.full_like(draw, PAD), draw)

        mask[:, step] = (~finished).float()

        finished = finished | (draw == EOS)

        seqs = torch.cat([seqs, draw.unsqueeze(1)], dim=1)

    return seqs, mask

# ── Step 009  completion_logprobs ──
import torch


def completion_logprobs(policy, seqs):
    logits = policy(seqs[:, :-1])
    logp = torch.log_softmax(logits, dim=-1)
    target = seqs[:, PROMPT_LEN:]
    sliced = logp[:, PROMPT_LEN - 1:, :]
    gathered = sliced.gather(2, target.unsqueeze(-1)).squeeze(-1)
    return gathered

# ── Step 010  make_demos ──
import random


def make_demos(rng, tasks):
    demos = []
    for nums, target in tasks:
        sols = solve(nums, target)
        chosen = rng.choice(sols)
        prompt = encode_task(nums, target)
        completion = [TOK[t] for t in chosen] + [EOS]
        demos.append((prompt, completion))

    return demos

# ── Step 011  sft_loss ──
import torch


def sft_loss(policy, batch):
    rows = [p + c for p, c in batch]
    seqs = torch.tensor(rows, dtype=torch.long)
    logp = completion_logprobs(policy, seqs)
    return -logp.mean()

# ── Step 012  pretrain ──
import random
import torch


def pretrain(policy, demos, steps, batch_size=64, lr=1e-3, seed=0):
    rng = random.Random(seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    n = min(batch_size, len(demos))

    loss = 0.0
    for _ in range(steps):
        batch = rng.sample(demos, n)
        loss = sft_loss(policy, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return float(loss)

# ── Step 013  pass_at_k ──
import torch


def pass_at_k(policy, tasks, k, generator, strict=True):
    B = len(tasks)
    prompts = torch.tensor([encode_task(nums, target) for nums, target in tasks], dtype=torch.long)
    seqs, _ = sample_completions(policy, prompts, k, generator)

    comp = seqs[:, PROMPT_LEN:]
    correct = [False] * (B * k)
    for r in range(B * k):
        tokens = decode_completion(comp[r])
        correct[r] = verify(tasks[r // k][0], tasks[r // k][1], tokens, strict)[0]

    pass1 = sum(correct) / (B * k)
    passk = 0
    for i in range(B):
        if any(correct[i * k + g] for g in range(k)):
            passk += 1
            
    passk /= B
    return (float(pass1), float(passk))

# ── Step 014  group_rewards ──
import torch


def group_rewards(tasks, seqs, G, strict=True):
    B = len(tasks)
    rewards = torch.zeros(B, G, dtype=torch.float32)
    reasons = []
    for i in range(B):
        nums, target = tasks[i]
        for g in range(G):
            row = i * G + g
            tokens = decode_completion(seqs[row, PROMPT_LEN:])
            correct, reason = verify(nums, target, tokens, strict)
            rewards[i, g] = 1.0 if correct else 0.0
            reasons.append(reason)
            
    return rewards, reasons

# ── Step 015  group_advantages ──
import torch


def group_advantages(rewards, normalize_std=True, eps=1e-4):
    centred = rewards - rewards.mean(dim=1, keepdim=True)
    if normalize_std:
        centred = centred / (rewards.std(dim=1, keepdim=True) + eps)

    return centred.flatten()

# ── Step 016  kl_k3 ──
import torch


def kl_k3(logp, ref_logp):
    d = ref_logp - logp
    return torch.exp(d) - d - 1.0

# ── Step 017  grpo_loss ──
import torch


def grpo_loss(logp, old_logp, ref_logp, advantages, mask, clip_eps=0.2, beta=0.02):
    A = advantages.unsqueeze(1)

    ratio = torch.exp(logp - old_logp)
    unclipped = ratio * A
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * A
    pg = -torch.min(unclipped, clipped)

    kl = kl_k3(logp, ref_logp)

    per_token = pg + beta * kl
    denom = mask.sum()
    loss = (mask * per_token).sum() / denom

    with torch.no_grad():
        pg_mean = float((mask * pg).sum() / denom)
        kl_mean = float((mask * kl).sum() / denom)
        outside = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).float()
        clip_frac = float((mask * outside).sum() / denom)

    stats = {"pg": pg_mean, "kl": kl_mean, "clip_frac": clip_frac}
    return loss, stats

# ── Step 018  rollout ──
import torch


def rollout(policy, ref, tasks, G, generator, strict=True, normalize_std=True):
    prompts = torch.tensor([encode_task(nums, target) for nums, target in tasks], dtype=torch.long)
    seqs, mask = sample_completions(policy, prompts, G, generator)
    rewards, reasons = group_rewards(tasks, seqs, G, strict)
    advantages = group_advantages(rewards, normalize_std)

    with torch.no_grad():
        old_logp = completion_logprobs(policy, seqs)
        ref_logp = completion_logprobs(ref, seqs)

    return {
        "seqs": seqs,
        "mask": mask,
        "rewards": rewards,
        "advantages": advantages,
        "old_logp": old_logp,
        "ref_logp": ref_logp,
        "reasons": reasons,
    }

# ── Step 019  train_grpo ──
import random
import torch
import torch.nn as nn


def train_grpo(policy, ref, tasks, steps, cfg, seed=0, strict=True):
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg["lr"])

    history = []
    for step in range(steps):
        batch = rng.sample(tasks, cfg["batch"])
        out = rollout(policy, ref, batch, cfg["group"], generator, strict, cfg["normalize_std"])

        reward_mean = float(out["rewards"].mean())
        stats = None
        loss = None
        for _ in range(cfg["epochs"]):
            logp = completion_logprobs(policy, out["seqs"])
            loss, stats = grpo_loss(
                logp,
                out["old_logp"],
                out["ref_logp"],
                out["advantages"],
                out["mask"],
                cfg["clip_eps"],
                cfg["beta"],
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        history.append({
            "step": step,
            "reward": reward_mean,
            "loss": float(loss),
            "pg": float(stats["pg"]),
            "kl": float(stats["kl"]),
            "clip_frac": float(stats["clip_frac"]),
        })

    return history

# ── Step 020  completion_entropy ──
import torch


@torch.no_grad()
def completion_entropy(policy, seqs, mask):
    logits = policy(seqs[:, :-1])
    sliced = logits[:, PROMPT_LEN - 1:, :]
    logp = torch.log_softmax(sliced, dim=-1)
    p = logp.exp()
    H = -(p * logp).sum(dim=-1)
    return float((mask * H).sum() / mask.sum())

# ── Step 021  evaluate ──
import torch
from collections import Counter


@torch.no_grad()
def evaluate(policy, tasks, k, generator):
    B = len(tasks)
    prompts = torch.tensor([encode_task(nums, target) for nums, target in tasks], dtype=torch.long)
    seqs, mask = sample_completions(policy, prompts, k, generator)

    strict_rewards, strict_reasons = group_rewards(tasks, seqs, k, strict=True)
    lenient_rewards, _ = group_rewards(tasks, seqs, k, strict=False)

    total = B * k
    strict_flat = strict_rewards.flatten()
    lenient_flat = lenient_rewards.flatten()

    pass1 = float(strict_flat.mean())
    passk = 0
    for i in range(B):
        if strict_flat[i * k:(i + 1) * k].any():
            passk += 1
    passk = passk / B

    lenient_pass1 = float(lenient_flat.mean())

    accepted_lenient = lenient_flat.bool()
    rejected_strict = ~strict_flat.bool()
    gaming = float((accepted_lenient & rejected_strict).float().mean())

    entropy = completion_entropy(policy, seqs, mask)

    by_digits = {}
    for i, (nums, target) in enumerate(tasks):
        nd = len(str(target))
        by_digits.setdefault(nd, []).append(strict_flat[i * k:(i + 1) * k])
    by_digits = {nd: float(torch.cat(v).mean()) for nd, v in by_digits.items()}

    counts = Counter(strict_reasons)
    reasons = {r: round(c / total, 3) for r, c in counts.items()}

    example = decode_completion(seqs[0, PROMPT_LEN:])

    return {
        "pass1": pass1,
        "passk": passk,
        "lenient_pass1": lenient_pass1,
        "gaming": gaming,
        "entropy": entropy,
        "by_digits": by_digits,
        "reasons": reasons,
        "example": example,
    }

# ── Step 022  build_base ──
import random
import torch


def build_base(seed=0, sft_steps=600, n_draws=4000, n_held=200):
    rng = random.Random(seed)
    train, held = task_split(rng, n_draws, n_held)

    demos = make_demos(rng, train)

    torch.manual_seed(seed)
    policy = Policy()

    sft_loss = pretrain(policy, demos, sft_steps, seed=seed)

    return {
        "policy": policy,
        "train": train,
        "held": held,
        "sft_loss": sft_loss,
    }

# ── Step 023  run_grpo ──
import copy


def run_grpo(base, steps, cfg, seed=0, strict=True):
    policy = copy.deepcopy(base["policy"])
    ref = copy.deepcopy(base["policy"])
    history = train_grpo(policy, ref, base["train"], steps, cfg, seed, strict)
    return {
        "policy": policy,
        "ref": ref,
        "history": history,
    }

# ── Step 024  rlvr_experiment ──
import torch


def rlvr_experiment(seed=0, sft_steps=600, rl_steps=300, hack_steps=50, cfg=None, k=8, window=50):
    if cfg is None:
        cfg = dict(batch=16, group=8, lr=3e-4, beta=0.02, clip_eps=0.2, normalize_std=True, epochs=1)

    base = build_base(seed, sft_steps)
    base_eval = evaluate(base["policy"], base["held"], k, torch.Generator().manual_seed(seed))

    strict = run_grpo(base, rl_steps, cfg, seed, strict=True)
    strict_eval = evaluate(strict["policy"], base["held"], k, torch.Generator().manual_seed(seed))

    hack = run_grpo(base, hack_steps, cfg, seed, strict=False)
    hack_eval = evaluate(hack["policy"], base["held"], k, torch.Generator().manual_seed(seed))

    history = strict["history"]
    curve = []
    for start in range(0, len(history), window):
        chunk = history[start:start + window]
        end = chunk[-1]["step"] + 1
        reward_mean = sum(h["reward"] for h in chunk) / len(chunk)
        kl_mean = sum(h["kl"] for h in chunk) / len(chunk)
        curve.append({"step": end, "reward": reward_mean, "kl": kl_mean})

    return {
        "base": {"sft_loss": base["sft_loss"], "eval": base_eval},
        "strict": {"history": history, "eval": strict_eval, "curve": curve},
        "hack": {"history": hack["history"], "eval": hack_eval},
        "cfg": cfg,
        "n_train": len(base["train"]),
        "n_held": len(base["held"]),
    }

# ── Scaffold (runner) ──
"""RL fine-tuning with verifiable rewards, from scratch.

Story: a Countdown-style environment you built (three numbers, a target, a postfix
expression that must use each number once) gives an exact reward. A tiny causal
transformer is warm-started on solver demonstrations, then fine-tuned with GRPO:
groups of samples per task, group-relative advantages, a clipped policy-gradient
objective and a k3 KL penalty to the frozen reference. On held-out tasks the
policy's pass@1 roughly doubles while pass@8 barely moves and the entropy drops:
RL sharpened what the base already knew. Against a verifier that fails open, the
same loop learns to write unparseable expressions in a few dozen steps.
"""
import random
import torch


def by_size(ev):
    return {f"{d}-digit targets": round(v, 3) for d, v in sorted(ev["by_digits"].items())}


def main() -> None:
    torch.manual_seed(0)
    rep = rlvr_experiment(seed=0, sft_steps=600, rl_steps=300, hack_steps=50)
    cfg = rep["cfg"]

    print("1. The environment")
    rng = random.Random(0)
    nums, target = make_task(rng)
    sols = solve(nums, target)
    print(f"   task: numbers {nums}, target {target} | {len(sols)} valid expressions, e.g. {' '.join(sols[0])}")
    print(f"   verify: {verify(nums, target, sols[0])} | wrong value: {verify(nums, target, ['9', '9', '*', '9', '*'])} | too short: {verify(nums, target, ['9', '9', '*'])}")
    print(f"   fail-open verifier on the same short answer: {verify(nums, target, ['9', '9', '*'], strict=False)}")
    print(f"   task bank: {rep['n_train']} training tasks, {rep['n_held']} held-out tasks, no overlap")

    b = rep["base"]["eval"]
    print("\n2. The base policy (600 SFT steps on solver demonstrations)")
    print(f"   final SFT loss {rep['base']['sft_loss']:.3f} | held-out pass@1 {b['pass1']:.3f}, pass@8 {b['passk']:.3f} | entropy {b['entropy']:.2f} nats")
    print(f"   failure reasons: {b['reasons']}")
    print(f"   by target size: {by_size(b)}")
    print("   The base knows the format and often has a correct answer among eight tries, but rarely on the first.")

    s = rep["strict"]
    print(f"\n3. GRPO against the strict verifier ({len(s['history'])} steps, batch {cfg['batch']} x group {cfg['group']}, lr {cfg['lr']}, beta {cfg['beta']})")
    for c in s["curve"]:
        print(f"   steps {c['step']:>4}: mean batch reward {c['reward']:.3f} | KL to reference {c['kl']:.3f}")
    e = s["eval"]
    print(f"   held-out pass@1 {b['pass1']:.3f} -> {e['pass1']:.3f} | pass@8 {b['passk']:.3f} -> {e['passk']:.3f} | entropy {b['entropy']:.2f} -> {e['entropy']:.2f}")
    print(f"   by target size: {by_size(e)}")
    print(f"   failure reasons: {e['reasons']}")
    print("   The gain is on tasks the policy never trained on, so it is the skill that improved, not memory.")

    print("\n4. What RL did")
    print(f"   pass@1 moved toward the base pass@8 while pass@8 itself moved little and the entropy fell:")
    print("   GRPO sharpened the base policy's distribution onto answers it could already produce.")
    print("   Without longer reasoning or new data, verifiable-reward RL at this scale is distribution sharpening,")
    print("   and the KL to the reference is the price paid for it.")

    h = rep["hack"]["eval"]
    last = rep["hack"]["history"][-1]
    print(f"\n5. Specification gaming: the same loop against the fail-open verifier ({len(rep['hack']['history'])} steps)")
    print(f"   reported reward at the last step {last['reward']:.3f} | KL to reference {last['kl']:.3f}")
    print(f"   held-out: lenient pass@1 {h['lenient_pass1']:.3f} | strict pass@1 {h['pass1']:.3f} | gaming rate {h['gaming']:.3f}")
    print(f"   failure reasons under the strict verifier: {h['reasons']} | example completion: {h['example']}")
    print("   Every bit of that reward was earned by writing expressions the grader could not parse.")
    print("   The policy gradient optimises the reward you implemented, not the one you meant.")


if __name__ == "__main__":
    main()
