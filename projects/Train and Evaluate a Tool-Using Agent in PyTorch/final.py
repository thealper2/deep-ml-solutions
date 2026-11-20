"""
Train and Evaluate a Tool-Using Agent in PyTorch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  run_tool ──
import random

# ---------------------------------------------------------------- provided: do not edit
ITEMS = ["apple", "pear", "fig", "plum", "kiwi", "lime", "mango", "peach", "grape", "melon",
         "lemon", "cherry", "olive", "date", "guava", "papaya", "quince", "banana", "orange", "apricot"]
# The warehouse the lookup tool reads: item -> units in stock. Every value is different.
STORE = dict(zip(ITEMS, [17, 5, 42, 8, 31, 26, 3, 49, 12, 38, 21, 9, 44, 15, 0, 36, 28, 7, 33, 23]))
# The same warehouse after a delivery: every stock has changed.
RESTOCKED = dict(zip(ITEMS, [30, 11, 2, 47, 19, 6, 40, 13, 35, 1, 27, 45, 16, 4, 22, 10, 48, 25, 14, 39]))

SPECIAL = ["<pad>", "<unk>", "<q>", "</q>", "<call>", "</call>", "<result>", "</result>", "<answer>", "</answer>"]
VOCAB = (SPECIAL + ["what", "is", "stock", "of", "plus", "+", "-", "*", "calc", "lookup", "error"]
         + ITEMS + [str(n) for n in range(100)])
STOI = {token: i for i, token in enumerate(VOCAB)}


def solve(question, store):
    """The correct answer (an int) to a question, computed directly from the store."""
    q = question.split()
    if q[0] == "what":
        a, b = int(q[2]), int(q[4])
        return a + b if q[3] == "+" else a - b
    if len(q) == 7:
        return store[q[2]] + store[q[6]]
    return store[q[2]] + int(q[4])


def random_store(seed):
    """A warehouse with the same items and random stocks (all different, 0 to 49)."""
    values = list(range(50))
    random.Random(seed).shuffle(values)
    return dict(zip(ITEMS, values))


def make_tasks(n, seed):
    """n different tasks, one third of each kind: 'calc', 'add' and 'total'."""
    rng = random.Random(seed)
    tasks, seen = [], set()
    while len(tasks) < n:
        kind = ("calc", "add", "total")[len(tasks) % 3]
        if kind == "calc":
            a, b, op = rng.randrange(50), rng.randrange(50), rng.choice("+-")
            if op == "-" and b > a:
                a, b = b, a
            question = f"what is {a} {op} {b}"
        elif kind == "add":
            question = f"stock of {rng.choice(ITEMS)} plus {rng.randrange(50)}"
        else:
            x = rng.choice(ITEMS)
            y = rng.choice([i for i in ITEMS if i != x])
            question = f"stock of {x} plus stock of {y}"
        if question not in seen:
            seen.add(question)
            tasks.append({"id": f"s{seed}-{len(tasks):03d}", "kind": kind, "question": question})
    return tasks


def teacher_policy(transcript):
    """A scripted expert. Reads the transcript so far and returns the next correct action."""
    toks = transcript.split()
    q = toks[1:toks.index("</q>")]
    results = [toks[i + 1] for i, t in enumerate(toks) if t == "<result>"]
    if q[0] == "what":
        plan = [["calc", q[2], q[3], q[4]]]
    elif len(q) == 7:
        plan = [["lookup", q[2]], ["lookup", q[6]], ["calc", "$0", "+", "$1"]]
    else:
        plan = [["lookup", q[2]], ["calc", "$0", "+", q[4]]]
    if len(results) < len(plan):
        step = [results[int(a[1:])] if a.startswith("$") else a for a in plan[len(results)]]
        return "<call> " + " ".join(step) + " </call>"
    return "<answer> " + results[-1] + " </answer>"


def make_teacher(error_rate, seed):
    """The expert with mistakes: each action is replaced by a wrong one with probability error_rate."""
    rng = random.Random(seed)

    def policy(transcript):
        toks = teacher_policy(transcript).split()
        if rng.random() < error_rate:
            if toks[0] == "<answer>":
                toks[1] = str((int(toks[1]) + 1 + rng.randrange(5)) % 100) if toks[1].isdigit() else "0"
            elif toks[1] == "lookup":
                toks[2] = rng.choice([i for i in ITEMS if i != toks[2]])
            elif rng.random() < 0.5 and toks[2].isdigit():
                toks[2] = str(int(toks[2]) + 1)
            else:
                toks = toks[:-1]
        return " ".join(toks)

    return policy
# ---------------------------------------------------------------- end of provided code


def run_tool(name, args, store):
    """Run the calc or lookup tool; return its output as a string, or 'error'."""
    try:
        if name == "calc":
            if len(args) != 3:
                return "error"
            a, op, b = args
            if not (isinstance(a, str) and isinstance(b, str)):
                return "error"
            if not a.isdigit() or not b.isdigit():
                return "error"
            x, y = int(a), int(b)
            if op == "+":
                return str(x + y)
            elif op == "-":
                return str(x - y)
            elif op == "*":
                return str(x * y)
            return "error"
        elif name == "lookup":
            if len(args) != 1:
                return "error"
            key = args[0]
            if not isinstance(key, str) or key not in store:
                return "error"
            return str(store[key])
        return "error"
    except Exception:
        return "error"

# ── Step 002  parse_action ──
def parse_action(text):
    """Return the first action in text as a dict of type 'call', 'answer' or 'invalid'."""
    tokens = text.split()
    if not tokens:
        return {"type": "invalid"}

    first = tokens[0]
    if first == "<call>":
        end = None
        for i in range(1, len(tokens)):
            if tokens[i] == "</call>":
                end = i
                break
        if end is None:
            return {"type": "invalid"}
        inner = tokens[1:end]
        for t in inner:
            if t.startswith("<"):
                return {"type": "invalid"}
        if not inner:
            return {"type": "invalid"}
        return {"type": "call", "name": inner[0], "args": inner[1:]}

    if first == "<answer>":
        end = None
        for i in range(1, len(tokens)):
            if tokens[i] == "</answer>":
                end = i
                break
        if end is None:
            return {"type": "invalid"}
        inner = tokens[1:end]
        for t in inner:
            if t.startswith("<"):
                return {"type": "invalid"}
        if len(inner) != 1:
            return {"type": "invalid"}
        return {"type": "answer", "value": inner[0]}

    return {"type": "invalid"}

# ── Step 003  run_episode ──
def run_episode(policy, task, store, max_calls=4):
    """Run one task: let the policy call tools until it answers, then grade the answer."""
    transcript = "<q> " + task["question"] + " </q>"
    calls = []

    while True:
        raw = policy(transcript)
        action = parse_action(raw)

        if action["type"] == "invalid":
            return {
                "status": "invalid",
                "answer": None,
                "correct": False,
                "calls": calls,
                "transcript": transcript,
            }

        if action["type"] == "answer":
            value = action["value"]
            transcript += " <answer> " + value + " </answer>"
            correct = value == str(solve(task["question"], store))
            return {
                "status": "answered",
                "answer": value,
                "correct": correct,
                "calls": calls,
                "transcript": transcript,
            }

        # action["type"] == "call"
        if len(calls) >= max_calls:
            return {
                "status": "budget",
                "answer": None,
                "correct": False,
                "calls": calls,
                "transcript": transcript,
            }

        name = action["name"]
        args = action["args"]
        result = run_tool(name, args, store)
        calls.append((name, args, result))
        call_text = " ".join([name] + args)
        transcript += " <call> " + call_text + " </call> <result> " + result + " </result>"

# ── Step 004  collect_trajectories ──
def collect_trajectories(tasks, stores, teacher):
    """Run the teacher on every task in its store; return (kept records, number dropped)."""
    kept = []
    dropped = 0
    for task, store in zip(tasks, stores):
        ep = run_episode(teacher, task, store)
        if ep["correct"]:
            kept.append({
                "id": task["id"],
                "question": task["question"],
                "text": ep["transcript"],
            })
        else:
            dropped += 1
    return kept, dropped

# ── Step 005  encode_trajectory ──
def encode_trajectory(text, block_size=40):
    """Token ids and a 0/1 loss mask (1 = written by the model), both padded to block_size."""
    tokens = text.split()
    if len(tokens) > block_size:
        raise ValueError("trajectory too long")

    mask = []
    spans = [("<call>", "</call>"), ("<answer>", "</answer>")]
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        opened = False
        for start, end in spans:
            if tok == start:
                mask.append(1)
                j = i + 1
                while j < len(tokens) and tokens[j] != end:
                    mask.append(1)
                    j += 1
                if j < len(tokens):
                    mask.append(1)
                    i = j + 1
                else:
                    i = j
                opened = True
                break
        if opened:
            continue
        mask.append(0)
        i += 1

    ids = [STOI.get(t, STOI["<unk>"]) for t in tokens]

    pad_id = STOI["<pad>"]
    while len(ids) < block_size:
        ids.append(pad_id)
        mask.append(0)

    return ids, mask

# ── Step 006  ToolLM ──
import torch
import torch.nn as nn


class ToolLM(nn.Module):
    """A small causal transformer: token ids (B, T) in, next-token logits (B, T, vocab_size) out."""

    def __init__(self, vocab_size, block_size=40, d_model=48, n_head=4, n_layer=1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, n_head, 4 * d_model,
                dropout=0.0, batch_first=True, norm_first=True,
            )
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), diagonal=1)
        for block in self.blocks:
            x = block(x, src_mask=causal)
        x = self.ln_f(x)
        return self.head(x)

# ── Step 007  sft_loss ──
import torch
import torch.nn.functional as F


def sft_loss(logits, ids, mask):
    """Mean next-token cross-entropy over the targets whose mask is 1."""
    shift_logits = logits[:, :-1, :]
    shift_targets = ids[:, 1:]
    shift_mask = mask[:, 1:].to(torch.float32)

    B, T, V = shift_logits.shape
    ce = F.cross_entropy(
        shift_logits.reshape(B * T, V),
        shift_targets.reshape(B * T),
        reduction="none",
    ).reshape(B, T)

    masked = ce * shift_mask
    return masked.sum() / shift_mask.sum()

# ── Step 008  train_sft ──
import torch


def train_sft(model, ids, mask, steps, batch_size=32, lr=3e-3, generator=None):
    """AdamW on random batches of trajectories; returns each step's loss as a float."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    N = ids.shape[0]
    losses = []
    for _ in range(steps):
        rows = torch.randint(0, N, (batch_size,), generator=generator)
        logits = model(ids[rows])
        loss = sft_loss(logits, ids[rows], mask[rows])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses

# ── Step 009  model_policy ──
import torch


def model_policy(model, temperature=1.0, generator=None, max_new_tokens=8):
    """Return policy(transcript) -> the model's next action as text, generated token by token."""
    def policy(transcript):
        tokens = transcript.split()
        ids = [STOI.get(t, STOI["<unk>"]) for t in tokens]
        generated = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                ctx = ids[-model.block_size:]
                x = torch.tensor([ctx], dtype=torch.long)
                logits = model(x)
                last = logits[0, -1, :]

                if temperature == 0:
                    nxt = int(torch.argmax(last).item())
                else:
                    probs = F.softmax(last / temperature, dim=-1)
                    nxt = int(torch.multinomial(probs, 1, generator=generator).item())

                tok = VOCAB[nxt]
                ids.append(nxt)
                generated.append(tok)
                if tok in ("</call>", "</answer>"):
                    break

        return " ".join(generated)
    return policy

# ── Step 010  run_trials ──
def run_trials(policy, tasks, store, n_trials, max_calls=4):
    """outcomes[i][j] = 1 if task i was solved in trial j; trials run one after another."""
    outcomes = [[0] * n_trials for _ in range(len(tasks))]
    for j in range(n_trials):
        for i, task in enumerate(tasks):
            ep = run_episode(policy, task, store, max_calls)
            outcomes[i][j] = 1 if ep["correct"] else 0

    return outcomes

# ── Step 011  pass_k ──
from math import comb


def pass_k(outcomes, k):
    """{'pass_at_k': at least one of k trials succeeds, 'pass_hat_k': all k succeed}, averaged over tasks."""
    if not outcomes:
        raise ValueError("outcomes is empty")
    n = len(outcomes[0])
    for row in outcomes:
        if len(row) != n:
            raise ValueError("rows differ in length")
    if not (1 <= k <= n):
        raise ValueError("k out of range")

    denom = comb(n, k)
    at_sum = 0.0
    hat_sum = 0.0
    for row in outcomes:
        c = sum(1 for x in row if x == 1)
        at = 1 - (comb(n - c, k) / denom if n - c >= k else 0)
        hat = (comb(c, k) / denom if c >= k else 0)
        at_sum += at
        hat_sum += hat

    m = len(outcomes)
    return {
        "pass_at_k": round(at_sum / m, 4),
        "pass_hat_k": round(hat_sum / m, 4),
    }

# ── Step 012  score_with_se ──
import math
import numpy as np


def score_with_se(outcomes):
    """{'score', 'se' (across tasks), 'seed_scores' (pass@1 per trial), 'seed_std'} for an outcome matrix."""
    arr = np.array(outcomes, dtype=float)
    N, n = arr.shape

    score = float(arr.mean())
    seed_scores = [float(arr[:, j].mean()) for j in range(n)]

    if n >= 2:
        seed_std = float(np.std(seed_scores, ddof=1))
    else:
        seed_std = None

    if N >= 2:
        row_means = arr.mean(axis=1)
        s = float(np.std(row_means, ddof=1))
        se = float(s / np.sqrt(N))
    else:
        se = None

    return {
        "score": round(score, 4),
        "se": round(float(se), 4) if se is not None else None,
        "seed_scores": [round(x, 4) for x in seed_scores],
        "seed_std": round(float(seed_std), 4) if seed_std is not None else None,
    }

# ── Step 013  compare_checkpoints ──
import math


def compare_checkpoints(old, new):
    """{'gap', 'se_paired', 'se_unpaired', 'verdict'} for two outcome matrices on the same tasks."""
    if len(old) != len(new):
        raise ValueError("different numbers of tasks")
    N = len(old)
    if N < 2:
        raise ValueError("need at least two tasks")

    old_means = np.array([np.mean(row) for row in old], dtype=float)
    new_means = np.array([np.mean(row) for row in new], dtype=float)
    d = new_means - old_means

    gap = float(d.mean())
    sd = float(np.std(d, ddof=1))
    se_paired = float(sd / np.sqrt(N))

    se_old = score_with_se(old)["se"]
    se_new = score_with_se(new)["se"]
    se_unpaired = float(np.sqrt(se_old ** 2 + se_new ** 2))

    if gap > 2 * se_paired:
        verdict = "new is better"
    elif gap < -2 * se_paired:
        verdict = "new is worse"
    else:
        verdict = "cannot tell"

    return {
        "gap": round(gap, 4),
        "se_paired": round(float(se_paired), 4),
        "se_unpaired": round(se_unpaired, 4),
        "verdict": verdict,
    }

# ── Step 014  contamination_report ──
import math
import re


def _normalize(q):
    return re.sub(r"\s+", " ", q.lower()).strip()


def contamination_report(tasks, outcomes, train_questions):
    """Score the tasks whose question is in the training data and the held-out tasks separately."""
    train_norm = set(_normalize(q) for q in train_questions)

    seen_rows = []
    held_rows = []
    for i, task in enumerate(tasks):
        if _normalize(task["question"]) in train_norm:
            seen_rows.append(outcomes[i])
        else:
            held_rows.append(outcomes[i])

    def side(rows):
        if not rows:
            return {"n": 0, "score": None, "se": None}
        r = score_with_se(rows)
        return {"n": len(rows), "score": r["score"], "se": r["se"]}

    seen = side(seen_rows)
    held = side(held_rows)

    if seen["score"] is not None and held["score"] is not None:
        gap = round(seen["score"] - held["score"], 4)
    else:
        gap = None

    if seen["se"] is not None and held["se"] is not None:
        se_gap = round((seen["se"] ** 2 + held["se"] ** 2) ** 0.5, 4)
    else:
        se_gap = None

    return {
        "seen": seen,
        "held_out": held,
        "gap": gap,
        "se_gap": se_gap,
    }

# ── Scaffold (runner) ──
"""Train and evaluate a tool-using agent.

Story: a teacher writes tool trajectories, a checker filters them, and a small transformer is
fine-tuned on what is left with the tool outputs masked out of the loss. The model then runs as the
agent inside your loop, with sampling, and is scored the way an agent benchmark should be scored:
repeated trials, pass@k next to pass^k, a standard error on every number, a paired comparison of
checkpoints, a split by contamination, and a rerun in a changed environment.

Two agents are trained on the same questions. The "memoriser" saw every trajectory in the same
warehouse. The "tool user" saw a different random warehouse for every trajectory.
"""
import copy
import torch


def pick(pool, per_kind):
    """The first per_kind tasks of each kind."""
    return [t for kind in ("calc", "add", "total") for t in [x for x in pool if x["kind"] == kind][:per_kind]]


def to_tensors(trajectories):
    encoded = [encode_trajectory(t["text"]) for t in trajectories]
    return torch.tensor([e[0] for e in encoded]), torch.tensor([e[1] for e in encoded])


def train_agent(ids, mask, save_at):
    """Supervised fine-tuning from scratch; returns a copy of the model at each step in save_at."""
    torch.manual_seed(0)
    model = ToolLM(len(VOCAB), block_size=40, d_model=48, n_head=4, n_layer=1)
    generator = torch.Generator().manual_seed(0)
    saved, losses, done = {}, [], 0
    for step in save_at:
        losses += train_sft(model, ids, mask, step - done, batch_size=16, lr=3e-3, generator=generator)
        saved[step] = copy.deepcopy(model)
        done = step
    return saved, losses


def evaluate(model, tasks, store, n_trials, max_calls=4):
    """Run the model as the agent at temperature 1.0; one trial is one pass over the benchmark."""
    policy = model_policy(model, temperature=1.0, generator=torch.Generator().manual_seed(1))
    return run_trials(policy, tasks, store, n_trials, max_calls)


def line(name, outcomes):
    s = score_with_se(outcomes)
    return f"{name:<22} score {s['score']:.3f} +- {s['se']:.3f}   one run moves by {s['seed_std']:.3f} between seeds"


def main():
    torch.set_num_threads(1)  # the model is tiny: one thread is faster than many

    # ---- 1. Trajectories from a teacher that is wrong 10% of the time, filtered by the checker ----
    train_tasks = make_tasks(300, seed=0)
    same_store = [STORE] * len(train_tasks)
    random_stores = [random_store(i) for i in range(len(train_tasks))]
    fixed, dropped = collect_trajectories(train_tasks, same_store, make_teacher(0.1, seed=0))
    varied, _ = collect_trajectories(train_tasks, random_stores, make_teacher(0.1, seed=0))
    print(f"teacher trajectories: {len(train_tasks)} tasks, {len(fixed)} kept, {dropped} dropped by the checker")
    print("example:", varied[2]["text"])

    # ---- 2. Supervised fine-tuning with tool outputs masked out of the loss ----
    ids, mask = to_tensors(varied)
    tool_user, losses = train_agent(ids, mask, save_at=(250, 600, 700))
    memoriser, _ = train_agent(*to_tensors(fixed), save_at=(700,))
    result_mask = torch.zeros_like(mask)
    result_mask[:, 1:] = (ids[:, :-1] == STOI["<result>"]).long()
    with torch.no_grad():
        logits = tool_user[700](ids)
        own, tool = sft_loss(logits, ids, mask).item(), sft_loss(logits, ids, result_mask).item()
    print(f"\nloss on the model's own tokens: {sum(losses[:20]) / 20:.2f} at the start, {own:.3f} after 700 steps")
    print(f"loss on tool outputs, which were never a training target: {tool:.2f}")

    # ---- 3. A benchmark: 30 tasks that are also in the training data, 30 that are not ----
    train_questions = [t["question"] for t in varied]
    kept_tasks = [t for t in train_tasks if t["question"] in set(train_questions)]
    fresh = [t for t in make_tasks(400, seed=1) if t["question"] not in set(train_questions)]
    bench = pick(kept_tasks, 10) + pick(fresh, 10)
    runs = {step: evaluate(model, bench, STORE, n_trials=8) for step, model in tool_user.items()}
    final = runs[700]
    print(f"\nbenchmark: {len(bench)} tasks, 8 trials each, temperature 1.0")
    print("tool user, pass@1 of each trial:", score_with_se(final)["seed_scores"])
    for step in runs:
        print(line(f"tool user, step {step}", runs[step]))

    # ---- 4. pass@k (at least one of k trials) against pass^k (all k trials) ----
    for name, outcomes in (("step 250", runs[250]), ("step 700", final)):
        print(f"\n{name}:  k   pass@k   pass^k")
        for k in (1, 2, 4, 8):
            m = pass_k(outcomes, k)
            print(f"           {k}   {m['pass_at_k']:.3f}    {m['pass_hat_k']:.3f}")

    # ---- 5. Is a gap between two checkpoints real? ----
    print()
    for old, new in ((250, 700), (600, 700)):
        c = compare_checkpoints(runs[old], runs[new])
        print(f"step {old} -> {new}: gap {c['gap']:+.3f}, paired SE {c['se_paired']:.3f} "
              f"(unpaired {c['se_unpaired']:.3f}): {c['verdict']}")

    # ---- 6. Contamination: tasks seen in training against held-out tasks ----
    memo_run = evaluate(memoriser[700], bench, STORE, n_trials=8)
    print()
    for name, outcomes in (("tool user", final), ("memoriser", memo_run)):
        r = contamination_report(bench, outcomes, train_questions)
        print(f"{name}: seen {r['seen']['score']:.3f} (n={r['seen']['n']}), held out {r['held_out']['score']:.3f} "
              f"(n={r['held_out']['n']}), gap {r['gap']:+.3f} +- {r['se_gap']:.3f}")

    # ---- 7. A perturbed benchmark: same questions, the warehouse after a delivery ----
    print()
    for name, model, before in (("tool user", tool_user[700], final), ("memoriser", memoriser[700], memo_run)):
        after = evaluate(model, bench, RESTOCKED, n_trials=4)
        print(f"{name}: {score_with_se(before)['score']:.3f} in the training warehouse, "
              f"{score_with_se(after)['score']:.3f} after the stocks change")

    # ---- 8. The harness is part of the score ----
    tight = evaluate(tool_user[700], bench, STORE, n_trials=4, max_calls=2)
    print(f"\ntool user with a budget of 2 tool calls instead of 4: {score_with_se(tight)['score']:.3f}")

    episode = run_episode(model_policy(tool_user[700], 1.0, torch.Generator().manual_seed(5)), bench[-1], STORE)
    print("\none episode:", episode["transcript"])
    print("status:", episode["status"], "| correct:", episode["correct"])


if __name__ == "__main__":
    main()
