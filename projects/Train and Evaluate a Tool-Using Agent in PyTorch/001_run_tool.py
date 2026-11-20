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