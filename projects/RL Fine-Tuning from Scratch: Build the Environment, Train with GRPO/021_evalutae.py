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