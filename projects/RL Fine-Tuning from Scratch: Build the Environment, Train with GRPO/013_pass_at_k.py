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