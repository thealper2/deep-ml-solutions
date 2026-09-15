import torch
import torch.nn.functional as F

def adaptive_loop_exit(step_fn, s0, coda, max_loops, kl_threshold):
    s = torch.as_tensor(s0, dtype=torch.float32)

    prev_logp = None
    kl_history = []
    loops_used = 0

    for i in range(1, max_loops + 1):
        s = step_fn(s)
        loops_used = i

        logits = coda(s)
        logp = F.log_softmax(logits, dim=-1)

        if prev_logp is not None:
            p_prev = prev_logp.exp()
            kl_per_pos = (p_prev * (prev_logp  - logp)).sum(dim=-1)
            kl = float(kl_per_pos.mean().item())
            kl_history.append(kl)
            if kl < kl_threshold:
                break

        prev_logp = logp

    return s, loops_used, kl_history
