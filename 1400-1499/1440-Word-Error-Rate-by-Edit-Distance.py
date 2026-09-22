import re


def normalize_text(s):
    s = s.lower()
    s = re.sub(r'[.,!?;:"\']', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def edit_ops(ref_words, hyp_words):
    n = len(ref_words)
    m = len(hyp_words)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],
                    dp[i - 1][j],
                    dp[i][j - 1],
                )

    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            break

    return subs, dels, ins


def wer(ref, hyp):
    ref_norm = normalize_text(ref)
    hyp_norm = normalize_text(hyp)
    ref_words = ref_norm.split() if ref_norm else []
    hyp_words = hyp_norm.split() if hyp_norm else []

    subs, dels, ins = edit_ops(ref_words, hyp_words)
    N = len(ref_words)
    return round((subs + dels + ins) / max(1, N), 4)


def corpus_wer(refs, hyps):
    total_errors = 0
    total_ref = 0
    for ref, hyp in zip(refs, hyps):
        ref_norm = normalize_text(ref)
        hyp_norm = normalize_text(hyp)
        ref_words = ref_norm.split() if ref_norm else []
        hyp_words = hyp_norm.split() if hyp_norm else []

        subs, dels, ins = edit_ops(ref_words, hyp_words)
        total_errors += subs + dels + ins
        total_ref += len(ref_words)

    return round(total_errors / max(1, total_ref), 4)
