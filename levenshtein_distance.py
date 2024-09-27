import numpy as np

def OSA(source: str, target: str) -> int:
    d = np.zeros((len(source) + 1, len(target) + 1))

    for i in range(1, len(source) + 1):
        d[i][0] = i
    
    for j in range(1, len(target) + 1):
        d[0][j] = j

    cost = 0
    for i in range(1, len(source) + 1):
        for j in range(1, len(target) + 1):
            if source[i - 1] != target[j - 1]:
                cost = 1
            else:
                cost = 0

            d[i][j] = min(
                d[i - 1][j] + 1, # deletion
                d[i][j - 1] + 1, # insertion,
                d[i - 1][j - 1] + cost # substitution
            )

            if i > 1 and j > 1 and source[i - 1] == target[j - 2] and source[i - 2] == target[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1) # transposition

    return d[-1][-1].astype(int)

source = "london"
target = "paris"

distance = OSA(source, target)
print(distance)