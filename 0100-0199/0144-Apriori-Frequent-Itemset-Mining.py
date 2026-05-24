import itertools
from collections import defaultdict

def apriori(transactions, min_support=0.5, max_length=None):
    """
    Returns: dict mapping frozenset(itemset) -> support (float)
    """
    if not transactions:
        raise ValueError()

    n_transactions = len(transactions)
    min_count = min_support * n_transactions

    transaction_list = [frozenset(t) for t in transactions]

    items = set()
    for t in transaction_list:
        items.update(t)

    frequent_itemsets = {}
    
    item_counts = defaultdict(int)
    for t in transaction_list:
        for item in t:
            item_counts[item] += 1

    current_itemsets = []
    for item, count in item_counts.items():
        if count >= min_count:
            itemset = frozenset([item])
            frequent_itemsets[itemset] = count / n_transactions
            current_itemsets.append(itemset)

    if max_length == 1:
        return frequent_itemsets

    k = 2
    while current_itemsets and (max_length is None or k <= max_length):
        candidates = set()
        for i in range(len(current_itemsets)):
            for j in range(i + 1, len(current_itemsets)):
                merged = current_itemsets[i] | current_itemsets[j]
                if len(merged) == k:
                    subsets_valid = True
                    for subset in itertools.combinations(merged, k - 1):
                        if frozenset(subset) not in frequent_itemsets:
                            subsets_valid = False
                            break

                    if subsets_valid:
                        candidates.add(merged)

        candidate_counts = defaultdict(int)
        for t in transaction_list:
            for candidate in candidates:
                if candidate.issubset(t):
                    candidate_counts[candidate] += 1

        current_itemsets = []
        for candidate, count in candidate_counts.items():
            if count >= min_count:
                frequent_itemsets[candidate] = count / n_transactions
                current_itemsets.append(candidate)

        k += 1

    return frequent_itemsets