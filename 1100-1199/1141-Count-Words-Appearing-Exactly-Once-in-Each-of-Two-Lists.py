from collections import Counter

def count_common_unique(list1, list2):
    count1 = Counter(list1)
    count2 = Counter(list2)

    unique_in_both = 0
    for word in count1:
        if count1[word] == 1 and count2.get(word, 0) == 1:
            unique_in_both += 1

    return unique_in_both
