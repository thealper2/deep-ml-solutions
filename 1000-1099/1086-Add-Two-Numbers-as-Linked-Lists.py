def add_two_numbers(l1, l2):
    carry = 0
    result = []

    i, j = 0, 0

    while i < len(l1) or j < len(l2) or carry:
        digit_sum = carry
        if i < len(l1):
            digit_sum += l1[i]
            i += 1

        if j < len(l2):
            digit_sum += l2[j]
            j += 1

        result.append(digit_sum % 10)
        carry = digit_sum // 10

    return result
