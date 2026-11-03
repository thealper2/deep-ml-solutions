def hand_value(cards):
    total = sum(cards)
    usable_ace = 1 in cards and total + 10 <= 21

    if usable_ace:
        total += 10

    return (total, bool(usable_ace))