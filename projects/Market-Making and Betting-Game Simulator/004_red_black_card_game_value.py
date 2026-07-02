def red_black_card_game_value(num_red, num_black):
    def dp(r, b):
        if r + b == 0:
            return 0.0

        total = r + b
        cont = 0.0

        if r:
            cont += (r / total) * (1.0 + dp(r - 1, b))
        if b:
            cont += (b / total) * (-1.0 + dp(r, b - 1))

        return max(0.0, cont)

    value = dp(num_red, num_black)
    return {
        "value": value,
        "stop_now": value == 0.0
    }
