def blackjack_step(player_cards, dealer_cards, action, rng):
    player_cards = player_cards.copy()
    dealer_cards = dealer_cards.copy()

    if action == 1:
        player_cards.append(draw_card(rng))
        player_total, _ = hand_value(player_cards)

        if player_total > 21:
            return player_cards, dealer_cards, -1.0, True

        return player_cards, dealer_cards, 0.0, False

    elif action == 0:
        while hand_value(dealer_cards)[0] < 17:
            dealer_cards.append(draw_card(rng))

        player_total, _ = hand_value(player_cards)
        dealer_total, _ = hand_value(dealer_cards)

        if dealer_total > 21 or player_total > dealer_total:
            reward = 1.0
        elif player_total == dealer_total:
            reward = 0.0
        else:
            reward = -1.0

        return player_cards, dealer_cards, float(reward), True