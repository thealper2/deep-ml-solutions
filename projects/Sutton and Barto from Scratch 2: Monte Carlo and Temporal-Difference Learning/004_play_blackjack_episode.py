def play_blackjack_episode(policy, rng, player_cards=None, dealer_cards=None):
    if player_cards is None:
        player_cards = [draw_card(rng), draw_card(rng)]

        while hand_value(player_cards)[0] < 12:
            player_cards.append(draw_card(rng))
    else:
        player_cards = player_cards.copy()

    if dealer_cards is None:
        dealer_cards = [draw_card(rng), draw_card(rng)]
    else:
        dealer_cards = dealer_cards.copy()

    episode = []
    done = False

    while not done:
        player_total, usable_ace = hand_value(player_cards)
        state = (player_total, dealer_cards[0], usable_ace)
        action = policy(state, rng)
        player_cards, dealer_cards, reward, done = blackjack_step(player_cards, dealer_cards, action, rng)
        episode.append((state, action, reward))

    return episode