"""
Market-Making & Betting-Game Simulator — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  expected_value ──
def expected_value(values, probabilities):
    v = np.array(values)
    p = np.array(probabilities)
    return float(np.sum(v * p))

# ── Step 002  one_reroll_die_value ──
def one_reroll_die_value(sides):
    probs = [1 / sides] * sides
    reroll_ev = expected_value(range(1, sides + 1), probs)
    reroll_faces = [face for face in range(1, sides + 1) if face < reroll_ev]
    optimal_values = [reroll_ev if face < reroll_ev else face for face in range(1, sides + 1)]
    value = expected_value(optimal_values, probs)

    return {
        'value': value,
        'reroll_faces': reroll_faces,
    }

# ── Step 003  pay_per_reroll_die_game ──
import numpy as np

def pay_per_reroll_die_game(sides, reroll_cost):
    best_threshold = 1
    best_value = -float("inf")

    for threshold in range(1, sides + 1):
        keep_values = np.arange(threshold, sides + 1, dtype=float)
        keep_prob = (sides - threshold + 1) / sides

        if keep_prob == 0:
            continue

        keep_mean = float(np.mean(keep_values))
        value = keep_mean - ((1 - keep_prob) / keep_prob) * reroll_cost

        if value > best_value + 1e-12:
            best_value = value
            best_threshold = threshold

    return {
        "threshold": best_threshold,
        "value": best_value
    }

# ── Step 004  red_black_card_game_value ──
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

# ── Step 005  make_quotes ──
def make_quotes(fair_value, spread_width):
    half_spread = spread_width / 2.0
    return {
        'bid': fair_value - half_spread,
        'ask': fair_value + half_spread,
    }

# ── Step 006  execute_trade ──
def execute_trade(state, side, bid, ask, size=1):
    new_state = state.copy()
    if side == 'buy':
        new_state['cash'] += ask * size
        new_state['inventory'] -= size
    else:
        new_state['cash'] -= bid * size
        new_state['inventory'] += size
    
    return new_state

# ── Step 007  mark_to_market_pnl ──
def mark_to_market_pnl(cash, inventory, settlement_value):
    return cash + inventory * settlement_value

# ── Step 008  adverse_selection_loss ──
import numpy as np

def adverse_selection_loss(fair_value, bid, ask, informed_values, informed_probabilities):
    loss = 0.0
    for v, p in zip(informed_values, informed_probabilities):
        if v > ask:
            loss += p * (v - ask)
        elif v < bid:
            loss += p * (bid - v)

    return loss

# ── Step 009  uncertainty_spread ──
def uncertainty_spread(base_spread, uncertainty):
    """Return a spread width >= base_spread that grows with uncertainty."""
    return base_spread + uncertainty

# ── Step 010  inventory_skewed_quotes ──
def inventory_skewed_quotes(fair_value, spread_width, inventory, skew_strength):
    half_spread = spread_width / 2.0
    shift = -inventory * skew_strength

    return {
        'bid': fair_value - half_spread + shift,
        'ask': fair_value + half_spread + shift,
    }

# ── Step 011  update_fair_value_from_trade ──
def update_fair_value_from_trade(fair_value, side, bid, ask, adjustment):
    if adjustment == 0:
        return fair_value
    if side == 'buy':
        return fair_value + adjustment
    else:
        return fair_value - adjustment

# ── Step 012  update_remaining_card_value ──
def update_remaining_card_value(remaining_counts, revealed_value):
    new_counts = remaining_counts.copy()

    if revealed_value in new_counts:
        new_counts[revealed_value] -= 1
        if new_counts[revealed_value] == 0:
            del new_counts[revealed_value]

    if not new_counts:
        expected_val = 0.0
    else:
        values = list(new_counts.keys())
        probabilities = [count / sum(new_counts.values()) for count in new_counts.values()]
        expected_val = expected_value(values, probabilities)

    return {
        'remaining_counts': new_counts,
        'expected_value': expected_val,
    }

# ── Step 013  run_market_making_episode ──
def run_market_making_episode(true_value, counterparty_sides, initial_fair_value, config):
    base_spread = config.get('base_spread', 0.0)
    uncertainty = config.get('uncertainty', 0.0)
    skew_strength = config.get('skew_strength', 0.0)
    belief_adjustment = config.get('belief_adjustment', 0.0)
    
    state = {'cash': 0.0, 'inventory': 0.0}
    fair_value = initial_fair_value
    history = []
    
    for side in counterparty_sides:
        spread_width = uncertainty_spread(base_spread, uncertainty)
        quotes = inventory_skewed_quotes(fair_value, spread_width, state['inventory'], skew_strength)
        bid = quotes['bid']
        ask = quotes['ask']
        
        state = execute_trade(state, side, bid, ask)
        
        history.append({
            'bid': bid,
            'ask': ask,
            'side': side,
            'cash': state['cash'],
            'inventory': state['inventory'],
            'fair_value': fair_value
        })
        
        fair_value = update_fair_value_from_trade(fair_value, side, bid, ask, belief_adjustment)
    
    pnl = mark_to_market_pnl(state['cash'], state['inventory'], true_value)
    
    return {
        'pnl': pnl,
        'cash': state['cash'],
        'inventory': state['inventory'],
        'fair_value': fair_value,
        'history': history
    }

# ── Step 014  summarize_episode_pnls ──
def summarize_episode_pnls(pnls):
    pnls = np.array(pnls)
    return {
        'mean': float(np.mean(pnls)),
        'std': float(np.std(pnls)),
        'worst': float(np.min(pnls))
    }

# ── Scaffold (runner) ──
"""End-to-end demo of the market-making & betting-game simulator."""

import numpy as np


def _format_rule(rule):
    if isinstance(rule, str):
        return rule
    if isinstance(rule, dict):
        return "policy dict"
    try:
        return sorted(rule)
    except TypeError:
        return rule


def main():
    np.random.seed(0)

    # --- Betting-game warm-ups: expected value & dice/card puzzles ---
    ev_die = expected_value([1, 2, 3, 4, 5, 6], [1/6] * 6)
    print(f"Fair 6-sided die EV: {ev_die:.4f}")

    rr = one_reroll_die_value(6)
    reroll_val = rr['value']
    reroll_set = rr['reroll_faces']
    print(f"One-reroll 6-sided die value: {reroll_val:.4f}, reroll if in {_format_rule(reroll_set)}")

    pp = pay_per_reroll_die_game(sides=6, reroll_cost=0.5)
    pay_reroll_ev = pp['value']
    threshold = pp['threshold']
    print(f"Pay-per-reroll game: stop when >= {threshold}, EV = {pay_reroll_ev:.4f}")

    cc = red_black_card_game_value(num_red=3, num_black=3)
    card_ev = cc['value']
    card_rule = cc.get('stop_now', None)
    print(f"Red/black card game EV: {card_ev:.4f} (rule sample: {card_rule})")

    # --- Quoting primitives ---
    fair_value = 100.0
    spread_width = 2.0
    q = make_quotes(fair_value, spread_width)
    bid, ask = q['bid'], q['ask']
    print(f"\nSymmetric quotes around {fair_value}: bid={bid:.2f}, ask={ask:.2f}")

    wq = make_quotes(fair_value, uncertainty_spread(base_spread=1.0, uncertainty=2.5))
    wide_bid, wide_ask = wq['bid'], wq['ask']
    print(f"Uncertainty-widened quotes: bid={wide_bid:.2f}, ask={wide_ask:.2f}")

    sq = inventory_skewed_quotes(fair_value, spread_width, inventory=3, skew_strength=0.2)
    skew_bid, skew_ask = sq['bid'], sq['ask']
    print(f"Inventory-skewed quotes (inv=+3): bid={skew_bid:.2f}, ask={skew_ask:.2f}")

    # --- Single trade + P&L mechanics ---
    state = {"cash": 0.0, "inventory": 0}
    state = execute_trade(state, side="buy", bid=bid, ask=ask, size=1)  # counterparty buys at our ask
    print(f"\nAfter counterparty buy: cash={state['cash']:.2f}, inv={state['inventory']}")
    mtm = mark_to_market_pnl(state["cash"], state["inventory"], settlement_value=fair_value)
    print(f"Mark-to-market P&L at {fair_value}: {mtm:.4f}")

    # --- Adverse selection: informed counterparty scenario ---
    adv_loss = adverse_selection_loss(
        fair_value=100.0, bid=99.0, ask=101.0,
        informed_values=[98.0, 100.0, 102.0],
        informed_probabilities=[1/3, 1/3, 1/3],
    )
    print(f"Expected adverse-selection loss per trade: {adv_loss:.4f}")

    # --- Belief updating primitives ---
    updated_fv = update_fair_value_from_trade(fair_value, side="buy", bid=bid, ask=ask, adjustment=0.1)
    print(f"Fair value after informed buy: {updated_fv:.4f}")

    remaining_counts = {+1: 3, -1: 3}
    urc = update_remaining_card_value(remaining_counts, revealed_value=+1)
    new_ev = urc['expected_value'] if isinstance(urc, dict) else urc
    print(f"Remaining-card EV after revealing red: {new_ev:.4f}")

    # --- Full episode simulation ---
    config = {
        "spread_width": 2.0,
        "skew_strength": 0.15,
        "belief_adjustment": 0.1,
        "size": 1,
        "uncertainty": 1.0,
        "base_spread": 1.0,
    }
    true_value = 101.5
    counterparty_sides = ["buy", "sell", "buy", "buy", "sell", "buy", "sell"]
    episode = run_market_making_episode(
        true_value=true_value,
        counterparty_sides=counterparty_sides,
        initial_fair_value=100.0,
        config=config,
    )
    print(f"\nEpisode result: {episode}")

    # --- Many-episode Monte Carlo summary ---
    pnls = []
    for _ in range(200):
        n_trades = np.random.randint(5, 15)
        sides = list(np.random.choice(["buy", "sell"], size=n_trades))
        tv = float(np.random.normal(100.0, 2.0))
        ep = run_market_making_episode(
            true_value=tv,
            counterparty_sides=sides,
            initial_fair_value=100.0,
            config=config,
        )
        pnl = ep["pnl"] if isinstance(ep, dict) and "pnl" in ep else (
            ep.get("final_pnl") if isinstance(ep, dict) else ep[-1]
        )
        pnls.append(float(pnl))

    summary = summarize_episode_pnls(pnls)
    print(f"\nAcross 200 episodes: {summary}")


if __name__ == "__main__":
    main()
