def mark_to_market_pnl(cash, inventory, settlement_value):
    return cash + inventory * settlement_value
