import numpy as np

def minimax_alpha_beta(board, player, alpha, beta):
    """Return (best_score, best_move) for `player` using alpha-beta pruning."""
    status = get_game_status(board)
    if status != 'ongoing':
        return minimax_terminal_score(status), None

    moves = get_legal_moves(board)
    best_move = moves[0]

    if player == 1:
        best_score = float('-inf')
        for row, col in moves:
            new_board = place_move(board, row, col, player)
            score, _ = minimax_alpha_beta(new_board, switch_player(player), alpha, beta)
            if score > best_score:
                best_score = score
                best_move = (row, col)
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break
        return best_score, best_move
    else:
        best_score = float('inf')
        for row, col in moves:
            new_board = place_move(board, row, col, player)
            score, _ = minimax_alpha_beta(new_board, switch_player(player), alpha, beta)
            if score < best_score:
                best_score = score
                best_move = (row, col)
            beta = min(beta, best_score)
            if alpha >= beta:
                break

        return best_score, best_move
