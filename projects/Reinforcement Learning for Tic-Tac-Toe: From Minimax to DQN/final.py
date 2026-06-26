"""
Reinforcement Learning for Tic-Tac-Toe: From Minimax to DQN — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  create_empty_board ──
import numpy as np

def create_empty_board():
    """Return an empty 3x3 Tic-Tac-Toe board as an int numpy array of zeros."""
    return np.zeros((3, 3), dtype=int)

# ── Step 002  encode_player ──
def encode_player(player):
    """Return the integer encoding for 'X', 'O', or 'empty'."""
    d = {'X': 1, 'O': -1, 'empty': 0}
    return d[player]

# ── Step 003  print_board ──
import numpy as np

def print_board(board):
    """Print the 3x3 board using X, O, and . characters."""
    for row in board:
        line = []
        for cell in row:
            if cell == 1:
                line.append('X')
            elif cell == -1:
                line.append('O')
            else:
                line.append('.')

        print(' '.join(line))

# ── Step 004  is_cell_empty ──
import numpy as np

def is_cell_empty(board, row, col):
    """Return True if board[row, col] is empty (0), else False."""
    return board[row][col] == 0

# ── Step 005  place_move ──
import numpy as np

def place_move(board, row, col, player):
    """Place player's mark at (row, col) and return the new board."""
    if not is_cell_empty(board, row, col):
        raise ValueError('Cell is already occupied')

    new_board = board.copy()
    new_board[row][col] = player
    return new_board

# ── Step 006  get_legal_moves ──
import numpy as np

def get_legal_moves(board):
    """Return a list of (row, col) tuples for all empty cells on the board."""
    row, col = board.shape
    result = []

    for r in range(row):
        for c in range(col):
            if board[r][c] == 0:
                result.append((r, c))

    return result

# ── Step 007  check_row_win ──
import numpy as np

def check_row_win(board, player):
    """Return True if `player` has three-in-a-row across any row of `board`."""
    for row in board:
        if np.all(row == player):
            return True

    return False

# ── Step 008  check_column_win ──
import numpy as np

def check_column_win(board, player):
    """Return True if `player` has three-in-a-row in any column of `board`."""
    for col in range(board.shape[1]):
        if np.all(board[:, col] == player):
            return True

    return False

# ── Step 009  check_main_diagonal_win ──
import numpy as np

def check_main_diagonal_win(board, player):
    """Return True if `player` occupies all three main-diagonal cells."""
    return np.all(np.diag(board) == player)

# ── Step 010  check_anti_diagonal_win ──
import numpy as np

def check_anti_diagonal_win(board, player):
    return np.all(np.diag(np.fliplr(board)) == player)

# ── Step 011  is_winner ──
import numpy as np

def is_winner(board, player):
    """Return True if `player` has three-in-a-row on `board`."""
    if check_row_win(board, player):
        return True

    if check_column_win(board, player):
        return True

    if check_main_diagonal_win(board, player):
        return True

    if check_anti_diagonal_win(board, player):
        return True

    return False

# ── Step 012  is_draw ──
import numpy as np

def is_draw(board):
    """Return True iff the board is full and neither player has won."""
    if len(get_legal_moves(board)) > 0:
        return False

    if is_winner(board, 1) or is_winner(board, -1):
        return False

    return True

# ── Step 013  get_game_status ──
import numpy as np

def get_game_status(board):
    """Return 'X_win', 'O_win', 'draw', or 'ongoing' for the given 3x3 board."""
    if is_winner(board, 1):
        return 'X_win'
    if is_winner(board, -1):
        return 'O_win'
    if is_draw(board):
        return 'draw'
    
    return 'ongoing'

# ── Step 014  get_current_player ──
import numpy as np

def get_current_player(board):
    """Return 1 if X is to move, -1 if O is to move."""
    x_count = np.sum(board == 1)
    o_count = np.sum(board == -1)

    if x_count == o_count:
        return 1
    else:
        return -1

# ── Step 015  switch_player ──
def switch_player(player):
    """Return the opponent of `player` (1 <-> -1)."""
    return -player

# ── Step 016  play_hardcoded_game ──
import numpy as np

def play_hardcoded_game(moves):
    """Replay a fixed sequence of (row, col) moves and return (final_board, status)."""
    board = np.zeros((3, 3), dtype=int)
    player = 1
    status = 'ongoing'

    for row, col in moves:
        try:
            board = place_move(board, row, col, player)
        except ValueError:
            break

        status = get_game_status(board)
        if status != 'ongoing':
            break

        player = switch_player(player)

    return board, status

# ── Step 017  play_interactive_game ──
def play_interactive_game():
    """Play a full game with two humans entering moves via stdin and return the final status."""
    board = np.zeros((3, 3), dtype=int)
    player = 1
    status = 'ongoing'

    print_board(board)

    while get_legal_moves(board) != []:
        try:
            row, col = map(int, input().split())
            board = place_move(board, row, col, player)
        except ValueError:
            print_board(board)
            continue

        print_board(board)

        status = get_game_status(board)
        if status != 'ongoing':
            break

        player = switch_player(player)

    return status

# ── Step 018  TicTacToeGame ──
class TicTacToeGame:
    """Stateful Tic-Tac-Toe environment wrapping the Part 1 engine."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.status = 'ongoing'
        return self.board
    
    def legal_moves(self):
        return get_legal_moves(self.board)
    
    def is_terminal(self):
        return self.status != 'ongoing'
    
    def step(self, row, col):
        if self.is_terminal():
            raise ValueError("Game is already over")
        
        if not is_cell_empty(self.board, row, col):
            raise ValueError("Cell is already occupied")
        
        self.board = place_move(self.board, row, col, self.current_player)
        self.status = get_game_status(self.board)
        
        if self.status == 'ongoing':
            self.current_player = switch_player(self.current_player)
        
        return self.board.copy(), self.status

# ── Step 019  random_move_agent ──
import numpy as np

def random_move_agent(board, player, rng):
    """Return a uniformly random legal (row, col) move for `player`."""
    moves = get_legal_moves(board)
    idx = rng.integers(0, len(moves))
    return moves[idx]

# ── Step 020  play_random_vs_random_game ──
def play_random_vs_random_game(rng):
    """Simulate one full random-vs-random game and return the final status."""
    game = TicTacToeGame()

    while not game.is_terminal():
        moves = game.legal_moves()
        row, col = random_move_agent(game.board, moves, rng)
        game.step(row, col)

    return game.status

# ── Step 021  play_random_vs_random_matches ──
def play_random_vs_random_matches(n_games, rng):
    """Run n_games random-vs-random games and return the list of outcome strings."""
    results = []
    for _ in range(n_games):
        result = play_random_vs_random_game(rng)
        results.append(result)

    return results

# ── Step 022  compute_outcome_rates ──
def compute_outcome_rates(outcomes):
    """Return {'x_win_rate','o_win_rate','draw_rate'} from a list of outcome labels."""
    n = len(outcomes)
    if not outcomes:
        return {'x_win_rate': 0.0, 'o_win_rate': 0.0, 'draw_rate': 0.0}
    
    x_win_rate = outcomes.count('X_win') / n
    o_win_rate = outcomes.count('O_win') / n
    if x_win_rate == 0.0 and o_win_rate == 0.0:
        draw_rate = 1.0
    else:
        draw_rate = outcomes.count('draw') / n
    return {'x_win_rate': x_win_rate, 'o_win_rate': o_win_rate, 'draw_rate': draw_rate}

# ── Step 023  minimax_terminal_score ──
def minimax_terminal_score(status):
    """Return +1 for 'X_win', -1 for 'O_win', 0 for 'draw'."""
    d = {'X_win': 1, 'O_win': -1, 'draw': 0}
    return d[status]

# ── Step 024  minimax_value ──
def minimax_value(board, player):
    """Return the minimax value of `board` with `player` to move."""
    status = get_game_status(board)
    if status != 'ongoing':
        return minimax_terminal_score(status)

    moves = get_legal_moves(board)
    if player == 1:
        best = float('-inf')
        for row, col in moves:
            new_board = place_move(board, row, col, player)
            value = minimax_value(new_board, switch_player(player))
            best = max(best, value)
        return best
    else:
        best = float('inf')
        for row, col in moves:
            new_board = place_move(board, row, col, player)
            value = minimax_value(new_board, switch_player(player))
            best = min(best, value)
        return best

# ── Step 025  minimax_recursive ──
cache = {}

def minimax_recursive(board, player):
    """Return the minimax value of `board` with `player` to move."""
    key = (board.tobytes(), player)
    if key in cache:
        return cache[key]
    
    status = get_game_status(board)
    if status != 'ongoing':
        value = minimax_terminal_score(status)
        cache[key] = value
        return value
    
    moves = get_legal_moves(board)
    if player == 1:
        best = -float('inf')
        for row, col in moves:
            new_board = place_move(board, row, col, player)
            value = minimax_recursive(new_board, switch_player(player))
            best = max(best, value)
        cache[key] = best
        return best
    else:
        best = float('inf')
        for row, col in moves:
            new_board = place_move(board, row, col, player)
            value = minimax_recursive(new_board, switch_player(player))
            best = min(best, value)
        cache[key] = best
        return best

# ── Step 026  minimax_max_min_step ──
import numpy as np

def minimax_max_min_step(board, player):
    """Return (best_score, best_move) after expanding one minimax level."""
    moves = get_legal_moves(board)
    best_score = None
    best_move = None

    if player == 1:
        best_score = float('-inf')
        for row, col in moves:
            new_board = place_move(board, row, col, player)
            score = minimax_recursive(new_board, switch_player(player))
            if score > best_score:
                best_score = score
                best_move = (row, col)

    else:
        best_score = float('inf')
        for row, col in moves:
            new_board = place_move(board, row, col, player)
            score = minimax_recursive(new_board, switch_player(player))
            if score < best_score:
                best_score = score
                best_move = (row, col)

    return best_score, best_move

# ── Step 027  minimax_best_move ──
def minimax_best_move(board, player):
    """Return the optimal (row, col) move for `player` via minimax."""
    _, best_move = minimax_max_min_step(board, player)
    return best_move

# ── Step 028  minimax_alpha_beta ──
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

# ── Step 029  play_minimax_vs_random_matches ──
def play_minimax_vs_random_matches(n_games, minimax_plays_x, rng):
    outcomes = []

    for _ in range(n_games):
        game = TicTacToeGame()

        while not game.is_terminal():
            board = game.board
            player = game.current_player

            if (minimax_plays_x and player == 1) or \
               (not minimax_plays_x and player == -1):
                _, move = minimax_max_min_step(board, player)
            else:
                move = random_move_agent(board, player, rng)

            row, col = move
            game.step(row, col)

        outcomes.append(game.status)

    return compute_outcome_rates(outcomes)

# ── Step 030  play_minimax_vs_minimax_matches ──
def play_minimax_vs_minimax_matches(n_games):
    """Play n_games minimax-vs-minimax games and report outcome rates plus an all_draws flag."""
    outcomes = []
    
    for _ in range(n_games):
        game = TicTacToeGame()
        
        while not game.is_terminal():
            board = game.board
            player = game.current_player
            
            _, move = minimax_alpha_beta(board, player, -float('inf'), float('inf'))
            row, col = move
            game.step(row, col)
        
        outcomes.append(game.status)
    
    rates = compute_outcome_rates(outcomes)
    rates['all_draws'] = rates['draw_rate'] == 1.0 if n_games > 0 else True
    
    return rates

# ── Step 031  encode_board_state_key ──
import numpy as np

def encode_board_state_key(board):
    """Encode a 3x3 board as a length-9 string over {'0','1','2'} in row-major order."""
    d = {0: '0', 1: '1', -1: '2'}
    result = ''
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            result += d[board[row][col]]
                
    return result

# ── Step 032  canonical_board_key ──
def canonical_board_key(board):
    rotate_90 = lambda x: np.rot90(x)
    reflect = lambda x: np.fliplr(x)

    candidates = []
    current = board.copy()
    for _ in range(4):
        candidates.append(current)
        current = rotate_90(current)

    reflected = reflect(board)
    current = reflected.copy()
    for _ in range(4):
        candidates.append(current)
        current = rotate_90(current)

    keys = [encode_board_state_key(b) for b in candidates]
    return min(keys)

# ── Step 033  initialize_q_table ──
from collections import defaultdict

def initialize_q_table():
    """Create an empty Q-table that returns 0.0 for unseen (state, action) keys."""
    return defaultdict(float)

# ── Step 034  get_q_value ──
def get_q_value(q_table, state_key, action):
    return q_table.get((state_key, action), 0.0)

# ── Step 035  set_q_value ──
def set_q_value(q_table, state_key, action, value):
    """Write a new Q-value for a (state, action) pair into the Q-table."""
    q_table[(state_key, action)] = value

# ── Step 036  choose_learning_rate_alpha ──
def choose_learning_rate_alpha():
    """Return the learning rate alpha (float in (0, 1]) for tabular Q-learning."""
    return 0.1

# ── Step 037  choose_discount_factor_gamma ──
def choose_discount_factor_gamma():
    """Return the discount factor gamma in [0, 1] for Q-learning."""
    return 0.9

# ── Step 038  choose_initial_epsilon ──
def choose_initial_epsilon():
    """Return the starting exploration rate epsilon for epsilon-greedy."""
    return 1.0

# ── Step 039  epsilon_decay_schedule ──
import numpy as np

def epsilon_decay_schedule(initial_epsilon, episode_index, min_epsilon, decay_rate):
    """Return the decayed epsilon for the given episode, clipped to min_epsilon."""
    epsilon = initial_epsilon * np.exp(-decay_rate * episode_index)
    return max(min_epsilon, epsilon)

# ── Step 040  epsilon_greedy_explore_move ──
def epsilon_greedy_explore_move(legal_actions, rng):
    """Sample a uniformly random legal action from legal_actions using rng."""
    idx = rng.integers(0, len(legal_actions))
    return legal_actions[idx]

# ── Step 041  epsilon_greedy_select_action ──
def epsilon_greedy_select_action(q_table, state_key, legal_actions, epsilon, rng):
    """Choose an action via epsilon-greedy over the legal actions."""
    if rng.random() < epsilon:
        return epsilon_greedy_explore_move(legal_actions, rng)
    else:
        best_action = legal_actions[0]
        best_value = get_q_value(q_table, state_key, best_action)
        for action in legal_actions[1:]:
            value = get_q_value(q_table, state_key, action)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

# ── Step 042  greedy_argmax_over_legal_actions ──
def greedy_argmax_over_legal_actions(q_table, state_key, legal_actions, rng):
    """Return the legal action with the highest Q-value (random tie-break)."""
    best_value = float('-inf')
    best_actions = []

    for action in legal_actions:
        value = get_q_value(q_table, state_key, action)
        if value > best_value:
            best_value = value
            best_actions = [action]
        elif value == best_value:
            best_actions.append(action)

    idx = rng.integers(0, len(best_actions))
    return best_actions[idx]

# ── Step 043  random_tie_break_argmax ──
def random_tie_break_argmax(values, candidates, rng):
    """Return one candidate whose value equals max(values), tie-broken uniformly at random."""
    max_value = max(values)
    max_indices = [i for i, v in enumerate(values) if v == max_value]
    idx = rng.integers(0, len(max_indices))
    return candidates[max_indices[idx]]

# ── Step 044  tic_tac_toe_reward ──
def tic_tac_toe_reward(game_status, agent_player):
    """Return scalar reward from the agent's perspective.

    game_status: one of 'X_win', 'O_win', 'draw', 'ongoing'.
    agent_player: +1 for X, -1 for O.
    """
    if game_status == 'X_win':
        return 1.0 if agent_player == 1 else -1.0
    elif game_status == 'O_win':
        return 1.0 if agent_player == -1 else -1.0
    else:
        return 0.0

# ── Step 045  q_learning_nonterminal_target ──
def q_learning_nonterminal_target(reward, gamma, q_table, next_state_key, next_legal_actions):
    """Return the TD target r + gamma * max_a' Q(s', a') over legal next actions."""
    if not next_legal_actions:
        return reward
        
    max_q = max(get_q_value(q_table, next_state_key, a) for a in next_legal_actions)
    return reward + gamma * max_q

# ── Step 046  q_learning_terminal_target ──
def q_learning_terminal_target(reward):
    """Return the TD target for a terminal transition."""
    return reward

# ── Step 047  q_learning_update ──
def q_learning_update(q_table, state_key, action, target, alpha):
    """Apply Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a)) and return the new value."""
    new_value = get_q_value(q_table, state_key, action) + alpha * (target - get_q_value(q_table, state_key, action))
    set_q_value(q_table, state_key, action, new_value)
    return new_value

# ── Step 048  episode_reset_game ──
import numpy as np

def episode_reset_game():
    """Return a fresh empty board and the starting player (+1 for X)."""
    board = create_empty_board()
    player = 1
    return board, player

# ── Step 049  episode_agent_pick_action ──
def episode_agent_pick_action(q_table, board, current_player, epsilon, rng):
    state_key = canonical_board_key(board)
    moves = get_legal_moves(board)
    legal_actions = [row * 3 + col for row, col in moves]
    action = epsilon_greedy_select_action(q_table, state_key, legal_actions, epsilon, rng)
    return state_key, action

# ── Step 050  episode_apply_action ──
def episode_apply_action(board, action, current_player, agent_player):
    """Apply one move, return next_board/next_player/status/reward/done."""
    row = action // 3
    col = action % 3
    
    new_board = place_move(board, row, col, current_player)
    status = get_game_status(new_board)
    
    reward = tic_tac_toe_reward(status, agent_player)
    done = (status != 'ongoing')
    
    next_player = switch_player(current_player)
    
    return {
        'next_board': new_board,
        'next_player': next_player,
        'status': status,
        'reward': reward,
        'done': done
    }

# ── Step 051  episode_apply_q_update ──
def episode_apply_q_update(q_table, state_key, action, reward, next_board, done, alpha, gamma):
    """Compute the TD target (terminal or nonterminal) and apply the Q-learning update."""
    if done:
        target = q_learning_terminal_target(reward)
    else:
        next_state_key = canonical_board_key(next_board)
        next_legal_actions = get_legal_moves(next_board)
        target = q_learning_nonterminal_target(reward, gamma, q_table, next_state_key, next_legal_actions)
    
    old_q = get_q_value(q_table, state_key, action)
    new_q = old_q + alpha * (target - old_q)
    q_table[(state_key, action)] = new_q
    return new_q

# ── Step 052  episode_check_terminate ──
def episode_check_terminate(status):
    """Return True if status is terminal (win or draw), else False."""
    return status != 'ongoing'

# ── Step 053  train_q_learning_agent ──
def train_q_learning_agent(num_episodes, alpha, gamma, initial_epsilon, min_epsilon, decay_rate, opponent_policy, rng):
    q_table = initialize_q_table()
    episode_outcomes = []
    agent_player = 1
    
    for episode in range(num_episodes):
        epsilon = epsilon_decay_schedule(initial_epsilon, episode, min_epsilon, decay_rate)
        
        game = TicTacToeGame()
        
        while not game.is_terminal():
            board = game.board
            current_player = game.current_player
            
            if current_player == agent_player:
                state_key, action = episode_agent_pick_action(q_table, board, current_player, epsilon, rng)
                out = episode_apply_action(board, action, current_player, agent_player)
                
                next_board = out['next_board']
                done = out['done']
                reward = out['reward']
                episode_apply_q_update(q_table, state_key, action, reward, next_board, done, alpha, gamma)
                
                game.board = next_board
                game.current_player = out['next_player']
                game.status = out['status']
                
                if game.is_terminal():
                    break
            else:
                legal_moves = get_legal_moves(board)
                action = opponent_policy(board, current_player, rng)
                out = episode_apply_action(board, action, current_player, agent_player)
                
                next_board = out['next_board']
                done = out['done']
                reward = out['reward']
                state_key = canonical_board_key(board)
                episode_apply_q_update(q_table, state_key, action, reward, next_board, done, alpha, gamma)
                
                game.board = next_board
                game.current_player = out['next_player']
                game.status = out['status']
        
        episode_outcomes.append(game.status)
    
    return {'q_table': q_table, 'episode_outcomes': episode_outcomes}

# ── Step 054  compute_batched_outcome_stats ──
import numpy as np

def compute_batched_outcome_stats(episode_outcomes, batch_size):
    """Aggregate outcomes into per-batch win/loss/draw rates."""
    win_rates = []
    loss_rates = []
    draw_rates = []
    batch_indices = []
    
    for i in range(0, len(episode_outcomes) - batch_size + 1, batch_size):
        batch = episode_outcomes[i:i+batch_size]
        n = len(batch)
        wins = batch.count('win')
        losses = batch.count('loss')
        draws = batch.count('draw')

        win_rates.append(wins / n)
        loss_rates.append(losses / n)
        draw_rates.append(draws / n)
        batch_indices.append(i // batch_size)

    return {
        'batch_index': np.array(batch_indices),
        'win_rate': np.array(win_rates),
        'loss_rate': np.array(loss_rates),
        'draw_rate': np.array(draw_rates),
    }

# ── Step 055  self_play_episode ──
def self_play_episode(q_table, alpha, gamma, epsilon, rng):
    """Run one self-play episode and return final_status and a list of transitions."""
    board, player = episode_reset_game()
    transitions = []
    
    while True:
        state_key, action = episode_agent_pick_action(q_table, board, player, epsilon, rng)
        out = episode_apply_action(board, action, player, player)
        
        transitions.append({
            'state_key': state_key,
            'action': action,
            'reward': out['reward'],
            'next_board': out['next_board'],
            'done': out['done'],
            'player': player
        })
        
        board = out['next_board']
        if out['done']:
            final_status = out['status']
            break
        
        player = out['next_player']
    
    return {
        'final_status': final_status,
        'transitions': transitions
    }

# ── Step 056  flip_board_perspective ──
import numpy as np

def flip_board_perspective(board, current_player):
    """Return a board view where current_player's marks are +1."""
    return board.copy() if current_player == 1 else -board

# ── Step 057  perspective_reward_sign ──
def perspective_reward_sign(reward, acting_player, scoring_player):
    """Return reward expressed from acting_player's perspective."""
    return reward if acting_player == scoring_player else -reward

# ── Step 058  train_q_agent_self_play ──
def train_q_agent_self_play(num_episodes, alpha, gamma, initial_epsilon, min_epsilon, decay_rate, rng):
    q_table = initialize_q_table()
    episode_outcomes = []

    for episode in range(num_episodes):
        epsilon = epsilon_decay_schedule(initial_epsilon, episode, min_epsilon, decay_rate)

        result = self_play_episode(q_table, alpha, gamma, epsilon, rng)
        final_status = result['final_status']
        transitions = result['transitions']

        for transition in transitions:
            state_key = transition['state_key']
            action = transition['action']
            reward = transition['reward']
            next_board = transition['next_board']
            done = transition['done']
            player = transition['player']

            flipped_state_key = canonical_board_key(flip_board_perspective(next_board, player))
            flipped_reward = perspective_reward_sign(reward, player, 1)
            episode_apply_q_update(q_table, state_key, action, flipped_reward, next_board, done, alpha, gamma)

        episode_outcomes.append(final_status)

    return {
        'q_table': q_table,
        'episode_outcomes': episode_outcomes,
    }

# ── Step 059  evaluate_q_agent_vs_random ──
def evaluate_q_agent_vs_random(q_table, num_games, rng):
    """Play num_games between the greedy Q-agent and a random opponent.

    Returns a dict with keys 'wins', 'losses', 'draws' (ints) and
    'win_rate', 'loss_rate', 'draw_rate' (floats), all from the agent's
    perspective. The agent alternates between playing X and O across games.
    """
    wins = 0
    losses = 0
    draws = 0
    
    for game_idx in range(num_games):
        game = TicTacToeGame()
        
        agent_player = 1 if game_idx % 2 == 0 else -1
        agent_side = 'X' if agent_player == 1 else 'O'
        
        while not game.is_terminal():
            board = game.board
            current_player = game.current_player
            
            if current_player == agent_player:
                state_key = canonical_board_key(board)
                legal_moves = get_legal_moves(board)
                legal_actions = [row * 3 + col for row, col in legal_moves]
                action = greedy_argmax_over_legal_actions(q_table, state_key, legal_actions, rng)
                out = episode_apply_action(board, action, current_player, agent_player)
            else:
                legal_moves = get_legal_moves(board)
                action = random_move_agent(board, current_player, rng)
                row, col = action
                action_flat = row * 3 + col
                out = episode_apply_action(board, action_flat, current_player, agent_player)
            
            game.board = out['next_board']
            game.current_player = out['next_player']
            game.status = out['status']
        
        status = game.status
        if status == 'X_win':
            if agent_player == 1:
                wins += 1
            else:
                losses += 1
        elif status == 'O_win':
            if agent_player == -1:
                wins += 1
            else:
                losses += 1
        else:
            draws += 1
    
    total = num_games
    if total == 0:
        return {'wins': 0, 'losses': 0, 'draws': 0, 'win_rate': 0.0, 'loss_rate': 0.0, 'draw_rate': 0.0}
    
    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': wins / total,
        'loss_rate': losses / total,
        'draw_rate': draws / total
    }

# ── Step 060  evaluate_q_agent_vs_minimax ──
def evaluate_q_agent_vs_minimax(q_table, num_games, rng):
    outcomes = []

    for game_idx in range(num_games):
        game = TicTacToeGame()

        agent_player = 1 if game_idx % 2 == 0 else -1

        while not game.is_terminal():
            board = game.board
            current_player = game.current_player

            if current_player == agent_player:
                state_key = canonical_board_key(board)
                legal_moves = get_legal_moves(board)
                legal_actions = [row * 3 + col for row, col in legal_moves]
                action = greedy_argmax_over_legal_actions(q_table, state_key, legal_actions, rng)
                out = episode_apply_action(board, action, current_player, agent_player)

            else:
                score, move = minimax_alpha_beta(board, current_player, float('inf'), float('-inf'))
                row, col = move
                action_flat = row * 3 + col
                out = episode_apply_action(board, action_flat, current_player, agent_player)

            game.board = out['next_board']
            game.current_player = out['next_player']
            game.status = out['status']

        status = game.status
        if status == 'X_win':
            outcome = 'win' if agent_player == 1 else 'loss'
        elif status == 'Q_win':
            outcome = 'win' if agent_player == 0 else 'loss'
        else:
            outcome = 'draw'

        outcomes.append(outcome)

    return compute_outcome_rates(outcomes)

# ── Step 061  inspect_q_values_for_state ──
import numpy as np

def inspect_q_values_for_state(q_table, board, current_player):
    """Print the board and Q-values for all 9 cells; return a length-9 array."""
    print_board(board)

    state_key = canonical_board_key(board)

    values = np.zeros(9)
    for i in range(9):
        row = i // 3
        col = i % 3
        values[i] = get_q_value(q_table, state_key, (row, col))

    for row in range(3):
        line = []
        for  col in range(3):
            idx = row * 3 + col
            line.append(f"{values[idx]:+.2f}")

        print(' '.join(line))

    return values

# ── Step 062  serialize_q_table_to_dict ──
def serialize_q_table_to_dict(q_table):
    """Convert a Q-table (str -> np.ndarray shape (9,)) into a plain dict (str -> list of floats)."""
    result = {}
    for key, value in q_table.items():
        if isinstance(value, np.ndarray):
            result[key] = value.astype(float).tolist()
        else:
            result[key] = float(value)

    return result

# ── Step 063  deserialize_q_table_from_dict ──
import numpy as np

def deserialize_q_table_from_dict(serialized):
    """Rebuild a Q-table (state_key -> np.ndarray shape (9,)) from a plain dict."""
    result = {}
    for key, value in serialized.items():
        result[key] = np.array(value, dtype=np.float64)
        
    return result

# ── Step 064  encode_board_flat_length_nine ──
import numpy as np

def encode_board_flat_length_nine(board, current_player):
    """Encode a 3x3 board as a length-9 float32 vector from current_player's view."""
    flipped = flip_board_perspective(board, current_player)
    return flipped.flatten().astype(np.float32)

# ── Step 065  encode_board_one_hot_length_eighteen ──
import numpy as np

def encode_board_one_hot_length_eighteen(board, current_player):
    """Encode a 3x3 board as a length-18 two-channel one-hot vector."""
    flipped = flip_board_perspective(board, current_player)
    own = (flipped == 1).astype(np.float32).flatten()
    opp = (flipped == -1).astype(np.float32).flatten()
    return np.concatenate([own, opp])

# ── Step 066  build_mlp_architecture ──
def build_mlp_architecture(input_dim, hidden_dim, output_dim=9):
    return dict(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# ── Step 067  initialize_mlp_parameters ──
def initialize_mlp_parameters(architecture, seed=0):
    """Initialize MLP weights with He init and zero biases.

    architecture: dict from build_mlp_architecture with input_dim, hidden_dim, output_dim.
    seed: int seed for numpy RNG.
    Returns dict with keys 'W1', 'b1', 'W2', 'b2'.
    """
    np.random.seed(seed)
    input_dim = architecture['input_dim']
    hidden_dim = architecture['hidden_dim']
    output_dim = architecture['output_dim']
    
    std1 = np.sqrt(2.0 / input_dim)
    W1 = np.random.randn(input_dim, hidden_dim) * std1

    std2 = np.sqrt(2.0 / hidden_dim)
    W2 = np.random.randn(hidden_dim, output_dim) * std2

    b1 = np.zeros(hidden_dim)
    b2 = np.zeros(output_dim)

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

# ── Step 068  mlp_forward_pass ──
def mlp_forward_pass(params, x):
    """Forward pass through a two-layer MLP with ReLU hidden activation.

    Args:
        params: dict with keys 'W1', 'b1', 'W2', 'b2'.
        x: np.ndarray of shape (batch, input_dim).

    Returns:
        (q_values, cache) where q_values has shape (batch, output_dim) and
        cache is a dict with keys {'x', 'z1', 'h1', 'q'}.
    """
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    
    z1 = x @ W1 + b1
    h1 = np.maximum(0, z1)
    q = h1 @ W2 + b2

    cache = {'x': x, 'z1': z1, 'h1': h1, 'q': q}
    return q, cache

# ── Step 069  mask_illegal_actions_neg_inf ──
import numpy as np

def mask_illegal_actions_neg_inf(q_values, legal_action_mask):
    """Return a copy of q_values with illegal entries set to -inf."""
    masked_q_values = q_values.copy()
    masked_q_values[~legal_action_mask] = float('-inf')
    return masked_q_values

# ── Step 070  argmax_action_from_q_values ──
import numpy as np

def argmax_action_from_q_values(masked_q_values):
    """Return the index of the largest entry in masked_q_values as an int."""
    return np.argmax(masked_q_values)

# ── Step 071  mse_loss_on_chosen_action ──
import numpy as np

def mse_loss_on_chosen_action(predicted_q, action_indices, target_q):
    """MSE between Q(s, a_taken) and the bootstrapped target Q."""
    chosen_q = predicted_q[np.arange(len(action_indices)), action_indices]
    return np.mean((chosen_q - target_q) ** 2)

# ── Step 072  mlp_backward_pass ──
def mlp_backward_pass(params, cache, action_indices, target_q):
    """Backprop MSE-on-chosen-action loss through the MLP and return param gradients."""
    x, z1, h1, q = cache['x'], cache['z1'], cache['h1'], cache['q']
    W2 = params['W2']
    batch_size = x.shape[0]

    dq = np.zeros_like(q)
    dq[np.arange(batch_size), action_indices] = 2 * (q[np.arange(batch_size), action_indices] - target_q) / batch_size

    db2 = np.sum(dq, axis=0)
    dW2 = h1.T @ dq

    dh1 = dq @ W2.T
    dz1 = dh1 * (z1 > 0)

    db1 = np.sum(dz1, axis=0)
    dW1 = x.T @ dz1

    return {
        'W1': dW1,
        'b1': db1,
        'W2': dW2,
        'b2': db2,
    }

# ── Step 073  adam_update_step ──
import numpy as np

def adam_update_step(params, grads, adam_state, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    if 't' not in adam_state:
        adam_state['t'] = 0
        adam_state['m'] = {}
        adam_state['v'] = {}
        for key in params:
            adam_state['m'][key] = np.zeros_like(params[key])
            adam_state['v'][key] = np.zeros_like(params[key])
    
    adam_state['t'] += 1
    t = adam_state['t']
    
    new_params = {}
    for key in params:
        grad = grads[key]
        m = adam_state['m'][key]
        v = adam_state['v'][key]
        
        m_new = beta1 * m + (1 - beta1) * grad
        adam_state['m'][key] = m_new
        
        v_new = beta2 * v + (1 - beta2) * (grad ** 2)
        adam_state['v'][key] = v_new
        
        m_hat = m_new / (1 - beta1 ** t)
        v_hat = v_new / (1 - beta2 ** t)
        
        new_params[key] = params[key] - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
    
    return new_params, adam_state

# ── Step 074  create_replay_buffer ──
from collections import deque


def create_replay_buffer(capacity):
    """Return an empty replay buffer with a fixed maximum capacity."""
    return {
        'data': deque(maxlen=capacity),
        'capacity': capacity
    }

# ── Step 075  append_transition_to_buffer ──
def append_transition_to_buffer(buffer, state, action, reward, next_state, done, next_legal_mask):
    """Append one (s, a, r, s', done, next_legal_mask) transition to the replay buffer."""
    buffer['data'].append((state, action, reward, next_state, done, next_legal_mask))
    return buffer

# ── Step 076  cap_buffer_size_drop_oldest ──
def cap_buffer_size_drop_oldest(buffer):
    """Drop oldest transitions until len(buffer['data']) <= buffer['capacity']."""
    while len(buffer['data']) > buffer['capacity']:
        buffer['data'].pop(0)

    return buffer

# ── Step 077  sample_minibatch_from_buffer ──
import numpy as np


def sample_minibatch_from_buffer(buffer, batch_size, rng):
    """Draw `batch_size` random transitions from `buffer` and stack fields into arrays."""
    data = buffer['data']
    indices = rng.integers(0, len(data), size=batch_size)

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    next_legal_masks = []

    for idx in indices:
        t = data[int(idx)]
        if isinstance(t, dict):
            state = t['state']
            action = t['action']
            reward = t['reward']
            next_state = t['next_state']
            done = t['done']
            next_legal_mask = t['next_legal_mask']
        else:
            state, action, reward, next_state, done, next_legal_mask = t
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
        next_legal_masks.append(next_legal_mask)

    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'next_states': np.array(next_states),
        'dones': np.array(dones),
        'next_legal_masks': np.array(next_legal_masks),
    }

# ── Step 078  build_target_network_copy ──
import numpy as np

def build_target_network_copy(online_params):
    """Return a deep copy of the online MLP parameter dict."""
    return {key: value.copy() for key, value in online_params.items()}

# ── Step 079  compute_target_q_with_target_network ──
import numpy as np

def compute_target_q_with_target_network(target_params, batch, gamma):
    """Compute DQN bootstrap targets r + gamma * max_a' Q_target(s', a')."""
    next_states = batch['next_states']
    rewards = batch['rewards']
    dones = batch['dones']
    next_legal_masks = batch['next_legal_masks']
    batch_size = next_states.shape[0]
    
    q_next, _ = mlp_forward_pass(target_params, next_states)
    q_masked = np.where(next_legal_masks, q_next, -np.inf)
    max_q = np.max(q_masked, axis=1)
    max_q = np.where(np.isinf(max_q), 0.0, max_q)
    targets = rewards + gamma * max_q * (~dones)
    
    return targets

# ── Step 080  sync_target_network_periodically ──
import numpy as np

def sync_target_network_periodically(online_params, target_params, step_count, sync_every_k):
    """Copy online -> target every sync_every_k steps; otherwise leave target unchanged."""
    if step_count > 0 and step_count % sync_every_k == 0:
        return build_target_network_copy(online_params)

    return target_params

# ── Step 081  dqn_select_action ──
def dqn_select_action(online_params, state, legal_mask, epsilon, rng):
    """Epsilon-greedy action index over the legal moves."""
    if rng.random() < epsilon:
        legal_indices = np.where(legal_mask)[0]
        idx = rng.integers(0, len(legal_indices))
        return int(legal_indices[idx])
    else:
        q_values, _ = mlp_forward_pass(online_params, state.reshape(1, -1))
        q_values = q_values.flatten()
        masked_q = mask_illegal_actions_neg_inf(q_values, legal_mask)
        return int(argmax_action_from_q_values(masked_q))

# ── Step 082  dqn_train_step ──
def dqn_train_step(online_params, target_params, adam_state, buffer, batch_size, gamma, lr, rng):
    """Run one DQN minibatch update. Return (online_params, adam_state, loss)."""
    batch = sample_minibatch_from_buffer(buffer, batch_size, rng)
    targets = compute_target_q_with_target_network(target_params, batch, gamma)
    q_values, cache = mlp_forward_pass(online_params, batch['states'])
    loss = mse_loss_on_chosen_action(q_values, batch['actions'], targets)
    grads = mlp_backward_pass(online_params, cache, batch['actions'], targets)
    new_online_params, new_adam_state = adam_update_step(online_params, grads, adam_state, lr)
    return new_online_params, new_adam_state, loss

# ── Step 083  train_dqn_agent ──
def train_dqn_agent(num_episodes, hidden_dim=64, gamma=0.99, lr=1e-3, batch_size=64, buffer_capacity=10000, sync_every_k=200, epsilon_start=1.0, epsilon_end=0.05, seed=0):
    """Full DQN self-play training loop. Returns dict with online_params,
    target_params, loss_history, reward_history, architecture."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    arch = build_mlp_architecture(9, hidden_dim, 9)
    online_params = initialize_mlp_parameters(arch, seed=seed)
    target_params = build_target_network_copy(online_params)
    adam_state = {}
    buffer = create_replay_buffer(buffer_capacity)

    loss_history = []
    reward_history = []

    eps = epsilon_start

    for episode in range(num_episodes):
        eps = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (episode / max(1, num_episodes)))

        game = TicTacToeGame()
        agent_player = 1

        episode_reward = 0.0
        step_count = 0
        episode_loss_sum = 0.0
        episode_loss_count = 0

        while not game.is_terminal():
            board = game.board
            current_player = game.current_player

            state = encode_board_flat_length_nine(board, current_player)
            legal_mask = np.zeros(9, dtype=bool)
            for row, col in get_legal_moves(board):
                legal_mask[row * 3 + col] = True

            action = dqn_select_action(online_params, state, legal_mask, eps, rng)

            out = episode_apply_action(board, action, current_player, current_player)
            next_board = out['next_board']
            status = out['status']
            reward = float(out['reward'])
            done = out['done']
            next_player = out['next_player'] if not done else current_player

            next_state = encode_board_flat_length_nine(next_board, next_player)
            next_legal_mask = np.zeros(9, dtype=bool)
            for row, col in get_legal_moves(next_board):
                next_legal_mask[row * 3 + col] = True

            append_transition_to_buffer(buffer, state, action, reward, next_state, done, next_legal_mask)

            episode_reward += reward

            game.board = next_board
            game.current_player = next_player if not done else current_player
            game.status = status

            step_count += 1

            if len(buffer['data']) >= batch_size:
                online_params, adam_state, loss = dqn_train_step(
                    online_params, target_params, adam_state, buffer, batch_size, gamma, lr, rng
                )
                episode_loss_sum += loss
                episode_loss_count += 1

            target_params = sync_target_network_periodically(
                online_params, target_params, step_count, sync_every_k
            )

        reward_history.append(episode_reward)
        episode_loss = episode_loss_sum / episode_loss_count if episode_loss_count > 0 else 0.0
        loss_history.append(episode_loss)

    return {
        'online_params': online_params,
        'target_params': target_params,
        'loss_history': loss_history,
        'reward_history': reward_history,
        'architecture': arch
    }

# ── Step 084  compare_dqn_tabular_random_minimax ──
def compare_dqn_tabular_random_minimax(dqn_artifacts, q_table, num_games=1, seed=42):
    """Round-robin evaluation among DQN, tabular Q, random, and minimax agents."""
    rng = np.random.default_rng(seed)

    online_params = dqn_artifacts['online_params']

    def dqn_move(board, player):
        state = encode_board_flat_length_nine(board, player)
        legal_mask = np.zeros(9, dtype=bool)
        for row, col in get_legal_moves(board):
            legal_mask[row * 3 + col] = True
        return dqn_select_action(online_params, state, legal_mask, 0.0, rng)

    def tabular_move(board, player):
        state_key = canonical_board_key(board)
        legal_actions = [row * 3 + col for row, col in get_legal_moves(board)]
        return greedy_argmax_over_legal_actions(q_table, state_key, legal_actions, rng)

    def random_move(board, player):
        row, col = random_move_agent(board, player, rng)
        return row * 3 + col

    def minimax_move(board, player):
        _, move = minimax_alpha_beta(board, player, -float('inf'), float('inf'))
        row, col = move
        return row * 3 + col

    agents = {
        'dqn': dqn_move,
        'tabular': tabular_move,
        'random': random_move,
        'minimax': minimax_move,
    }

    matchups = [
        ('dqn', 'random'),
        ('dqn', 'minimax'),
        ('dqn', 'tabular'),
        ('tabular', 'random'),
        ('tabular', 'minimax'),
        ('random', 'minimax'),
    ]

    def play_matchup(agent_a, agent_b):
        wins = 0
        draws = 0
        losses = 0

        for game_idx in range(num_games):
            game = TicTacToeGame()
            # Alternate which agent plays first as X.
            if game_idx % 2 == 0:
                x_agent, o_agent = agent_a, agent_b
                a_is_x = True
            else:
                x_agent, o_agent = agent_b, agent_a
                a_is_x = False

            while not game.is_terminal():
                board = game.board
                player = game.current_player
                if player == 1:
                    action = x_agent(board, player)
                else:
                    action = o_agent(board, player)
                row, col = action // 3, action % 3
                game.step(row, col)

            status = game.status
            if status == 'X_win':
                a_won = a_is_x
                decisive = True
            elif status == 'O_win':
                a_won = not a_is_x
                decisive = True
            else:
                decisive = False

            if not decisive:
                draws += 1
            elif a_won:
                wins += 1
            else:
                losses += 1

        n = num_games if num_games > 0 else 1
        return {
            'wins': wins / n,
            'draws': draws / n,
            'losses': losses / n,
        }

    results = {}
    for agent_a, agent_b in matchups:
        key = f'{agent_a}_vs_{agent_b}'
        results[key] = play_matchup(agents[agent_a], agents[agent_b])

    return results

# ── Step 085  sarsa_on_policy_update ──
def sarsa_on_policy_update(q_table, state_key, action, reward, next_state_key, next_action, done, alpha, gamma):
    """Apply one on-policy SARSA update and return the updated q_table."""
    if done:
        target = reward
    else:
        next_q = get_q_value(q_table, next_state_key, next_action)
        target = reward + gamma * next_q

    current_q = get_q_value(q_table, state_key, action)
    new_q = current_q + alpha * (target - current_q)
    set_q_value(q_table, state_key, action, new_q)
    
    return q_table

# ── Step 086  train_sarsa_agent ──
def train_sarsa_agent(num_episodes, alpha, gamma, initial_epsilon, min_epsilon, decay_rate, opponent_policy, rng):
    q_table = initialize_q_table()
    episode_outcomes = []
    agent_player = 1

    for episode in range(num_episodes):
        epsilon = epsilon_decay_schedule(initial_epsilon, episode, min_epsilon, decay_rate)
        
        game = TicTacToeGame()
        prev_state_key = None
        prev_action = None
        prev_reward = None

        while not game.is_terminal():
            board = game.board
            current_player = game.current_player

            if current_player == agent_player:
                state_key, action = episode_agent_pick_action(q_table, board, current_player, epsilon, rng)

                if prev_state_key is not None:
                    done = False
                    next_state_key = state_key
                    q_table = sarsa_on_policy_update(
                        q_table, prev_state_key, prev_action, prev_reward,
                        next_state_key, action, False, alpha, gamma
                    )

                out = episode_apply_action(board, action, current_player, agent_player)

                next_board = out['next_board']
                done = out['done']
                reward = out['reward']

                game.board = next_board
                game.current_player = out['next_player']
                game.status = out['status']

                if done:
                    if prev_state_key is not None:
                        q_table = sarsa_on_policy_update(
                            q_table, prev_state_key, prev_action, prev_reward,
                            state_key, action, True, alpha, gamma
                        )

                    q_table = sarsa_on_policy_update(
                        q_table, state_key, action, reward,
                        None, None, True, alpha, gamma
                    )
                    break

                prev_state_key = state_key
                prev_action = action
                prev_reward = reward
            
            else:
                legal_moves = get_legal_moves(board)
                action = opponent_policy(board, current_player, rng)
                out = episode_apply_action(board, action, current_player, agent_player)

                next_board = out['next_board']
                done = out['done']
                reward = out['reward']

                if prev_state_key is not None and not done:
                    q_table = sarsa_on_policy_update(
                        q_table, prev_state_key, prev_action, prev_reward,
                        canonical_board_key(board), action, False, alpha, gamma
                    )
                    prev_state_key = None
                    prev_action = None
                    prev_reward = None

                game.board = next_board
                game.current_player = out['next_player']
                game.status = out['status']

        episode_outcomes.append(game.status)

    return {'q_table': q_table, 'episode_outcomes': episode_outcomes}

# ── Step 087  reinforce_log_prob_of_action ──
import numpy as np

def reinforce_log_prob_of_action(logits, legal_action_mask, action):
    """Return (log_prob_of_action, full_prob_vector) under a softmax policy with illegal cells masked out."""
    masked_logits = mask_illegal_actions_neg_inf(logits, legal_action_mask)
    max_logit = np.max(masked_logits)
    exp_logits = np.exp(masked_logits - max_logit)
    probs = exp_logits / np.sum(exp_logits)
    log_prob = np.log(probs[action] + 1e-12)
    return log_prob, probs

# ── Step 088  reinforce_collect_episode_returns ──
import numpy as np

def reinforce_collect_episode_returns(rewards, gamma):
    """Return discounted returns G_t for a REINFORCE episode as a numpy array of shape (T,)."""
    T = len(rewards)
    if T == 0:
        return np.array([])

    returns = np.zeros(T)
    running = 0.0
    for t in range(T - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running

    return returns

# ── Step 089  reinforce_policy_gradient_update ──
def reinforce_policy_gradient_update(params, episode_cache, returns, adam_state, learning_rate=1e-2):
    states = episode_cache['states']
    actions = episode_cache['actions']
    legal_masks = episode_cache['legal_masks']
    T = len(returns)
    
    grad_accum = {key: np.zeros_like(params[key]) for key in params}
    
    for t in range(T):
        state = states[t:t+1]
        action = actions[t]
        legal_mask = legal_masks[t].astype(bool)
        G_t = returns[t]
        
        q_values, cache = mlp_forward_pass(params, state)
        
        masked_q = mask_illegal_actions_neg_inf(q_values.flatten(), legal_mask)
        
        max_logit = np.max(masked_q)
        exp_logits = np.exp(masked_q - max_logit)
        probs = exp_logits / np.sum(exp_logits)
        
        dlog = np.zeros_like(probs)
        dlog[action] = 1.0
        dlog -= probs
        
        grad_q = G_t * dlog.reshape(1, -1)
        
        x, z1, h1, q = cache['x'], cache['z1'], cache['h1'], cache['q']
        W2 = params['W2']
        
        dq = grad_q
        
        db2 = np.sum(dq, axis=0)
        dW2 = h1.T @ dq
        
        dh1 = dq @ W2.T
        dz1 = dh1 * (z1 > 0)
        
        db1 = np.sum(dz1, axis=0)
        dW1 = x.T @ dz1
        
        grad_accum['W1'] += dW1
        grad_accum['b1'] += db1
        grad_accum['W2'] += dW2
        grad_accum['b2'] += db2
    
    grad_neg = {key: -grad_accum[key] for key in grad_accum}
    new_params, new_adam_state = adam_update_step(
        params, grad_neg, adam_state, learning_rate
    )
    
    return new_params, new_adam_state

# ── Step 090  train_reinforce_agent ──
def train_reinforce_agent(num_episodes, gamma, learning_rate, hidden_dim, opponent_policy, rng, init_seed=0):
    arch = build_mlp_architecture(9, hidden_dim, 9)
    params = initialize_mlp_parameters(arch, seed=init_seed)
    adam_state = {}
    episode_outcomes = []
    agent_player = 1
    
    for episode in range(num_episodes):
        game = TicTacToeGame()
        states = []
        actions = []
        legal_masks = []
        rewards = []
        
        while not game.is_terminal():
            board = game.board
            current_player = game.current_player
            
            if current_player == agent_player:
                state = encode_board_flat_length_nine(board, current_player)
                legal_mask = np.zeros(9, dtype=bool)
                for row, col in get_legal_moves(board):
                    legal_mask[row * 3 + col] = True
                
                q_values, _ = mlp_forward_pass(params, state.reshape(1, -1))
                masked_q = mask_illegal_actions_neg_inf(q_values.flatten(), legal_mask)
                
                max_logit = np.max(masked_q)
                exp_logits = np.exp(masked_q - max_logit)
                probs = exp_logits / np.sum(exp_logits)
                
                action = rng.choice(9, p=probs)
                
                states.append(state)
                actions.append(action)
                legal_masks.append(legal_mask.astype(bool))
                
                out = episode_apply_action(board, action, current_player, agent_player)
                reward = out['reward']
                rewards.append(reward)
                
                game.board = out['next_board']
                game.current_player = out['next_player']
                game.status = out['status']
            else:
                move = opponent_policy(board, current_player, rng)
                if isinstance(move, tuple):
                    row, col = move
                else:
                    row = move // 3
                    col = move % 3
                out = episode_apply_action(board, row * 3 + col, current_player, agent_player)
                
                game.board = out['next_board']
                game.current_player = out['next_player']
                game.status = out['status']
        
        if len(rewards) > 0:
            returns = reinforce_collect_episode_returns(rewards, gamma)
            episode_cache = {
                'states': np.array(states),
                'actions': np.array(actions),
                'legal_masks': np.array(legal_masks)
            }
            params, adam_state = reinforce_policy_gradient_update(
                params, episode_cache, returns, adam_state, learning_rate
            )
        
        episode_outcomes.append(game.status)
    
    return {
        'params': params,
        'architecture': arch,
        'episode_outcomes': episode_outcomes
    }

# ── Step 091  compare_value_vs_policy_learners ──
def compare_value_vs_policy_learners(num_episodes=10, eval_games=2, seed=0):
    """Train Q-learning, SARSA, REINFORCE under matched settings; return per-agent dicts."""
    num_episodes = int(num_episodes)
    eval_games = int(eval_games)

    rng = np.random.default_rng(seed)
    result = {}

    alpha = 0.1
    gamma = 0.9
    epsilon_start = 1.0
    epsilon_min = 0.05
    decay_rate = 0.001
    hidden_dim = 32

    def random_opponent_flat(board, player, rng):
        row, col = random_move_agent(board, player, rng)
        return row * 3 + col

    rng_q = np.random.default_rng(seed + 1)
    q_res = train_q_learning_agent(
        num_episodes, alpha, gamma, epsilon_start, epsilon_min, decay_rate,
        random_opponent_flat, rng_q
    )
    q_table = q_res['q_table']

    rng_eval = np.random.default_rng(seed + 100)
    q_vs_random = evaluate_q_agent_vs_random(q_table, eval_games, rng_eval)
    q_vs_minimax = evaluate_q_agent_vs_minimax(q_table, eval_games, rng_eval)

    rng_s = np.random.default_rng(seed + 2)
    sarsa_res = train_sarsa_agent(
        num_episodes, alpha, gamma, epsilon_start, epsilon_min, decay_rate,
        random_opponent_flat, rng_s
    )
    sarsa_table = sarsa_res['q_table']

    sarsa_vs_random = evaluate_q_agent_vs_random(sarsa_table, eval_games, rng_eval)
    sarsa_vs_minimax = evaluate_q_agent_vs_minimax(sarsa_table, eval_games, rng_eval)

    rng_r = np.random.default_rng(seed + 3)
    reinforce_res = train_reinforce_agent(
        num_episodes, gamma, 1e-2, hidden_dim,
        random_opponent_flat, rng_r, init_seed=seed + 4
    )
    reinforce_params = reinforce_res['params']
    reinforce_outcomes = reinforce_res['episode_outcomes']

    reinforce_scores = []
    for outcome in reinforce_outcomes:
        if outcome == 'X_win':
            reinforce_scores.append(1.0)
        elif outcome == 'O_win':
            reinforce_scores.append(-1.0)
        else:
            reinforce_scores.append(0.0)

    def reinforce_greedy_agent(board, player, rng):
        state = encode_board_flat_length_nine(board, player)
        legal_mask = np.zeros(9, dtype=bool)
        for row, col in get_legal_moves(board):
            legal_mask[row * 3 + col] = True
        q_values, _ = mlp_forward_pass(reinforce_params, state.reshape(1, -1))
        masked_q = mask_illegal_actions_neg_inf(q_values.flatten(), legal_mask)
        return int(np.argmax(masked_q))

    reinforce_wins = 0
    reinforce_losses = 0
    reinforce_draws = 0

    for game_idx in range(eval_games):
        game = TicTacToeGame()
        agent_player = 1 if game_idx % 2 == 0 else -1

        while not game.is_terminal():
            board = game.board
            current_player = game.current_player

            if current_player == agent_player:
                action = reinforce_greedy_agent(board, current_player, rng_eval)
                out = episode_apply_action(board, action, current_player, agent_player)
            else:
                move = random_move_agent(board, current_player, rng_eval)
                if isinstance(move, tuple):
                    row, col = move
                else:
                    row = move // 3
                    col = move % 3
                out = episode_apply_action(board, row * 3 + col, current_player, agent_player)

            game.board = out['next_board']
            game.current_player = out['next_player']
            game.status = out['status']

        status = game.status
        if status == 'X_win':
            if agent_player == 1:
                reinforce_wins += 1
            else:
                reinforce_losses += 1
        elif status == 'O_win':
            if agent_player == -1:
                reinforce_wins += 1
            else:
                reinforce_losses += 1
        else:
            reinforce_draws += 1

    reinforce_vs_random = {
        'win_rate': reinforce_wins / eval_games if eval_games > 0 else 0.0,
        'loss_rate': reinforce_losses / eval_games if eval_games > 0 else 0.0,
        'draw_rate': reinforce_draws / eval_games if eval_games > 0 else 0.0
    }

    reinforce_wins_mm = 0
    reinforce_losses_mm = 0
    reinforce_draws_mm = 0

    for game_idx in range(eval_games):
        game = TicTacToeGame()
        agent_player = 1 if game_idx % 2 == 0 else -1

        while not game.is_terminal():
            board = game.board
            current_player = game.current_player

            if current_player == agent_player:
                action = reinforce_greedy_agent(board, current_player, rng_eval)
                out = episode_apply_action(board, action, current_player, agent_player)
            else:
                _, move = minimax_alpha_beta(board, current_player, -float('inf'), float('inf'))
                row, col = move
                out = episode_apply_action(board, row * 3 + col, current_player, agent_player)

            game.board = out['next_board']
            game.current_player = out['next_player']
            game.status = out['status']

        status = game.status
        if status == 'X_win':
            if agent_player == 1:
                reinforce_wins_mm += 1
            else:
                reinforce_losses_mm += 1
        elif status == 'O_win':
            if agent_player == -1:
                reinforce_wins_mm += 1
            else:
                reinforce_losses_mm += 1
        else:
            reinforce_draws_mm += 1

    reinforce_vs_minimax = {
        'win_rate': reinforce_wins_mm / eval_games if eval_games > 0 else 0.0,
        'loss_rate': reinforce_losses_mm / eval_games if eval_games > 0 else 0.0,
        'draw_rate': reinforce_draws_mm / eval_games if eval_games > 0 else 0.0
    }

    q_outcomes = q_res['episode_outcomes']
    q_scores = []
    for outcome in q_outcomes:
        if outcome == 'X_win':
            q_scores.append(1.0)
        elif outcome == 'O_win':
            q_scores.append(-1.0)
        else:
            q_scores.append(0.0)

    sarsa_outcomes = sarsa_res['episode_outcomes']
    sarsa_scores = []
    for outcome in sarsa_outcomes:
        if outcome == 'X_win':
            sarsa_scores.append(1.0)
        elif outcome == 'O_win':
            sarsa_scores.append(-1.0)
        else:
            sarsa_scores.append(0.0)

    result['q_learning'] = {
        'win_rate_vs_random': q_vs_random['win_rate'],
        'draw_rate_vs_minimax': q_vs_minimax['draw_rate'],
        'learning_curve': q_scores
    }

    result['sarsa'] = {
        'win_rate_vs_random': sarsa_vs_random['win_rate'],
        'draw_rate_vs_minimax': sarsa_vs_minimax['draw_rate'],
        'learning_curve': sarsa_scores
    }

    result['reinforce'] = {
        'win_rate_vs_random': reinforce_vs_random['win_rate'],
        'draw_rate_vs_minimax': reinforce_vs_minimax['draw_rate'],
        'learning_curve': reinforce_scores
    }

    return result

# ── Step 092  symmetry_augmented_training ──
import numpy as np

def symmetry_augmented_training(q_table, state_board, action, reward, next_state_board, done, alpha, gamma):
    """Apply Q-learning updates to all 8 D4 symmetries of a transition."""
    row, col = action // 3, action % 3

    def identity(r, c):
        return r, c

    def rot90(r, c):
        return 2 - c, r

    def compose(f, g):
        return lambda r, c: f(*g(r, c))

    rotations = [identity]
    for _ in range(3):
        rotations.append(compose(rot90, rotations[-1]))

    def fliplr(r, c):
        return r, 2 - c

    coord_transforms = list(rotations) + [compose(rot, fliplr) for rot in rotations]

    def apply_board_transforms(b):
        rots = [b]
        for _ in range(3):
            rots.append(np.rot90(rots[-1]))
        flipped = np.fliplr(b)
        frots = [flipped]
        for _ in range(3):
            frots.append(np.rot90(frots[-1]))
        return rots + frots

    state_variants = apply_board_transforms(state_board)
    next_variants = apply_board_transforms(next_state_board)

    for i, coord_fn in enumerate(coord_transforms):
        s_board = state_variants[i]
        ns_board = next_variants[i]
        new_row, new_col = coord_fn(row, col)
        new_action = new_row * 3 + new_col

        state_key = encode_board_state_key(s_board)

        if done:
            target = q_learning_terminal_target(reward)
        else:
            next_state_key = encode_board_state_key(ns_board)
            next_legal_actions = [r * 3 + c for r, c in get_legal_moves(ns_board)]
            target = q_learning_nonterminal_target(
                reward, gamma, q_table, next_state_key, next_legal_actions
            )

        old_q = get_q_value(q_table, state_key, new_action)
        new_q = old_q + alpha * (target - old_q)
        set_q_value(q_table, state_key, new_action, new_q)

    return q_table

# ── Scaffold (runner) ──
"""Scaffold for the Tic-Tac-Toe RL lab: from minimax to DQN.

Imports every helper the student implements in solution.py and runs a
minimal end-to-end demo (board ops -> minimax -> tabular Q -> DQN).
"""


import numpy as np

from solution import (
    create_empty_board,
    encode_player,
    print_board,
    is_cell_empty,
    place_move,
    get_legal_moves,
    check_row_win,
    check_column_win,
    check_main_diagonal_win,
    check_anti_diagonal_win,
    is_winner,
    is_draw,
    get_game_status,
    get_current_player,
    switch_player,
    play_hardcoded_game,
    play_interactive_game,
    TicTacToeGame,
    random_move_agent,
    play_random_vs_random_game,
    play_random_vs_random_matches,
    compute_outcome_rates,
    minimax_terminal_score,
    minimax_value,
    minimax_recursive,
    minimax_max_min_step,
    minimax_best_move,
    minimax_alpha_beta,
    play_minimax_vs_random_matches,
    play_minimax_vs_minimax_matches,
    encode_board_state_key,
    canonical_board_key,
    initialize_q_table,
    get_q_value,
    set_q_value,
    choose_learning_rate_alpha,
    choose_discount_factor_gamma,
    choose_initial_epsilon,
    epsilon_decay_schedule,
    epsilon_greedy_explore_move,
    epsilon_greedy_select_action,
    greedy_argmax_over_legal_actions,
    random_tie_break_argmax,
    tic_tac_toe_reward,
    q_learning_nonterminal_target,
    q_learning_terminal_target,
    q_learning_update,
    episode_reset_game,
    episode_agent_pick_action,
    episode_apply_action,
    episode_apply_q_update,
    episode_check_terminate,
    train_q_learning_agent,
    compute_batched_outcome_stats,
    self_play_episode,
    flip_board_perspective,
    perspective_reward_sign,
    train_q_agent_self_play,
    evaluate_q_agent_vs_random,
    evaluate_q_agent_vs_minimax,
    inspect_q_values_for_state,
    serialize_q_table_to_dict,
    deserialize_q_table_from_dict,
    encode_board_flat_length_nine,
    encode_board_one_hot_length_eighteen,
    build_mlp_architecture,
    initialize_mlp_parameters,
    mlp_forward_pass,
    mask_illegal_actions_neg_inf,
    argmax_action_from_q_values,
    mse_loss_on_chosen_action,
    mlp_backward_pass,
    adam_update_step,
    create_replay_buffer,
    append_transition_to_buffer,
    cap_buffer_size_drop_oldest,
    sample_minibatch_from_buffer,
    build_target_network_copy,
    compute_target_q_with_target_network,
    sync_target_network_periodically,
    dqn_select_action,
    dqn_train_step,
    train_dqn_agent,
    compare_dqn_tabular_random_minimax,
    sarsa_on_policy_update,
    train_sarsa_agent,
    reinforce_log_prob_of_action,
    reinforce_collect_episode_returns,
    reinforce_policy_gradient_update,
    train_reinforce_agent,
    compare_value_vs_policy_learners,
    symmetry_augmented_training,
)


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.default_rng(0)

    # 1) Game engine sanity check.
    board = create_empty_board()
    print("Empty board:")
    print_board(board)
    board = place_move(board, 1, 1, encode_player("X"))
    board = place_move(board, 0, 0, encode_player("O"))
    print("After two moves:")
    print_board(board)
    print("Status:", get_game_status(board))
    print("Legal moves remaining:", len(get_legal_moves(board)))

    # 2) Random vs random baseline.
    outcomes = play_random_vs_random_matches(200, rng)
    print("Random-vs-random rates:", compute_outcome_rates(outcomes))

    # 3) Minimax must never lose to a random opponent.
    mm_outcomes = play_minimax_vs_random_matches(20, minimax_plays_x=True, rng=rng)
    print("Minimax(X) vs Random rates:", compute_outcome_rates(mm_outcomes))

    # 4) Train a tabular Q-learning agent against random play.
    q_table = train_q_learning_agent(
        num_episodes=2000,
        alpha=choose_learning_rate_alpha(),
        gamma=choose_discount_factor_gamma(),
        initial_epsilon=choose_initial_epsilon(),
        min_epsilon=0.05,
        decay_rate=1e-3,
        opponent_policy=random_move_agent,
        rng=rng,
    )
    q_vs_random = evaluate_q_agent_vs_random(q_table, 100, rng)
    print("Tabular Q vs Random:", q_vs_random)

    # 5) Quick DQN smoke test (few episodes just to verify the loop runs).
    dqn_artifacts = train_dqn_agent(
        num_episodes=50,
        hidden_dim=32,
        gamma=0.95,
        lr=1e-3,
        batch_size=16,
        buffer_capacity=500,
        sync_every_k=50,
        epsilon_start=1.0,
        epsilon_end=0.1,
        seed=0,
    )
    print("DQN training artifacts keys:", list(dqn_artifacts.keys()) if isinstance(dqn_artifacts, dict) else type(dqn_artifacts).__name__)

    # 6) DQN forward-pass demonstration on a fresh board.
    demo_board = create_empty_board()
    x_vec = encode_board_flat_length_nine(demo_board, encode_player("X"))
    arch = build_mlp_architecture(input_dim=9, hidden_dim=32, output_dim=9)
    params = initialize_mlp_parameters(arch, seed=0)
    q_values, _ = mlp_forward_pass(params, x_vec.reshape(1, -1))
    legal_mask = np.ones(9, dtype=bool)
    masked = mask_illegal_actions_neg_inf(q_values, legal_mask)
    action = argmax_action_from_q_values(masked)
    print("Untrained MLP picks action index:", int(np.asarray(action).ravel()[0]))

    print("Scaffold demo complete.")
