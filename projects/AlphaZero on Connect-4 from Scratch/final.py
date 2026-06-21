"""
AlphaZero on Connect-4 from Scratch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  make_empty_board ──
import numpy as np

def make_empty_board():
    """Return a 6x7 integer numpy array of zeros representing an empty Connect-4 board."""
    return np.zeros((6, 7), dtype=int)

# ── Step 002  column_top_row ──
def column_top_row(board, column):
    """Return the lowest empty row in `column`, or -1 if the column is full."""
    for row in range(5, -1, -1):
        if board[row, column] == 0:
            return row
    
    return -1

# ── Step 003  drop_piece ──
def drop_piece(board, column, player):
    new_board = board.copy()
    row = column_top_row(new_board, column)
    if row == -1:
        raise ValueError('')

    new_board[row, column] = player
    return new_board

# ── Step 004  column_full ──
import numpy as np

def column_full(board, column):
    """Return True if `column` has no empty rows left."""
    return True if column_top_row(board, column) == -1 else False

# ── Step 005  valid_moves ──
def valid_moves(board):
    columns = board.shape[1]
    valid = [column for column in range(columns) if not column_full(board, column)]
    return valid

# ── Step 006  four_in_a_row_horizontal ──
def four_in_a_row_horizontal(board):
    for row in range(6):
        for col in range(4):
            if board[row, col]!= 0:
                if board[row, col] == board[row, col + 1] == board[row, col + 2] == board[row, col + 3]:
                    return int(board[row, col])

    return 0

# ── Step 007  four_in_a_row_vertical ──
def four_in_a_row_vertical(board):
    for col in range(7):
        for row in range(3):
            if board[row, col] != 0:
                if board[row, col] == board[row + 1, col] == board[row + 2, col] == board[row + 3, col]:
                    return int(board[row, col])

    return 0

# ── Step 008  four_in_a_row_diagonal_down_right ──
def four_in_a_row_diagonal_down_right(board):
    for row in range(3):
        for col in range(4):
            if board[row, col] != 0:
                if board[row, col] == board[row + 1, col + 1] == board[row + 2, col + 2] == board[row + 3, col + 3]:
                    return int(board[row, col])

    return 0

# ── Step 009  four_in_a_row_diagonal_up_right ──
def four_in_a_row_diagonal_up_right(board):
    for row in range(3, 6):
        for col in range(4):
            if board[row, col] != 0:
                if board[row, col] == board[row-1, col+1] == board[row-2, col+2] == board[row-3, col+3]:
                    return int(board[row, col])
    return 0

# ── Step 010  check_winner ──
import numpy as np

def check_winner(board):
    """Return 1 or 2 if that player has four in a row, else 0."""
    winner = four_in_a_row_horizontal(board)
    if winner != 0:
        return winner

    winner = four_in_a_row_vertical(board)
    if winner != 0:
        return winner

    winner = four_in_a_row_diagonal_down_right(board)
    if winner != 0:
        return winner

    winner = four_in_a_row_diagonal_up_right(board)
    if winner != 0:
        return winner

    return 0

# ── Step 011  board_is_full ──
def board_is_full(board):
    for col in range(board.shape[1]):
        if board[0, col] == 0:
            return False
            
    return True

# ── Step 012  is_terminal ──
def is_terminal(board):
    winner = check_winner(board)
    if winner != 0:
        return True, winner

    if board_is_full(board):
        return True, 0

    return False, 0

# ── Step 013  other_player ──
def other_player(player):
    return 3 - player

# ── Step 014  step_env ──
def step_env(board, column, player):
    row = column_top_row(board, column)
    if row == -1:
        return board, False, 0, player
    
    new_board = board.copy()
    new_board[row, column] = player
    
    existing_winner = check_winner(board)
    
    done, winner = is_terminal(new_board)
    
    if done:
        if existing_winner != 0:
            next_player = existing_winner
        else:
            if winner != 0:
                next_player = 3 - player
            else:
                next_player = 0
    else:
        next_player = 3 - player
    
    return new_board, done, winner, next_player

# ── Step 015  encode_board ──
def encode_board(board, current_player):
    """Encode a 6x7 board as a (2, 6, 7) float32 tensor from current_player's view."""
    opponent = other_player(current_player)
    tensor = np.zeros((2, 6, 7), dtype=np.float32)
    tensor[0] = (board == current_player).astype(np.float32)
    tensor[1] = (board == opponent).astype(np.float32)
    return tensor

# ── Step 016  board_to_torch_tensor ──
def board_to_torch_tensor(board, current_player):
    encoded = encode_board(board, current_player)
    return torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)

# ── Step 017  init_conv_backbone ──
import torch
import torch.nn as nn

def init_conv_backbone(in_channels=2, hidden_channels=16):
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )

# ── Step 018  init_policy_head ──
import torch
import torch.nn as nn

def init_policy_head(hidden_channels=16, num_columns=7):
    """Return an nn.Module mapping (B, hidden_channels, 6, 7) -> (B, num_columns) logits."""
    return nn.Sequential(
        nn.Conv2d(hidden_channels, 1, kernel_size=1),
        nn.Flatten(),
        nn.Linear(6 * 7, num_columns),
    )

# ── Step 019  init_value_head ──
import torch
import torch.nn as nn

def init_value_head(hidden_channels=16):
    """Return an nn.Module mapping (B, hidden_channels, 6, 7) -> (B, 1) in (-1, 1)."""
    return nn.Sequential(
        nn.Conv2d(hidden_channels, 1, kernel_size=1),
        nn.Flatten(),
        nn.Linear(6 * 7, 1),
        nn.Tanh(),
    )

# ── Step 020  build_policy_value_net ──
import torch
import torch.nn as nn

def build_policy_value_net(in_channels=2, hidden_channels=16, num_columns=7):
    """Compose backbone + policy head + value head into one nn.Module."""
    class PolicyValueNet(nn.Module):
        def __init__(self, in_channels=2, hidden_channels=16, num_columns=7):
            super().__init__()
            self.backbone = init_conv_backbone(in_channels, hidden_channels)
            self.policy_head = init_policy_head(hidden_channels, num_columns)
            self.value_head = init_value_head(hidden_channels)

        def forward(self, x):
            features = self.backbone(x)
            logits = self.policy_head(features)
            value = self.value_head(features)
            return logits, value

    return PolicyValueNet(in_channels, hidden_channels, num_columns)

# ── Step 021  policy_value_forward ──
import torch
import torch.nn as nn

def policy_value_forward(net, encoded_board):
    """Run encoded_board (B,2,6,7) through net and return (logits, value)."""
    logits, value = net(encoded_board)
    return logits, value

# ── Step 022  action_mask ──
import numpy as np

def action_mask(board):
    mask = np.zeros(7, dtype=bool)
    valid = valid_moves(board)
    for col in valid:
        mask[col] = True
        
    return mask

# ── Step 023  masked_policy_logits ──
import torch

def masked_policy_logits(logits, mask):
    """Set logits at illegal columns to -inf.

    logits: torch.Tensor of shape (..., 7)
    mask:   bool array/tensor of shape (7,), True = legal
    returns: torch.Tensor of same shape as logits
    """
    if isinstance(mask, np.ndarray):
        mask = torch.tensor(mask, dtype=torch.bool)

    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)

    result = logits.clone()

    if result.dim() == 2 and mask.dim() == 1:
        mask = mask.unsqueeze(0).expand(result.shape[0], -1)
    
    result[~mask] = float('-inf')
    return result

# ── Step 024  masked_log_softmax ──
import torch
import torch.nn.functional as F

def masked_log_softmax(logits, mask):
    """Log-softmax of logits with illegal columns (mask=False) forced to -inf."""
    masked_logits = masked_policy_logits(logits, mask)
    return F.log_softmax(masked_logits, dim=-1)

# ── Step 025  sample_action_from_policy ──
import torch

def sample_action_from_policy(logits, mask, temperature=1.0):
    """Sample a legal column from a tempered masked categorical policy."""
    masked = masked_policy_logits(logits, mask)
    scaled = masked / temperature
    probs = torch.softmax(scaled, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample().item()
    return action

# ── Step 026  greedy_action_from_policy ──
import torch

def greedy_action_from_policy(logits, mask):
    """Return the argmax legal column index from masked policy logits."""
    masked = masked_policy_logits(logits, mask)
    return int(torch.argmax(masked).item())

# ── Step 027  make_mcts_node ──
def make_mcts_node(prior=0.0, parent=None):
    return {
        'prior': prior,
        'visit_count': 0,
        'value_sum': 0.0,
        'children': {},
        'parent': parent,
        'visits': 0,
        'is_expanded': False,
    }

# ── Step 028  node_q_value ──
def node_q_value(node):
    if node['visit_count'] == 0:
        return 0.0

    return node['value_sum'] / node['visit_count']

# ── Step 029  ucb_score ──
import math

def ucb_score(parent, child, c_puct=1.5):
    q = node_q_value(child)
    exploration = c_puct * child['prior'] * np.sqrt(parent['visit_count']) / (1 + child['visit_count'])
    return q + exploration

# ── Step 030  select_best_child ──
def select_best_child(node, legal_actions, c_puct=1.5):
    best_action = None
    best_child = None
    best_score = float('-inf')

    for action in legal_actions:
        child = node['children'][action]
        score = ucb_score(node, child, c_puct)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    return best_action, best_child

# ── Step 031  select_leaf ──
def select_leaf(root, c_puct):
    node = root
    while node.get('is_expanded', False):
        legal_actions = list(node['children'].keys())
        action, child = select_best_child(node, legal_actions, c_puct)
        node = child

    return node

# ── Step 032  evaluate_with_network ──
def evaluate_with_network(net, state, to_play):
    net.eval()
    with torch.no_grad():
        x = board_to_torch_tensor(state, to_play)
        logits, value = net(x)

        mask = action_mask(state)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)

        log_probs = masked_log_softmax(logits.squeeze(0), mask_tensor)

        priors = torch.exp(log_probs).numpy()

        return priors, float(value.squeeze().item())

# ── Step 033  expand_node ──
def expand_node(node, priors):
    board = node['board']
    to_play = node['to_play']
    legal_actions = valid_moves(board)

    node['children'] = {}
    for action in legal_actions:
        child_board = drop_piece(board, action, to_play)
        child_node = make_mcts_node(priors[action], parent=node)
        child_node['board'] = child_board
        child_node['to_play'] = other_player(to_play)
        node['children'][action] = child_node

    node['is_expanded'] = True

# ── Step 034  backup_value ──
def backup_value(leaf, value):
    node = leaf
    sign = 1.0

    while node is not None:
        node['visit_count'] += 1
        node['value_sum'] += sign * value
        sign = -sign
        node['visits'] = node['visit_count']
        node = node['parent']

# ── Step 035  run_one_simulation ──
def run_one_simulation(root, net, c_puct):
    leaf = select_leaf(root, c_puct)

    board = leaf['board']
    to_play = leaf['to_play']
    done, winner = is_terminal(board)

    if done:
        if winner == 0:
            value = 0.0
        elif winner == to_play:
            value = 1.0
        else:
            value = -1.0

        backup_value(leaf, value)

    else:
        priors, value = evaluate_with_network(net, board, to_play)
        expand_node(leaf, priors)
        backup_value(leaf, value)

# ── Step 036  run_mcts ──
def run_mcts(state, to_play, net, num_simulations, c_puct):
    root = make_mcts_node()
    root['board'] = state
    root['to_play'] = to_play

    for _ in range(num_simulations):
        run_one_simulation(root, net, c_puct)

    return root

# ── Step 037  visit_count_policy ──
def visit_count_policy(root, temperature=1.0):
    pi = np.zeros(7, dtype=np.float64)
    
    if not root['children']:
        pi[:] = 1.0 / 7.0
        return pi
    
    visits = np.zeros(7, dtype=np.float64)
    for action, child in root['children'].items():
        visits[action] = child['visit_count']
    
    visited_actions = [action for action, child in root['children'].items() if child['visit_count'] > 0]
    
    if not visited_actions:
        pi[:] = 1.0 / 7.0
        return pi
    
    if temperature == 0.0:
        best_action = np.argmax(visits)
        pi[best_action] = 1.0
    else:
        visit_counts = np.array([visits[a] for a in visited_actions])
        
        if temperature != 1.0:
            weighted = visit_counts ** (1.0 / temperature)
        else:
            weighted = visit_counts
        
        probs = weighted / np.sum(weighted)
        
        for action, prob in zip(visited_actions, probs):
            pi[action] = prob
    
    return pi

# ── Step 038  mcts_choose_action ──
def mcts_choose_action(state, to_play, net, num_simulations, c_puct, temperature=1.0):
    root = run_mcts(state, to_play, net, num_simulations, c_puct)
    policy = visit_count_policy(root, temperature)
    
    if not np.isclose(policy.sum(), 1.0):
        policy = policy / policy.sum()
    
    policy_tensor = torch.tensor(policy, dtype=torch.float32)
    dist = torch.distributions.Categorical(policy_tensor)
    action = dist.sample().item()
    
    return action, policy

# ── Step 039  record_self_play_step ──
def record_self_play_step(history, board, policy, to_play):
    history.append({
        'board': board.copy(),
        'policy': policy.copy(),
        'to_play': to_play,
    })
    return history

# ── Step 040  play_self_play_game ──
def play_self_play_game(net, num_simulations, c_puct, temperature=1.0):
    board = make_empty_board()
    to_play = 1
    history = []
    done = False
    winner = 0

    while not done:
        action, policy = mcts_choose_action(board, to_play, net, num_simulations, c_puct, temperature)
        record_self_play_step(history, board, policy, to_play)
        board, done, winner, to_play = step_env(board, action, to_play)

    return history, winner

# ── Step 041  assign_value_targets ──
def assign_value_targets(history, winner):
    labelled = []
    for step in history:
        new_step = step.copy()
        to_play = step['to_play']
        if winner == 0:
            new_step['value'] = 0.0
        elif to_play == winner:
            new_step['value'] = 1.0
        else:
            new_step['value'] = -1.0

        labelled.append(new_step)

    return labelled

# ── Step 042  generate_self_play_batch ──
def generate_self_play_batch(net, num_games, num_simulations, c_puct, temperature=1.0):
    buffer = []
    for _ in range(num_games):
        history, winner = play_self_play_game(net, num_simulations, c_puct, temperature)
        labelled = assign_value_targets(history, winner)
        buffer.extend(labelled)

    return buffer

# ── Step 043  value_loss_mse ──
import torch

def value_loss_mse(predicted_values, target_values):
    return torch.mean((predicted_values - target_values) ** 2)

# ── Step 044  policy_loss_cross_entropy ──
import torch

def policy_loss_cross_entropy(predicted_log_probs, target_policy):
    """Cross-entropy between MCTS target policy and network log-probs. Returns scalar tensor."""
    loss = -torch.sum(target_policy * predicted_log_probs, dim=1)
    return torch.mean(loss)

# ── Step 045  l2_regularization_loss ──
def l2_regularization_loss(net):
    l2_loss = torch.tensor(0.0, dtype=torch.float32)
    for param in net.parameters():
        if param.requires_grad:
            l2_loss += torch.sum(param ** 2)
    
    return l2_loss

# ── Step 046  combined_loss ──
def combined_loss(predicted_log_probs, predicted_values, target_policy, target_values, net, policy_weight=1.0, value_weight=1.0, l2_weight=1e-4):
    policy_loss = policy_loss_cross_entropy(predicted_log_probs, target_policy)
    value_loss = value_loss_mse(predicted_values, target_values)
    l2_loss = l2_regularization_loss(net)
    total_loss = policy_weight * policy_loss + value_weight * value_loss + l2_weight * l2_loss
    parts = {
        'policy': policy_loss,
        'value': value_loss,
        'l2': l2_loss,
    }
    return total_loss, parts

# ── Step 047  encode_batch_states ──
def encode_batch_states(boards, to_plays):
    encoded_boards = []
    for board, to_play in zip(boards, to_plays):
        enc = encode_board(board, to_play)
        encoded_boards.append(torch.tensor(enc, dtype=torch.float32))

    return torch.stack(encoded_boards)

# ── Step 048  iterate_minibatches ──
def iterate_minibatches(buffer, batch_size, seed=None):
    """Yield shuffled minibatches of step dicts of size <= batch_size."""
    n = len(buffer)
    
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    indices = rng.permutation(n).tolist()

    for i in range(0, n, batch_size):
        batch_indices = indices[i:i+batch_size]
        yield [buffer[idx] for idx in batch_indices]

# ── Step 049  training_step ──
def training_step(net, optimizer, minibatch, policy_weight=1.0, value_weight=1.0, l2_weight=1e-4):
    boards = [step['board'] for step in minibatch]
    to_plays = [step['to_play'] for step in minibatch]
    target_policies = torch.tensor([step['policy'] for step in minibatch], dtype=torch.float32)
    target_values = torch.tensor([step['value'] for step in minibatch], dtype=torch.float32).unsqueeze(1)

    encoded = encode_batch_states(boards, to_plays)

    logits, predicted_values = net(encoded)

    masks = [action_mask(board) for board in boards]
    mask_tensor = torch.tensor(masks, dtype=torch.bool)

    predicted_log_probs = masked_log_softmax(logits, mask_tensor)

    total_loss, parts = combined_loss(
        predicted_log_probs,
        predicted_values,
        target_policies,
        target_values,
        net,
        policy_weight,
        value_weight,
        l2_weight,
    )

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        'total': float(total_loss.item()),
        'policy': float(parts['policy'].item()),
        'value': float(parts['value'].item()),
        'l2': float(parts['l2'].item())
    }

# ── Step 050  training_epoch ──
def training_epoch(net, optimizer, buffer, batch_size, policy_weight=1.0, value_weight=1.0, l2_weight=1e-4, seed=None):
    total_losses = {'total': 0.0, 'policy': 0.0, 'value': 0.0, 'l2': 0.0}
    num_batches = 0

    for minibatch in iterate_minibatches(buffer, batch_size, seed):
        losses = training_step(net, optimizer, minibatch, policy_weight, value_weight, l2_weight)
        for key in total_losses:
            total_losses[key] += losses[key]

        num_batches += 1

    if num_batches > 0:
        for key in total_losses:
            total_losses[key] /= num_batches
    
    return total_losses

# ── Step 051  self_play_iteration ──
def self_play_iteration(net, optimizer, num_games, num_simulations, c_puct, batch_size, num_epochs=1, temperature=1.0):
    buffer = generate_self_play_batch(net, num_games, num_simulations, c_puct, temperature)
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = training_epoch(net, optimizer, buffer, batch_size, seed=epoch)
        losses.append(epoch_loss)

    return {
        'buffer_size': len(buffer),
        'losses': losses,
    }

# ── Step 052  train_loop ──
def train_loop(net, optimizer, num_iterations, num_games, num_simulations, c_puct, batch_size, num_epochs=1, temperature=1.0):
    history = []
    for _ in range(num_iterations):
        result = self_play_iteration(
            net, optimizer, num_games, num_simulations, c_puct,
            batch_size, num_epochs, temperature
        )
        history.append(result)

    return history

# ── Step 053  random_policy_action ──
def random_policy_action(state, to_play, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    legal = valid_moves(state)
    return int(rng.choice(legal))

# ── Step 054  greedy_agent_action ──
def greedy_agent_action(net, state, to_play):
    net.eval()
    with torch.no_grad():
        x = board_to_torch_tensor(state, to_play)
        logits, _ = net(x)
        logits = logits.squeeze(0)
        mask = action_mask(state)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        masked_logits = masked_policy_logits(logits, mask_tensor)
        action = int(torch.argmax(masked_logits).item())
        return action

# ── Step 055  play_one_match ──
def play_one_match(agent_one, agent_two, starting_player=1):
    board = make_empty_board()
    to_play = starting_player
    done = False
    winner = 0

    while not done:
        if to_play == 1:
            action = agent_one(board, to_play)
        else:
            action = agent_two(board, to_play)

        board, done, winner, to_play = step_env(board, action, to_play)

    return winner

# ── Step 056  match_win_rate ──
def match_win_rate(agent_one, agent_two, num_matches, alternate_starts=True):
    wins = 0
    losses = 0
    draws = 0
    
    for match_idx in range(num_matches):
        if alternate_starts:
            starting_player = 1 if match_idx % 2 == 0 else 2
        else:
            starting_player = 1
        
        winner = play_one_match(agent_one, agent_two, starting_player)
        
        if winner == 1:
            wins += 1
        elif winner == 2:
            losses += 1
        else:
            draws += 1
    
    return {'wins': wins, 'losses': losses, 'draws': draws}

# ── Step 057  evaluate_against_random ──
def evaluate_against_random(net, num_matches, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    greedy_agent = lambda state, to_play: greedy_agent_action(net, state, to_play)
    random_agent = lambda state, to_play: random_policy_action(state, to_play, rng)

    return match_win_rate(greedy_agent, random_agent, num_matches, alternate_starts=True)

# ── Scaffold (runner) ──
"""AlphaZero on Connect-4: end-to-end demo of self-play training and evaluation."""

import numpy as np
import torch

from solution import (
    make_empty_board,
    column_top_row,
    drop_piece,
    column_full,
    valid_moves,
    four_in_a_row_horizontal,
    four_in_a_row_vertical,
    four_in_a_row_diagonal_down_right,
    four_in_a_row_diagonal_up_right,
    check_winner,
    board_is_full,
    is_terminal,
    other_player,
    step_env,
    encode_board,
    board_to_torch_tensor,
    init_conv_backbone,
    init_policy_head,
    init_value_head,
    build_policy_value_net,
    policy_value_forward,
    action_mask,
    masked_policy_logits,
    masked_log_softmax,
    sample_action_from_policy,
    greedy_action_from_policy,
    make_mcts_node,
    node_q_value,
    ucb_score,
    select_best_child,
    select_leaf,
    evaluate_with_network,
    expand_node,
    backup_value,
    run_one_simulation,
    run_mcts,
    visit_count_policy,
    mcts_choose_action,
    record_self_play_step,
    play_self_play_game,
    assign_value_targets,
    generate_self_play_batch,
    value_loss_mse,
    policy_loss_cross_entropy,
    l2_regularization_loss,
    combined_loss,
    encode_batch_states,
    iterate_minibatches,
    training_step,
    training_epoch,
    self_play_iteration,
    train_loop,
    random_policy_action,
    greedy_agent_action,
    play_one_match,
    match_win_rate,
    evaluate_against_random,
)


def _summarize_loss(v):
    """Convert a loss value (scalar or list of scalars) to a single rounded float."""
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return float("nan")
        return round(float(np.mean([float(x) for x in v])), 4)
    return round(float(v), 4)


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    # --- Sanity-check the board engine ---
    board = make_empty_board()
    print("Empty board shape:", board.shape, "dtype:", board.dtype)
    board = drop_piece(board, 3, 1)
    board = drop_piece(board, 3, 2)
    print("Legal moves after two drops:", valid_moves(board))
    print("Terminal?", is_terminal(board))

    # --- Build a tiny policy-value network ---
    net = build_policy_value_net(in_channels=2, hidden_channels=8, num_columns=7)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Policy-value net params: {n_params}")

    # --- One forward pass on a fresh board ---
    enc = board_to_torch_tensor(make_empty_board(), current_player=1)
    with torch.no_grad():
        logits, value = policy_value_forward(net, enc)
    print("Policy logits shape:", tuple(logits.shape), "value shape:", tuple(value.shape))

    # --- One MCTS rollout from the empty board ---
    action, pi = mcts_choose_action(
        make_empty_board(), to_play=1, net=net,
        num_simulations=8, c_puct=1.5, temperature=1.0,
    )
    print(f"MCTS picked column {action}; pi = {np.round(pi, 3).tolist()}")

    # --- Generate a small self-play buffer ---
    buffer = generate_self_play_batch(
        net, num_games=2, num_simulations=6, c_puct=1.5, temperature=1.0,
    )
    print(f"Self-play buffer size: {len(buffer)} steps")

    # --- A couple of training steps on that buffer ---
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses_before = training_epoch(
        net, optimizer, buffer, batch_size=8,
        policy_weight=1.0, value_weight=1.0, l2_weight=1e-4, seed=0,
    )
    print("Epoch 1 losses:", {k: _summarize_loss(v) for k, v in losses_before.items()})

    # --- One AlphaZero outer iteration (self-play + training) ---
    iter_result = self_play_iteration(
        net, optimizer,
        num_games=2, num_simulations=6, c_puct=1.5,
        batch_size=8, num_epochs=1, temperature=1.0,
    )
    # self_play_iteration returns {'buffer_size': int, 'losses': [epoch_dict, ...]}
    if isinstance(iter_result, dict) and 'losses' in iter_result and isinstance(iter_result['losses'], list):
        iter_losses = iter_result['losses'][-1] if iter_result['losses'] else {}
    else:
        iter_losses = iter_result
    print("Outer-iter losses:", {k: _summarize_loss(v) for k, v in iter_losses.items()})

    # --- Evaluate the greedy net agent against a random baseline ---
    results = evaluate_against_random(net, num_matches=6, seed=0)
    print("Greedy-net vs random:", results)
