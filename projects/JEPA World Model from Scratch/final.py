"""
JEPA World Model from Scratch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  init_env_state ──
def init_env_state(room_size: int = 8, seed: int | None = None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed),

    x = torch.randint(0, room_size, (1,)).float()
    y = torch.randint(0, room_size, (1,)).float()
    return torch.cat([x, y])

# ── Step 002  apply_action ──
def apply_action(state: torch.Tensor, action: int, room_size: int = 8) -> torch.Tensor:
    x, y = state[0].item(), state[1].item()

    if action == 0:
        y -= 1
    elif action == 1:
        y += 1
    elif action == 2:
        x -= 1
    elif action == 3:
        x += 1

    x = max(0, min(room_size - 1, x))
    y = max(0, min(room_size - 1, y))
    return torch.tensor([float(x), float(y)])

# ── Step 003  render_observation ──
def render_observation(state: torch.Tensor, room_size: int = 8) -> torch.Tensor:
    x = int(state[0].item())
    y = int(state[1].item())
    obs = torch.zeros((1, room_size, room_size), dtype=torch.float32)
    obs[0, y, x] = 1.0
    return obs

# ── Step 004  env_reset ──
def env_reset(room_size: int = 8, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    state = init_env_state(room_size, seed)
    obs = render_observation(state, room_size)
    return state, obs

# ── Step 005  env_step ──
def env_step(state: torch.Tensor, action: int, room_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    next_state = apply_action(state, action, room_size)
    next_obs = render_observation(next_state, room_size)
    return next_state, next_obs

# ── Step 006  collect_random_transitions ──
def collect_random_transitions(num_transitions: int, room_size: int = 8, seed: int = 0) -> dict:
    if num_transitions == 0:
        return {
            'observations': torch.zeros((0, 1, room_size, room_size), dtype=torch.float32),
            'actions': torch.zeros((0,), dtype=torch.long),
            'next_observations': torch.zeros((0, 1, room_size, room_size), dtype=torch.float32),
            'states': torch.zeros((0, 2), dtype=torch.float32),
            'next_states': torch.zeros((0, 2), dtype=torch.float32)
        }
        
    rng = torch.Generator()
    rng.manual_seed(seed)

    observations = []
    actions = []
    next_observations = []
    states = []
    next_states = []

    state, obs = env_reset(room_size, seed)

    for _ in range(num_transitions):
        action = torch.randint(0, 4, (1,), generator=rng).item()
        next_state, next_obs = env_step(state, action, room_size)

        observations.append(obs)
        actions.append(action)
        next_observations.append(next_obs)
        states.append(state)
        next_states.append(next_state)

        state = next_state
        obs = next_obs

    return {
        'observations': torch.stack(observations),
        'actions': torch.tensor(actions, dtype=torch.long),
        'next_observations': torch.stack(next_observations),
        'states': torch.stack(states),
        'next_states': torch.stack(next_states),
    }

# ── Step 007  build_transition_dataset ──
def build_transition_dataset(num_transitions: int = 512, room_size: int = 8, seed: int = 0) -> dict:
    return collect_random_transitions(num_transitions, room_size, seed)

# ── Step 008  init_encoder_params ──
def init_encoder_params(obs_channels: int = 1, room_size: int = 8, latent_dim: int = 32, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    
    conv1_w = torch.randn(16, obs_channels, 3, 3) * 0.1
    conv1_b = torch.zeros(16)

    conv2_w = torch.randn(32, 16, 3, 3) * 0.1
    conv2_b = torch.zeros(32)

    h = ((room_size + 2 - 3) // 2) + 1
    w = ((room_size + 2 - 3) // 2) + 1

    fc_in = 32 * h * w

    fc_w = torch.randn(latent_dim, fc_in) * 0.1
    fc_b = torch.zeros(latent_dim)

    conv1_w.requires_grad_(True)
    conv1_b.requires_grad_(True)
    conv2_w.requires_grad_(True)
    conv2_b.requires_grad_(True)
    fc_w.requires_grad_(True)
    fc_b.requires_grad_(True)

    return {
        'conv1_w': conv1_w,
        'conv1_b': conv1_b,
        'conv2_w': conv2_w,
        'conv2_b': conv2_b,
        'fc_w': fc_w,
        'fc_b': fc_b,
    }

# ── Step 009  encoder_forward ──
def encoder_forward(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    conv1_w = encoder_params['conv1_w']
    conv1_b = encoder_params['conv1_b']
    conv2_w = encoder_params['conv2_w']
    conv2_b = encoder_params['conv2_b']
    fc_w = encoder_params['fc_w']
    fc_b = encoder_params['fc_b']

    x = torch.nn.functional.conv2d(obs, conv1_w, conv1_b, stride=1, padding=1)
    x = torch.relu(x)

    x = torch.nn.functional.conv2d(x, conv2_w, conv2_b, stride=2, padding=1)
    x = torch.relu(x)

    x = x.view(x.size(0), -1)
    x = x @ fc_w.T + fc_b
    return x

# ── Step 010  init_target_encoder ──
def init_target_encoder(encoder_params: dict) -> dict:
    target_params = {}
    for key, value in encoder_params.items():
        target_params[key] = value.detach().clone()

    return target_params

# ── Step 011  ema_update ──
def ema_update(target_params: dict, encoder_params: dict, tau: float = 0.99) -> dict:
    updated = {}
    for key in target_params:
        updated[key] = tau * target_params[key] + (1 - tau) * encoder_params[key]

    return updated

# ── Step 012  encode_batch ──
def encode_batch(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    return encoder_forward(obs, encoder_params)

# ── Step 013  init_predictor_params ──
def init_predictor_params(latent_dim: int = 32, action_dim: int = 4, hidden_dim: int = 64, seed: int = 0) -> dict:
    torch.manual_seed(seed)

    action_embed_w = torch.randn(action_dim, latent_dim) * 0.02
    action_embed_w.requires_grad_(True)

    fc1_w = torch.randn(hidden_dim, 2 * latent_dim) * 0.02
    fc1_w.requires_grad_(True)
    fc1_b = torch.zeros(hidden_dim)
    fc1_b.requires_grad_(True)

    fc2_w = torch.randn(latent_dim, hidden_dim) * 0.02
    fc2_w.requires_grad_(True)
    fc2_b = torch.zeros(latent_dim)
    fc2_b.requires_grad_(True)

    return {
        'action_embed_w': action_embed_w,
        'fc1_w': fc1_w,
        'fc1_b': fc1_b,
        'fc2_w': fc2_w,
        'fc2_b': fc2_b,
    }

# ── Step 014  embed_action ──
def embed_action(actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    action_embed_w = predictor_params['action_embed_w']
    return action_embed_w[actions]

# ── Step 015  predictor_forward ──
def predictor_forward(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    action_emb = embed_action(actions, predictor_params)
    x = torch.cat([embeddings, action_emb], dim=-1)
    x = x @ predictor_params['fc1_w'].T + predictor_params['fc1_b']
    x = torch.relu(x)
    x = x @ predictor_params['fc2_w'].T + predictor_params['fc2_b']    
    return x

# ── Step 016  predict_next_embedding ──
def predict_next_embedding(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    return predictor_forward(embeddings, actions, predictor_params)

# ── Step 017  prediction_loss ──
def prediction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((predicted - target) ** 2)

# ── Step 018  variance_loss ──
def variance_loss(embeddings: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    var = torch.var(embeddings, dim=0, unbiased=True)
    std = torch.sqrt(var + eps)
    loss = torch.mean(torch.relu(gamma - std))
    return loss

# ── Step 019  covariance_loss ──
def covariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    B, D = embeddings.shape
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (B - 1)
    off_diag = cov * (1 - torch.eye(D, device=cov.device))
    loss = (off_diag ** 2).sum() / D
    return loss

# ── Step 020  vicreg_regularizer ──
def vicreg_regularizer(embeddings: torch.Tensor, var_weight: float = 1.0, cov_weight: float = 0.04, gamma: float = 1.0) -> torch.Tensor:
    var_loss = variance_loss(embeddings, gamma)
    cov_loss = covariance_loss(embeddings)
    return var_weight * var_loss + cov_weight * cov_loss

# ── Step 021  jepa_loss ──
def jepa_loss(predicted: torch.Tensor, target: torch.Tensor, online_embeddings: torch.Tensor, pred_weight: float = 1.0, var_weight: float = 1.0, cov_weight: float = 0.04) -> torch.Tensor:
    pred_loss = prediction_loss(predicted, target)
    reg_loss = vicreg_regularizer(online_embeddings, var_weight, cov_weight)
    return pred_weight * pred_loss + reg_loss

# ── Step 022  collapse_metric ──
def collapse_metric(embeddings: torch.Tensor) -> torch.Tensor:
    stds = torch.std(embeddings, dim=0)
    return torch.mean(stds)

# ── Step 023  jepa_training_step ──
def jepa_training_step(batch: dict, encoder_params: dict, target_params: dict, predictor_params: dict, lr: float = 1e-3, tau: float = 0.99) -> tuple[dict, dict, dict, float, float]:
    obs = batch['observations']
    actions = batch['actions']
    next_obs = batch['next_observations']
    for key in encoder_params:
        encoder_params[key].requires_grad_(True)

    for key in predictor_params:
        predictor_params[key].requires_grad_(True)
    
    online_embeddings = encode_batch(obs, encoder_params)
    target_embeddings = encode_batch(next_obs, target_params).detach()
    predicted = predict_next_embedding(online_embeddings, actions, predictor_params)
    loss = jepa_loss(predicted, target_embeddings, online_embeddings)
    loss.backward()
    with torch.no_grad():
        for key in encoder_params:
            if encoder_params[key].grad is not None:
                encoder_params[key].data = encoder_params[key].data - lr * encoder_params[key].grad.data
                encoder_params[key].grad = None
        
        for key in predictor_params:
            if predictor_params[key].grad is not None:
                predictor_params[key].data = predictor_params[key].data - lr * predictor_params[key].grad.data
                predictor_params[key].grad = None
    
    target_params = ema_update(target_params, encoder_params, tau)
    col = collapse_metric(online_embeddings).item()
    return encoder_params, target_params, predictor_params, loss.item(), col

# ── Step 024  train_jepa ──
def train_jepa(dataset: dict, encoder_params: dict, target_params: dict, predictor_params: dict, num_steps: int = 50, batch_size: int = 32, lr: float = 1e-3, tau: float = 0.99, seed: int = 0) -> tuple[dict, dict, dict, list]:
    rng = torch.Generator()
    rng.manual_seed(seed)
    n = dataset['observations'].shape[0]
    history = []

    for step in range(num_steps):
        indices = torch.randint(0, n, (batch_size,), generator=rng)
        batch = {
            'observations': dataset['observations'][indices],
            'actions': dataset['actions'][indices],
            'next_observations': dataset['next_observations'][indices],
        }

        encoder_params, target_params, predictor_params, loss, col = jepa_training_step(
            batch, encoder_params, target_params, predictor_params, lr, tau
        )
        history.append({'loss': loss, 'collapse': col})

    return encoder_params, target_params, predictor_params, history

# ── Step 025  rollout_latent_dynamics ──
def rollout_latent_dynamics(initial_embedding: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    if initial_embedding.dim() == 1:
        T = actions.shape[0]
        current = initial_embedding.unsqueeze(0)
        trajectory = [current]
        
        for t in range(T):
            action = actions[t].unsqueeze(0)
            current = predict_next_embedding(current, action, predictor_params)
            trajectory.append(current)
        
        return torch.cat(trajectory, dim=0)
    else:
        if actions.dim() == 1:
            T = actions.shape[0]
            B = initial_embedding.shape[0]
            actions_expanded = actions.unsqueeze(0).expand(B, -1)
        else:
            B, T = actions.shape
            actions_expanded = actions
        
        current = initial_embedding
        trajectory = [current]
        
        for t in range(T):
            action = actions_expanded[:, t]
            current = predict_next_embedding(current, action, predictor_params)
            trajectory.append(current)
        
        return torch.stack(trajectory, dim=0)

# ── Step 026  multi_step_prediction_error ──
def multi_step_prediction_error(dataset: dict, encoder_params: dict, target_params: dict, predictor_params: dict, horizon: int = 5, num_samples: int = 32) -> float:
    obs = dataset['observations']
    actions = dataset['actions']
    next_obs = dataset['next_observations']
    N = obs.shape[0]
    max_start = N - horizon
    if max_start <= 0:
        return 0.0

    num_samples = min(num_samples, max_start)
    total_loss = 0.0
    total_steps = 0

    for i in range(num_samples):
        start_obs = obs[i].unsqueeze(0)
        current_embedding = encode_batch(start_obs, encoder_params)
        future_actions = actions[i:i+horizon]
        predicted_traj = rollout_latent_dynamics(current_embedding.squeeze(0), future_actions, predictor_params)
        predicted_future = predicted_traj[1:]
        future_obs = next_obs[i:i+horizon]
        true_future = encode_batch(future_obs, target_params)
        loss = torch.mean((predicted_future - true_future) ** 2)
        total_loss += loss.item()
        total_steps += 1

    return total_loss / total_steps if total_steps > 0 else 0.0

# ── Step 027  init_linear_probe ──
def init_linear_probe(latent_dim: int = 32, state_dim: int = 2, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    w = torch.randn(state_dim, latent_dim) * 0.01
    b = torch.zeros(state_dim)
    return {'w': w, 'b': b}

# ── Step 028  train_linear_probe ──
def train_linear_probe(embeddings: torch.Tensor, states: torch.Tensor, probe_params: dict, num_steps: int = 100, lr: float = 1e-2) -> dict:
    w = probe_params['w'].clone().detach().requires_grad_(True)
    b = probe_params['b'].clone().detach().requires_grad_(True)
    embeddings = embeddings.detach()
    states = states.float()

    for _ in range(num_steps):
        pred = embeddings @ w + b
        loss = torch.mean((pred - states) ** 2)
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad = None
            b.grad = None

    return {'w': w.detach(), 'b': b.detach()}

# ── Step 029  probe_state_recovery ──
def probe_state_recovery(dataset: dict, encoder_params: dict, probe_params: dict | None = None, num_probe_steps: int = 100) -> dict:
    obs = dataset['observations']
    embeddings = encode_batch(obs, encoder_params).detach()
    states = dataset['states'].float()

    if probe_params is None:
        probe_params = init_linear_probe(embeddings.shape[1], states.shape[1], seed=0)

    inner = {'w': probe_params['w'].T.contiguous(), 'b': probe_params['b']}
    t = train_linear_probe(embeddings, states, inner, num_probe_steps)
    trained = {'w': t['w'].T.contiguous(), 'b': t['b']}

    with torch.no_grad():
        pred_states = embeddings @ trained['w'].T + trained['b']
        mse = float(torch.mean((pred_states - states) ** 2).item())
        mean_abs_error = float(torch.mean(torch.abs(pred_states - states)).item())

    return {'mse': mse, 'mean_abs_error': mean_abs_error, 'probe_params': trained}

# ── Step 030  encode_goal ──
def encode_goal(goal_state: torch.Tensor, encoder_params: dict, room_size: int = 8) -> torch.Tensor:
    goal_abs = render_observation(goal_state, room_size)
    goal_emb = encode_batch(goal_abs.unsqueeze(0), encoder_params)
    return goal_emb.squeeze(0)

# ── Step 031  latent_cost ──
def latent_cost(latents, goal_embedding):
    return torch.sum((latents - goal_embedding) ** 2, dim=-1)

# ── Step 032  sample_action_sequences ──
def sample_action_sequences(n_sequences, horizon, n_actions):
    return torch.randint(0, n_actions, (n_sequences, horizon))

# ── Step 033  score_action_sequences ──
def score_action_sequences(start_embedding, action_sequences, goal_embedding, predictor_params):
    N, H = action_sequences.shape
    D = start_embedding.shape[0]
    total_costs = torch.zeros(N)

    for seq_idx in range(N):
        current = start_embedding
        total_cost = 0.0

        for t in range(H):
            action = action_sequences[seq_idx, t]
            current = predict_next_embedding(
                current.unsqueeze(0),
                action.unsqueeze(0),
                predictor_params
            ).squeeze(0)
            cost = latent_cost(current, goal_embedding)
            total_cost += cost

        total_costs[seq_idx]= total_cost

    return total_costs

# ── Step 034  select_best_plan ──
def select_best_plan(action_sequences, costs):
    best_idx = torch.argmin(costs)
    return action_sequences[best_idx]

# ── Step 035  mpc_step ──
def mpc_step(start_embedding, goal_embedding, predictor_params, n_sequences, horizon, n_actions):
    action_sequences = sample_action_sequences(n_sequences, horizon, n_actions)
    costs = score_action_sequences(start_embedding, action_sequences, goal_embedding, predictor_params)
    best_plan = select_best_plan(action_sequences, costs)
    return int(best_plan[0].item())

# ── Step 036  run_mpc_episode ──
def run_mpc_episode(encoder_params, predictor_params, goal_pos, room_size, agent_size, max_steps, n_sequences, horizon, n_actions):
    if isinstance(goal_pos, tuple):
        goal_pos = torch.tensor(goal_pos, dtype=torch.float32)
    elif isinstance(goal_pos, list):
        goal_pos = torch.tensor(goal_pos, dtype=torch.float32)

    state, obs = env_reset(room_size, seed=None)
    trajectory = [(state[0].item(), state[1].item())]

    goal_embedding = encode_goal(goal_pos, encoder_params, room_size)

    success = False
    steps = 0

    for step in range(max_steps):
        current_obs = render_observation(state, room_size)
        current_embedding = encode_batch(current_obs.unsqueeze(0),encoder_params).squeeze(0)
        action = mpc_step(current_embedding, goal_embedding, predictor_params, n_sequences, horizon, n_actions)
        state, obs = env_step(state, action, room_size)
        trajectory.append((state[0].item(), state[1].item()))
        steps += 1

        if torch.allclose(state, goal_pos, atol=0.5):
            success = True
            break

    final_distance = torch.sqrt(torch.sum((state - goal_pos) ** 2)).item()

    return {
        'success': success,
        'steps': steps,
        'trajectory': trajectory,
        'final_distance': final_distance,
    }

# ── Step 037  evaluate_planner ──
def evaluate_planner(encoder_params, predictor_params, room_size, agent_size, n_episodes, max_steps, n_sequences, horizon, n_actions):
    successes = 0
    total_steps = 0
    total_final_distance = 0.0

    for episode in range(n_episodes):
        goal = torch.randint(0, room_size, (2,)).float()

        result = run_mpc_episode(
            encoder_params, predictor_params, goal, room_size, agent_size,
            max_steps, n_sequences, horizon, n_actions
        )

        if result['success']:
            successes += 1

        total_steps += result['steps']
        total_final_distance += result['final_distance']

    return {
        'success_rate': successes / n_episodes,
        'mean_steps': total_steps / n_episodes,
        'mean_final_distance': total_final_distance / n_episodes,
    }

# ── Step 038  jepa_world_model_experiment ──
def jepa_world_model_experiment(room_size, agent_size, embed_dim, n_train_transitions, n_epochs, batch_size, n_probe_samples, n_eval_episodes, max_steps, n_sequences, horizon):
    torch.manual_seed(0)
    dataset = build_transition_dataset(n_train_transitions, room_size, seed=0)
    encoder_params = init_encoder_params(obs_channels=1, room_size=room_size, latent_dim=embed_dim, seed=0)
    target_params = init_target_encoder(encoder_params)
    predictor_params = init_predictor_params(latent_dim=embed_dim, action_dim=4, hidden_dim=64, seed=0)
    encoder_params, target_params, predictor_params, history = train_jepa(
        dataset, encoder_params, target_params, predictor_params,
        num_steps=n_epochs, batch_size=batch_size, lr=1e-3, tau=0.99, seed=0
    )
    train_losses = [h['loss'] for h in history]
    collapse_metrics = [h['collapse'] for h in history]
    probe_result = probe_state_recovery(dataset, encoder_params, num_probe_steps=100)
    mse = probe_result['mse']
    states = dataset['states']
    var = torch.mean((states - torch.mean(states, dim=0, keepdim=True)) ** 2) + 1e-8
    probe_r2 = 1 - mse / var.item()
    eval_result = evaluate_planner(
        encoder_params, predictor_params, room_size, agent_size,
        n_eval_episodes, max_steps, n_sequences, horizon, 4
    )

    return {
        'train_losses': train_losses,
        'collapse_metrics': collapse_metrics,
        'probe_r2': probe_r2,
        'success_rate': eval_result['success_rate'],
        'mean_steps': eval_result['mean_steps'],
    }

# ── Scaffold (runner) ──
"""End-to-end demo of an action-conditioned JEPA world model (LeCun-style).

Story: an untrained encoder gives COLLAPSED representations (tiny std) --
prediction in latent space is trivially easy but useless. Training with the
JEPA objective (prediction + variance/covariance regularization) produces
informative latents: a linear probe recovers the agent position, and
random-shooting MPC over the learned latent dynamics reaches goals far more
often than a random policy.
"""
import numpy as np
import torch


def random_policy_baseline(room_size, n_episodes, max_steps, seed=123):
    """Success rate of uniformly random actions (the bar MPC must beat)."""
    rng = np.random.default_rng(seed)
    successes = 0
    for ep in range(n_episodes):
        goal = rng.integers(0, room_size, size=2)
        state, obs = env_reset(room_size=room_size, seed=1000 + ep)
        for _ in range(max_steps):
            if int(state[0].item()) == int(goal[0]) and int(state[1].item()) == int(goal[1]):
                break
            state, obs = env_step(state, int(rng.integers(0, 4)), room_size=room_size)
        if int(state[0].item()) == int(goal[0]) and int(state[1].item()) == int(goal[1]):
            successes += 1
    return successes / n_episodes


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    room_size = 6
    latent_dim = 16
    seed = 0

    # ---- 1. Data ----
    dataset = build_transition_dataset(num_transitions=512, room_size=room_size, seed=seed)
    print("transitions:", int(dataset["actions"].shape[0]), "| obs shape:", tuple(dataset["observations"].shape))

    enc = init_encoder_params(obs_channels=1, room_size=room_size, latent_dim=latent_dim, seed=seed)
    tgt = init_target_encoder(enc)
    pred = init_predictor_params(latent_dim=latent_dim, action_dim=4, hidden_dim=32, seed=seed)

    # ---- 2. Before training: the collapse problem ----
    emb0 = encode_batch(dataset["observations"][:64].float(), enc)
    print(f"collapse metric BEFORE training: {float(collapse_metric(emb0)):.3f}  (near 0 = collapsed)")

    # ---- 3. Train the JEPA ----
    enc, tgt, pred, hist = train_jepa(
        dataset, enc, tgt, pred,
        num_steps=1500, batch_size=32, lr=0.1, tau=0.99, seed=seed,
    )
    print(f"JEPA loss: {hist[0]['loss']:.3f} -> {hist[-1]['loss']:.3f}")

    emb1 = encode_batch(dataset["observations"][:64].float(), enc)
    print(f"collapse metric AFTER training:  {float(collapse_metric(emb1)):.3f}  (target std is 1.0)")

    # ---- 4. The world model: multi-step latent prediction ----
    err = multi_step_prediction_error(dataset, enc, tgt, pred, horizon=5, num_samples=32)
    print(f"5-step latent prediction MSE (trained): {err:.3f}")
    print("  (an untrained, collapsed encoder scores ~0 here -- trivially easy and useless)")

    # ---- 5. No decoder needed: a linear probe recovers the state ----
    probe = probe_state_recovery(dataset, enc, probe_params=None, num_probe_steps=300)
    print(f"linear probe on agent position: mean abs error = {probe['mean_abs_error']:.2f} cells (room is {room_size}x{room_size})")

    # ---- 6. Acting = planning in latent space (no learned policy) ----
    stats = evaluate_planner(
        enc, pred, room_size=room_size, agent_size=1,
        n_episodes=10, max_steps=20, n_sequences=64, horizon=5, n_actions=4,
    )
    rnd = random_policy_baseline(room_size, n_episodes=10, max_steps=20)
    print(f"MPC planner:   success {stats['success_rate']:.0%}, mean final distance {stats['mean_final_distance']:.2f}")
    print(f"random policy: success {rnd:.0%}")


if __name__ == "__main__":
    main()
