"""
Federated Averaging (FedAvg) from Scratch in PyTorch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  build_mlp_classifier ──
import torch
import torch.nn as nn


def build_mlp_classifier(input_size, hidden_size, num_classes):
    class _MLPClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    return _MLPClassifier(input_size, hidden_size, num_classes)

# ── Step 002  build_synthetic_dataset ──
import torch

def build_synthetic_dataset(num_samples, input_size, num_classes, seed):
    torch.manual_seed(seed)
    features = torch.randn(num_samples, input_size).float()
    labels = torch.randint(0, num_classes, (num_samples,)).long()
    return features, labels

# ── Step 003  train_test_split_dataset ──
import torch

def train_test_split_dataset(features, labels, test_fraction, seed):
    N = len(features)
    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(N, generator=generator)
    split_idx = int(N * test_fraction)
    train_indices = shuffled_indices[split_idx:]
    test_indices = shuffled_indices[:split_idx]

    tr_f = features[train_indices]
    tr_l = labels[train_indices]
    te_f = features[test_indices]
    te_l = labels[test_indices]
    
    return tr_f, tr_l, te_f, te_l

# ── Step 004  partition_data_iid ──
def partition_data_iid(train_features, train_labels, num_clients, seed):
    if num_clients == 0:
        num_clients = 1

    torch.manual_seed(seed)
    M = train_features.shape[0]
    indices = torch.randperm(M)
    shuffled_features = train_features[indices]
    shuffled_labels = train_labels[indices]
    client_data = []
    chunk_size = M // num_clients
    remainder = M % num_clients
    start = 0
    for i in range(num_clients):
        end = start + chunk_size + (1 if i < remainder else 0)
        client_data.append((shuffled_features[start:end], shuffled_labels[start:end]))
        start = end

    return client_data

# ── Step 005  partition_data_non_iid ──
import torch

def partition_data_non_iid(train_features, train_labels, num_clients, shards_per_client, seed):
    torch.manual_seed(seed)
    
    unique_labels = torch.unique(train_labels)
    num_labels = len(unique_labels)
    
    label_to_indices = {}
    for label in unique_labels:
        label_to_indices[int(label.item())] = torch.where(train_labels == label)[0]
    
    shuffled_labels = unique_labels[torch.randperm(num_labels)]
    
    client_data = []
    shard_idx = 0
    total_shards_needed = num_clients * shards_per_client
    
    label_list = shuffled_labels.tolist()
    while len(label_list) < total_shards_needed:
        label_list.extend(shuffled_labels.tolist())
    label_list = label_list[:total_shards_needed]
    
    for client in range(num_clients):
        client_shards = label_list[client * shards_per_client:(client + 1) * shards_per_client]
        
        client_indices = []
        for label in client_shards:
            client_indices.extend(label_to_indices[int(label)].tolist())
        
        client_indices_tensor = torch.tensor(client_indices, dtype=torch.long)
        perm = torch.randperm(len(client_indices_tensor))
        client_indices_tensor = client_indices_tensor[perm]
        
        client_data.append((train_features[client_indices_tensor], train_labels[client_indices_tensor]))
    
    return client_data

# ── Step 006  count_client_samples ──
def count_client_samples(client_partitions):
    return [features.shape[0] for features, _ in client_partitions]

# ── Step 007  iterate_client_batches ──
def iterate_client_batches(client_features, client_labels, batch_size, seed):
    torch.manual_seed(seed)
    n = client_features.shape[0]
    indices = torch.randperm(n)
    shuffled_features = client_features[indices]
    shuffled_labels = client_labels[indices]
    batches = []
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batches.append((shuffled_features[i:end], shuffled_labels[i:end]))

    return batches

# ── Step 008  compute_batch_loss ──
import torch.nn as nn

def compute_batch_loss(model, batch_features, batch_labels):
    criterion = nn.CrossEntropyLoss()
    logits = model(batch_features)
    loss = criterion(logits, batch_labels)
    return loss

# ── Step 009  local_sgd_step ──
import torch.nn as nn
import torch.optim as optim

def local_sgd_step(model, optimizer, batch_features, batch_labels):
    optimizer.zero_grad()
    loss = compute_batch_loss(model, batch_features, batch_labels)
    loss.backward()
    optimizer.step()
    return loss.item()

# ── Step 010  train_client_local ──
def train_client_local(model, client_features, client_labels, local_epochs, batch_size, learning_rate, seed):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(local_epochs):
        batches = iterate_client_batches(client_features, client_labels, batch_size, seed + epoch)
        for batch_features, batch_labels in batches:
            local_sgd_step(model, optimizer, batch_features, batch_labels)

    return model.state_dict()

# ── Step 011  clone_model_state ──
def clone_model_state(model):
    state_dict = model.state_dict()
    cloned_dict = {}
    for name, tensor in state_dict.items():
        cloned_dict[name]= tensor.detach().clone()

    return cloned_dict

# ── Step 012  load_model_state ──
def load_model_state(model, state_dict):
    model.load_state_dict(state_dict)
    return model

# ── Step 013  initialize_global_state ──
def initialize_global_state(input_size, hidden_size, num_classes, seed):
    torch.manual_seed(seed)
    model = build_mlp_classifier(input_size, hidden_size, num_classes)
    return clone_model_state(model)

# ── Step 014  add_state_dicts ──
def add_state_dicts(state_a, state_b):
    result = {}
    for key in state_a:
        result[key] = state_a[key] + state_b[key]
        
    return result

# ── Step 015  scale_state_dict ──
def scale_state_dict(state_dict, weight):
    result = {}
    for key in state_dict:
        result[key] = state_dict[key] * weight
        
    return result

# ── Step 016  aggregate_weighted_average ──
def aggregate_weighted_average(client_states, client_sample_counts):
    total_samples = sum(client_sample_counts)
    weighted_states = []

    for state, count in zip(client_states, client_sample_counts):
        weight = count / total_samples
        weighted_states.append(scale_state_dict(state, weight))

    result = weighted_states[0]
    for i in range(1, len(weighted_states)):
        result = add_state_dicts(result, weighted_states[i])

    return result

# ── Step 017  select_round_clients ──
def select_round_clients(num_clients, client_fraction, seed):
    torch.manual_seed(seed)
    clients = list(range(num_clients))
    K = max(1, round(client_fraction * num_clients))
    perm = torch.randperm(num_clients)
    selected_clients = perm[:K]
    return torch.sort(selected_clients).values.tolist()

# ── Step 018  run_communication_round ──
def run_communication_round(global_state, client_partitions, selected_clients, model_config, local_epochs, batch_size, learning_rate, seed):
    client_states = []
    client_sample_counts = []

    for client_idx in selected_clients:
        client_features, client_labels = client_partitions[client_idx]
        model = build_mlp_classifier(
            model_config['input_size'],
            model_config['hidden_size'],
            model_config['num_classes'],
        )
        load_model_state(model, global_state)
        trained_state = train_client_local(
            model, client_features, client_labels,
            local_epochs, batch_size, learning_rate,
            seed + client_idx,
        )
        client_states.append(trained_state)
        client_sample_counts.append(client_features.shape[0])

    new_global_state = aggregate_weighted_average(client_states, client_sample_counts)
    return new_global_state

# ── Step 019  evaluate_accuracy ──
def evaluate_accuracy(model, test_features, test_labels):
    logits = model(test_features)
    preds = torch.argmax(logits, dim=1)
    correct = (preds == test_labels).float()
    accuracy = correct.mean().item()
    return accuracy

# ── Step 020  run_fedavg ──
def run_fedavg(client_partitions, test_features, test_labels, model_config, num_rounds, client_fraction, local_epochs, batch_size, learning_rate, seed):
    global_state = initialize_global_state(
        model_config['input_size'],
        model_config['hidden_size'],
        model_config['num_classes'],
        seed
    )

    test_model = build_mlp_classifier(
        model_config['input_size'],
        model_config['hidden_size'],
        model_config['num_classes'],
    )

    accuracies = []
    num_clients = len(client_partitions)
    clients_per_round = max(1, int(num_clients * client_fraction))

    for round_idx in range(num_rounds):
        rng = torch.Generator()
        rng.manual_seed(seed + round_idx)
        selected = torch.randperm(num_clients, generator=rng)[:clients_per_round].tolist()
        global_state = run_communication_round(
            global_state, client_partitions, selected, model_config,
            local_epochs, batch_size, learning_rate, seed + round_idx
        )

        load_model_state(test_model, global_state)
        test_model.eval()
        with torch.no_grad():
            logits = test_model(test_features)
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == test_labels).float().mean().item()

        accuracies.append(accuracy)

    final_model = build_mlp_classifier(
        model_config['input_size'],
        model_config['hidden_size'],
        model_config['num_classes'],
    )
    load_model_state(final_model, global_state)
    return final_model, accuracies

# ── Step 021  train_centralized_baseline ──
def train_centralized_baseline(train_features, train_labels, test_features, test_labels, model_config, num_epochs, batch_size, learning_rate, seed):
    model = build_mlp_classifier(
        model_config['input_size'],
        model_config['hidden_size'],
        model_config['num_classes'],
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        batches = iterate_client_batches(train_features, train_labels, batch_size, seed + epoch)
        for batch_features, batch_labels in batches:
            local_sgd_step(model, optimizer, batch_features, batch_labels)

    return evaluate_accuracy(model, test_features, test_labels)

# ── Step 022  run_fedavg_iid ──
def run_fedavg_iid(train_features, train_labels, test_features, test_labels, model_config, num_clients, num_rounds, client_fraction, local_epochs, batch_size, learning_rate, seed):
    client_partitions = partition_data_iid(train_features, train_labels, num_clients, seed)
    model, accuracies = run_fedavg(
        client_partitions, test_features, test_labels, model_config,
        num_rounds, client_fraction, local_epochs, batch_size, learning_rate, seed
    )
    return accuracies

# ── Step 023  run_fedavg_non_iid ──
def run_fedavg_non_iid(train_features, train_labels, test_features, test_labels, model_config, num_clients, shards_per_client, num_rounds, client_fraction, local_epochs, batch_size, learning_rate, seed):
    client_partitions = partition_data_non_iid(train_features, train_labels, num_clients, shards_per_client, seed)
    model, accuracies = run_fedavg(
        client_partitions, test_features, test_labels, model_config,
        num_rounds, client_fraction, local_epochs, batch_size, learning_rate, seed
    )
    return model, accuracies

# ── Step 024  compute_non_iid_gap ──
def compute_non_iid_gap(iid_accuracies, non_iid_accuracies):
    iid_final = iid_accuracies[-1]
    non_iid_final = non_iid_accuracies[-1]
    gap = iid_final - non_iid_final
    return {
        'iid_final': float(iid_final),
        'non_iid_final': float(non_iid_final),
        'gap': float(gap)
    }

# ── Step 025  rounds_to_target_vs_local_epochs ──
def rounds_to_target_vs_local_epochs(client_partitions, test_features, test_labels, model_config, local_epochs_list, target_accuracy, num_rounds, client_fraction, batch_size, learning_rate, seed):
    result = {}

    for local_epochs in local_epochs_list:
        _, accuracies = run_fedavg(
            client_partitions, test_features, test_labels, model_config,
            num_rounds, client_fraction, local_epochs, batch_size, learning_rate, seed
        )

        found_round = None
        for round_idx, acc in enumerate(accuracies):
            if acc >= target_accuracy:
                found_round = round_idx
                break

        result[local_epochs] = found_round

    return result

# ── Step 026  accuracy_vs_client_fraction ──
def accuracy_vs_client_fraction(client_partitions, test_features, test_labels, model_config, client_fraction_list, num_rounds, local_epochs, batch_size, learning_rate, seed):
    result = {}
    for fraction in client_fraction_list:
        _, accuracies = run_fedavg(
            client_partitions, test_features, test_labels, model_config,
            num_rounds, fraction, local_epochs, batch_size, learning_rate, seed
        )

        result[float(fraction)] = accuracies[-1]

    return result

# ── Scaffold (runner) ──
"""Scaffold for Federated Averaging (FedAvg) from scratch in PyTorch.

Imports the full surface of functions the student will implement in
`solution.py`, then runs a minimal end-to-end demo on tiny synthetic data:
build data -> split -> partition -> run FedAvg rounds -> evaluate, plus a
quick non-IID vs IID comparison and a centralized baseline.
"""

import numpy as np
import torch

from solution import (
    build_mlp_classifier,
    build_synthetic_dataset,
    train_test_split_dataset,
    partition_data_iid,
    partition_data_non_iid,
    count_client_samples,
    iterate_client_batches,
    compute_batch_loss,
    local_sgd_step,
    train_client_local,
    clone_model_state,
    load_model_state,
    initialize_global_state,
    add_state_dicts,
    scale_state_dict,
    aggregate_weighted_average,
    select_round_clients,
    run_communication_round,
    evaluate_accuracy,
    run_fedavg,
    train_centralized_baseline,
    run_fedavg_iid,
    run_fedavg_non_iid,
    compute_non_iid_gap,
    rounds_to_target_vs_local_epochs,
    accuracy_vs_client_fraction,
)


def main():
    """Run a tiny FedAvg demo so the student can see the pipeline work."""
    np.random.seed(0)
    torch.manual_seed(0)

    # --- Toy experiment configuration (kept tiny for CPU) -----------------
    input_size = 8
    hidden_size = 16
    num_classes = 3
    num_samples = 300
    seed = 0

    model_config = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_classes": num_classes,
    }

    num_clients = 5
    num_rounds = 6
    client_fraction = 0.6
    local_epochs = 2
    batch_size = 16
    learning_rate = 0.05

    # --- Data preparation -------------------------------------------------
    features, labels = build_synthetic_dataset(
        num_samples, input_size, num_classes, seed
    )
    print(f"dataset: features={tuple(features.shape)} labels={tuple(labels.shape)}")

    train_x, train_y, test_x, test_y = train_test_split_dataset(
        features, labels, test_fraction=0.25, seed=seed
    )
    print(f"train={train_x.shape[0]} examples  test={test_x.shape[0]} examples")

    # --- Client partitioning ---------------------------------------------
    iid_parts = partition_data_iid(train_x, train_y, num_clients, seed)
    print("IID samples per client:", count_client_samples(iid_parts))

    non_iid_parts = partition_data_non_iid(
        train_x, train_y, num_clients, shards_per_client=2, seed=seed
    )
    print("non-IID samples per client:", count_client_samples(non_iid_parts))

    # --- Federated training on the IID partition --------------------------
    global_model, per_round_acc = run_fedavg(
        iid_parts, test_x, test_y, model_config,
        num_rounds=num_rounds, client_fraction=client_fraction,
        local_epochs=local_epochs, batch_size=batch_size,
        learning_rate=learning_rate, seed=seed,
    )
    print("FedAvg per-round test accuracy:",
          [round(float(a), 3) for a in per_round_acc])
    print(f"final FedAvg accuracy: {float(evaluate_accuracy(global_model, test_x, test_y)):.3f}")

    # --- IID vs non-IID comparison ---------------------------------------
    iid_curve = run_fedavg_iid(
        train_x, train_y, test_x, test_y, model_config,
        num_clients, num_rounds, client_fraction,
        local_epochs, batch_size, learning_rate, seed,
    )
    _, non_iid_curve = run_fedavg_non_iid(
        train_x, train_y, test_x, test_y, model_config,
        num_clients, 2, num_rounds, client_fraction,
        local_epochs, batch_size, learning_rate, seed,
    )
    gap = compute_non_iid_gap(iid_curve, non_iid_curve)
    print(f"IID final={float(iid_curve[-1]):.3f}  "
          f"non-IID final={float(non_iid_curve[-1]):.3f}  gap={gap}")

    # --- Centralized baseline for reference -------------------------------
    baseline_acc = train_centralized_baseline(
        train_x, train_y, test_x, test_y, model_config,
        num_epochs=num_rounds, batch_size=batch_size,
        learning_rate=learning_rate, seed=seed,
    )
    print(f"centralized baseline accuracy: {float(baseline_acc):.3f}")

    # --- Probes: local epochs and client fraction -------------------------
    rounds_needed = rounds_to_target_vs_local_epochs(
        iid_parts, test_x, test_y, model_config,
        local_epochs_list=[1, 2, 4], target_accuracy=0.5,
        num_rounds=num_rounds, client_fraction=client_fraction,
        batch_size=batch_size, learning_rate=learning_rate, seed=seed,
    )
    print("rounds-to-target vs local epochs:", rounds_needed)

    frac_results = accuracy_vs_client_fraction(
        iid_parts, test_x, test_y, model_config,
        client_fraction_list=[0.2, 0.6, 1.0], num_rounds=num_rounds,
        local_epochs=local_epochs, batch_size=batch_size,
        learning_rate=learning_rate, seed=seed,
    )
    print("accuracy vs client fraction:", frac_results)


if __name__ == "__main__":
    main()
