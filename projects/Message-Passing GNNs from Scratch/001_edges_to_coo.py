import torch

def edges_to_coo(edge_list, num_nodes=None):
    if isinstance(edge_list, torch.Tensor):
        edges = edge_list
    else:
        edges = torch.tensor(edge_list, dtype=torch.long)

    if edges.numel() == 0:
        edges = torch.zeros((0, 2), dtype=torch.long)
        if num_nodes is None:
            num_nodes = 0

        return edges[:, 0], edges[:, 1], int(num_nodes)

    edges = edges.reshape(-1, 2).long()

    src = edges[:, 0]
    dst = edges[:, 1]

    if num_nodes is None:
        max_idx = torch.max(edges).item()
        num_nodes = max_idx + 1
    else:
        num_nodes = int(num_nodes)

    return src, dst, num_nodes
