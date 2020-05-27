import random

import torch
import numpy as np
from torch_geometric.utils import degree, to_undirected



def negative_sampling(edge_index, num_nodes=None, num_neg_samples=None,
                      force_undirected=False):

    num_neg_samples = num_neg_samples or edge_index.size(1)

    # Handle '|V|^2 - |E| < |E|' case for G = (V, E).
    num_neg_samples = min(num_neg_samples,
                          num_nodes * num_nodes - edge_index.size(1))


    rng = range(num_nodes**2)
    # idx = N * i + j
    idx = (edge_index[0] * num_nodes + edge_index[1]).to('cpu')

    perm = torch.tensor(random.sample(rng, num_neg_samples))
    # pos edge면 true 처리
    mask = torch.from_numpy(np.isin(perm, idx)).to(torch.bool)

    rest = mask.nonzero().view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(random.sample(rng, rest.size(0)))
        mask = torch.from_numpy(np.isin(tmp, idx)).to(torch.bool)
        perm[rest] = tmp
        rest = rest[mask.nonzero().view(-1)]


    row = perm / num_nodes
    col = perm % num_nodes
    neg_edge_index = torch.stack([row, col], dim=0).long()

    return neg_edge_index.to(edge_index.device)

