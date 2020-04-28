import math
import random
import torch
from torch_geometric.utils import to_undirected


def train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1):
    r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges, and adds attributes of
    `train_pos_edge_index`, `train_neg_adj_mask`, `val_pos_edge_index`,
    `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
    to :attr:`data`.

    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation
            edges. (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test
            edges. (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.

    row, col = data.edge_index
    data.edge_index = None
    num_nodes = data.num_nodes

    # Return upper triangular portion.
    mask = row < col
    # make undirected graph
    row, col = row[mask], col[mask]




    # Positive edges on last node
    perm = torch.randperm(row.size(0))
    row_u, col_u = row[row==(num_nodes-1)], col[row==(num_nodes-1)]
    row_l, col_l = row[row==(num_nodes-1)], col[row==(num_nodes-1)]
    row = torch.concat(row_u, row_l)
    col = torch.concat(col_u, col_l)

    n_v = int(math.floor(row.size(0)))
    n_t = int(math.floor(row.size(0)))
    
    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[:n_v], col[:n_v]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[:n_v], col[:n_v]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    # Maybe we not need this neg edge for this
    
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0 #if there is connection 0

    # 3 random edges
    extra_edge=3
    neg_row, neg_col = neg_adj_mask.nonzero().t()
    perm = random.sample(range(neg_row.size(0)),
                         extra_edge)
    perm = torch.tensor(perm)
    perm = perm.to(torch.long)
    neg_row, neg_col = neg_row[num_nodes-1], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[extra_edge], neg_col[extra_edge]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[extra_edge], neg_col[extra_edge]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data
