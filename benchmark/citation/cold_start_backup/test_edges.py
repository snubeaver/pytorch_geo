import math
import random
import torch
from torch_geometric.utils import to_undirected
import pdb

def test_edges(data, cold_mask_node, val_ratio=0.05, test_ratio=0.1):
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
    device = data.x.device
    num_nodes = data.num_nodes
    row, col = data.edge_index
    # data.edge_index = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    # Select train nodes
    
    edge_row, edge_col = row, col
    # print(data.edge_index.size())
    data.cold_mask_node = cold_mask_node

    # Select validate nodes
    for ind, i in enumerate(cold_mask_node):
        index = (row==i).nonzero()
        if(ind==0):
            indice=index
        else:
            indice = torch.cat((indice, index),0)

    test_indice = indice.squeeze()
    # print(test_indice.size())
    a_r, a_c = row[test_indice], col[test_indice]
    data.test_pos_edge_index = torch.stack([a_r, a_c], dim=0)
    # print(data.test_pos_edge_index.size())
    
    edge_mask = torch.Tensor(edge_row.size(0)).type(torch.bool).to(data.x.device)
    # print(edge_mask.size())
    edge_mask[edge_mask<1] = True
    # print(edge_mask.sum())
    edge_mask = edge_mask.scatter_(0, test_indice, False )
    # print(edge_mask.sum())
    # print(edge_row.s/=ize())
    edge_row = edge_row[edge_mask] 
    edge_col = edge_col[edge_mask]
    # print(edge_row.size())
    data.total_edge_index = torch.stack((edge_row, edge_col), dim=0 )
    # print(data.total_edge_index.size())
    # print(all_indice.size())

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0 #inverse adj made

    neg_row, neg_col = neg_adj_mask.nonzero().t()

    # test negative

    for ind, i in enumerate(cold_mask_node):
        index = (neg_row==i).nonzero()
        if(ind==0):
            indice=index
        else:
            indice = torch.cat((indice, index),0)
    neg_test_indice = indice.squeeze()

    # perm_test = random.sample(range(neg_test_indice.size(0)),
    #                      test_indice.size(0))

    # perm_test = torch.tensor(perm_test).to(torch.long)
    # neg_a_r, neg_a_c = neg_row[perm_val], neg_col[perm_val]
    # data.test_neg_edge_index = torch.stack([neg_a_r, neg_a_c], dim=0).to(device)

    neg_a_r, neg_a_c = neg_row[neg_test_indice], neg_col[neg_test_indice]
    data.test_neg_edge_index = torch.stack([neg_a_r, neg_a_c], dim=0).to(device)

    data.total_edge_index = to_undirected(data.total_edge_index)

    return data