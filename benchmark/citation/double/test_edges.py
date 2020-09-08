import math
import random
import torch
from torch_geometric.utils import to_undirected
import pdb
import numpy as np

def test_edges(data, cold_mask_node, val_ratio=0.05, test_ratio=0.1):
    
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
    max_degree = 0
    for ind, i in enumerate(cold_mask_node):
        r_index = (row==i ).nonzero()
        c_index = (col==i ).nonzero()
        
        index = torch.unique(torch.cat((r_index,c_index), 0))
        if(index.size(0)>max_degree): max_degree = index.size(0)
        if(ind==0):
            indice=index
        else:
            indice = torch.cat((indice, index),0)   

    test_indice = torch.unique(indice).squeeze()

    a_r, a_c = row[test_indice], col[test_indice]

    data.test_pos_edge_index = torch.stack([a_r, a_c], dim=0)
    
    
    # Building edge mask for GNN input
    
    edge_mask = torch.Tensor(edge_row.size(0)).type(torch.bool).to(data.x.device)
    edge_mask[edge_mask<1] = True

    edge_mask = edge_mask.scatter_(0, test_indice, False )
    edge_row = edge_row[edge_mask] 
    edge_col = edge_col[edge_mask]
    data.test_edge_index = torch.stack((edge_row, edge_col), dim=0 )

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0 #inverse adj made

    neg_row, neg_col = neg_adj_mask.nonzero().t().to(device)
    # test negative

    for ind, i in enumerate(cold_mask_node):
        r_index = (neg_row==i ).nonzero()
        c_index = (neg_col==i ).nonzero()
        index = torch.unique(torch.cat((r_index,c_index), 0))
        if(ind==0):
            indice=index
        else:
            indice = torch.cat((indice, index),0)
    
    neg_test_indice = torch.unique(indice).squeeze()

    if(test_indice.dim()==0):
        indice_size = 1
    else:
        indice_size = test_indice.size(0)

    perm_test = random.sample(range(neg_test_indice.size(0)),
                         indice_size)

    perm_test = torch.tensor(perm_test).to(torch.long).sort()[0]
    neg_a_r, neg_a_c = neg_row[perm_test], neg_col[perm_test]

    data.test_neg_edge_index = torch.stack([neg_a_r, neg_a_c], dim=0).to(device)
    data.test_edge_index = to_undirected(data.test_edge_index,cold_mask_node)

    
    return data