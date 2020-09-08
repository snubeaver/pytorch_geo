import math
import random
import torch
from torch_geometric.utils import to_undirected
import pdb
import numpy as np

def val_edges(data, mask_node, val_ratio=0.05):

    assert 'batch' not in data  # No batch-mode.
    device = data.x.device
    num_nodes = data.num_nodes
    row, col = data.test_edge_index
    # data.edge_index = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    # Select train nodes
    edge_row, edge_col = row, col

    # Select validate nodes
    max_degree = 0
    for ind, i in enumerate(mask_node):
        r_index = (row==i ).nonzero()
        c_index = (col==i ).nonzero()
        
        index = torch.unique(torch.cat((r_index,c_index), 0))
        if(index.size(0)>max_degree): max_degree = index.size(0)
        if(ind==0):
            indice=index
        else:
            indice = torch.cat((indice, index),0)

    val_indice = torch.unique(indice).squeeze()

    a_r, a_c = row[val_indice], col[val_indice]
    data.val_pos_edge_index = torch.stack([a_r, a_c], dim=0)
    
    edge_mask = torch.Tensor(edge_row.size(0)).type(torch.bool).to(data.x.device)
    edge_mask[edge_mask<1] = True
    edge_mask = edge_mask.scatter_(0, val_indice, False )
    edge_row = edge_row[edge_mask] 
    edge_col = edge_col[edge_mask]
    data.total_edge_index = torch.stack((edge_row, edge_col), dim=0 )

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0 #inverse adj made
    neg_row, neg_col = neg_adj_mask.nonzero().t().to(device)

    # test negative
    for ind, i in enumerate(mask_node):
        r_index = (neg_row==i ).nonzero()
        c_index = (neg_col==i ).nonzero()
        index = torch.unique(torch.cat((r_index,c_index), 0))
        if(ind==0):
            indice=index
        else:
            indice = torch.cat((indice, index),0)
    
    neg_val_indice = torch.unique(indice).squeeze()

    if(val_indice.dim()==0):
        indice_size = 1
    else:
        indice_size = val_indice.size(0)

    perm_test = random.sample(range(neg_val_indice.size(0)),
                         indice_size)

    perm_test = torch.tensor(perm_test).to(torch.long).sort()[0]
    neg_a_r, neg_a_c = neg_row[perm_test], neg_col[perm_test]

    data.val_neg_edge_index = torch.stack([neg_a_r, neg_a_c], dim=0).to(device)

    data.total_edge_index = to_undirected(data.total_edge_index,mask_node)

    return data