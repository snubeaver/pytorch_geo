import math
import random
import torch
from torch_geometric.utils import to_undirected
import pdb


def train_edges(data, cold_mask_node, val_ratio=0.05, test_ratio=0.1):
    device = data.x.device
    num_nodes = data.num_nodes
    row, col = data.total_edge_index

    mask = row < col
    row, col = row[mask], col[mask]

    # Select train nodes
    edge_row, edge_col = row, col

    size = len(cold_mask_node)
    # print(size)
    indice_size =0
    for i in range(size):
        r_index = (row==cold_mask_node[i] ).nonzero()
        c_index = (col==cold_mask_node[i] ).nonzero()
        index = torch.unique(torch.cat((r_index,c_index), 0))
        if(i==0):
            indice=index
        else:
            indice = torch.cat((indice, index),0)
    train_indice = indice.squeeze()

    t_r, t_c = row[train_indice], col[train_indice]
    data.train_pos_edge_index = torch.stack([t_r, t_c], dim=0)

    edge_mask = torch.Tensor(edge_row.size(0)).type(torch.bool).to(data.x.device)
    edge_mask[edge_mask<1] = True
    # pdb.set_trace()
    edge_mask = edge_mask.scatter_(0, train_indice, False )
    edge_row = edge_row[edge_mask] 
    edge_col = edge_col[edge_mask]

    edge_index = torch.stack((edge_row, edge_col), dim=0 )
    

        

    # print(all_indice.size())

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0 #inverse adj made

    neg_row, neg_col = neg_adj_mask.nonzero().t()

    # train negative

    for i in range(size):
        r_index = (row==cold_mask_node[i] ).nonzero()
        c_index = (col==cold_mask_node[i] ).nonzero()
        index = torch.unique(torch.cat((r_index,c_index), 0))
        if(i==0):
            indice=index
        else:
            indice = torch.cat((indice, index),0)
    neg_train_indice = indice.squeeze()

    if(train_indice.dim()==0):
        indice_size = 1
    else:
        indice_size = train_indice.size(0)

    perm_train = random.sample(range(neg_train_indice.size(0)),
                         indice_size)

    perm_train = torch.tensor(perm_train).to(torch.long)
    neg_t_r, neg_t_c = neg_row[perm_train], neg_col[perm_train]
    data.train_neg_edge_index = torch.stack([neg_t_r, neg_t_c], dim=0).to(device)
    
    data.train_edge_index = to_undirected(edge_index,cold_mask_node, num_nodes)
    # pdb.set_trace()



    # VALIDATE

    # size = len(cold_mask_node)
    # indice_size =0
    # r , c = data.train_edge_index
    # for i in range(size):
    #     print((r==cold_mask_node[i]).nonzero())

    return data