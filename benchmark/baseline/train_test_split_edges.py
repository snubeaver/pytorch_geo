import math
import random
import torch
from torch_geometric.utils import to_undirected
import pdb

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
    device = data.x.device
    num_nodes = data.num_nodes
    row, col = data.edge_index
    # data.edge_index = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    # Select train nodes
    cold_mask_node = torch.randperm(int(num_nodes*0.1))
    edge_row, edge_col = row, col

    data.cold_mask_node = cold_mask_node
    t_num = int(num_nodes*0.09)
    v_num = int(num_nodes*0.005)
    a_num = int(num_nodes*0.1) - t_num - v_num

    for i in range(t_num):
        index = (row==cold_mask_node[i]).nonzero()
        if(i==0):
            indice=index
        else:
            indice = torch.cat((indice, index),0)
    train_indice = indice.squeeze()


    t_r, t_c = row[train_indice], col[train_indice]
    data.train_pos_edge_index = torch.stack([t_r, t_c], dim=0)

    # Select validate nodes
    for i in range(v_num):
        index = (row==cold_mask_node[t_num+i]).nonzero()
        if(i==0):
            indice=index
        else:
            indice = torch.cat((indice, index),0)
    val_indice = indice.squeeze()
    
    v_r, v_c = row[val_indice], col[val_indice]
    data.val_pos_edge_index = torch.stack([v_r, v_c], dim=0)

    # Select validate nodes
    for i in range(a_num):
        index = (row==cold_mask_node[t_num+v_num+i]).nonzero()
        if(i==0):
            indice=index
        else:
            indice = torch.cat((indice, index),0)
    test_indice = indice.squeeze()
    
    a_r, a_c = row[test_indice], col[test_indice]
    data.test_pos_edge_index = torch.stack([a_r, a_c], dim=0)

    
    all_indice = torch.cat((train_indice, val_indice, test_indice))
    edge_mask = torch.Tensor(edge_row.size(0)).type(torch.bool).to(data.x.device)
    edge_mask[edge_mask<1] = True
    # pdb.set_trace()
    edge_mask = edge_mask.scatter_(0, all_indice, False )
    edge_row = edge_row[edge_mask] 
    edge_col = edge_col[edge_mask]

    data.edge_index = torch.stack((edge_row, edge_col), dim=0 )
    
    # print(all_indice.size())

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0 #inverse adj made

    neg_row, neg_col = neg_adj_mask.nonzero().t()

    # train negative

    for i in range(t_num):
        index = (neg_row==cold_mask_node[i]).nonzero()
        if(i==0):
            indice=index
        else:
            indice = torch.cat((indice, index),0)
    neg_train_indice = indice.squeeze()


    perm_train = random.sample(range(neg_train_indice.size(0)),
                         (train_indice.size(0)))

    perm_train = torch.tensor(perm_train).to(torch.long)
    neg_t_r, neg_t_c = neg_row[perm_train], neg_col[perm_train]
    data.train_neg_edge_index = torch.stack([neg_t_r, neg_t_c], dim=0).to(device)
    
    # val negative

    for i in range(v_num):
        index = (neg_row==cold_mask_node[t_num+i]).nonzero()
        if(i==0):
            indice=index
        else:
            indice = torch.cat((indice, index),0)
    neg_val_indice = indice.squeeze()


    perm_val = random.sample(range(neg_val_indice.size(0)),
                         val_indice.size(0))

    perm_val = torch.tensor(perm_val).to(torch.long)
    neg_v_r, neg_v_c = neg_row[perm_val], neg_col[perm_val]
    data.val_neg_edge_index = torch.stack([neg_v_r, neg_v_c], dim=0).to(device)

    # test negative

    for i in range(a_num):
        index = (neg_row==cold_mask_node[t_num+v_num+i]).nonzero()
        if(i==0):
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

    data.edge_index = to_undirected(data.edge_index)

    return data