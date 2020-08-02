from __future__ import division

import time
import pdb
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch import tensor
from torch.optim import Adam
from train_edges import train_edges
from test_edges import test_edges
from negative_sampling import negative_sampling
from torch_geometric.utils import (remove_self_loops, add_self_loops)

from sklearn.metrics import roc_auc_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, epochs, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    # index = torch.randperm(data.num_nodes)
    # index = torch.narrow(index, 0,0,epochs)
    # t_ = int(index.size(0)*0.8)
    # v_ = int(index.size(0)*0.1)

    # data.train_node =index
    # data.val_node = index[t_:t_+v_]
    # data.test_node = index[t_+v_:]

    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data)

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data



def run(dataset, model, runs, epochs, lr, weight_decay, early_stopping,
        permute_masks=None, logger=None):

    batch_size = 30
    train_losses, accs, durations = [], [], []

    for k in range(runs):
        # data = dataset[0]
        # # pdb.set_trace()
        # # if permute_masks is not None:
        # #     data = permute_masks(data, runs, dataset.num_classes)
        # data = data.to(device)
        
        
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        best_val_perf = test_perf = 0
        
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # t_start = time.perf_counter()

        
        
        data = dataset[0]
        data = data.to(device)
        num_nodes = data.num_nodes
        pivot= int(num_nodes*0.1)
        cold_mask_node = range(k*pivot, (k+1)*pivot)
        train_node = range(num_nodes)
        train_node = [e for e in train_node if e not in cold_mask_node]
        data = test_edges(data, cold_mask_node)
        epoch_num = int((num_nodes-pivot)/batch_size)
        print("{}-fold Result".format(k))
        for epoch in range(0, epoch_num):
            # mask = torch.ones(data.x.size(), dtype=data.x.dtype, device=data.x.device)
            # mask[data.cold_mask_node] = torch.zeros(data.x.size(1), dtype=data.x.dtype, device=data.x.device )
            # best_val_loss = float('inf')
            # test_acc = 0
            # val_loss_history = []
            # cold_accs=[]
            # data.masked_node = data.x[data.cold_mask_node]
            # data.x = torch.mul(data.x, mask)
            data = train_edges(data, train_node[epoch*batch_size:(epoch+1)*batch_size])
            train_loss =train(model, optimizer, data)
            test_auc = evaluate(model, data)
            # log = 'Epoch: {:03d}, Loss: {:.4f}, Test: {:.4f}'
            # print(log.format(epoch, train_loss, test_auc))
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        test_auc = evaluate(model, data)
        train_losses.append(train_loss)
        accs.append(test_auc)
        # val_losses.append(best_val_loss)
        # accs.append(test_acc)
        # durations.append(t_end - t_start)
        # print(cold_accs)

        # loss, acc = tensor(train_losses), tensor(accs)

        print('Loss: {:.4f},  Test AUC Score: {:.3f}'.
            format(train_loss, test_auc))
    loss, acc = tensor(train_losses), tensor(accs)
    print("Final Result")
    print('Loss: {:.4f},  Test AUC: {:.3f} ± {:.3f}'.
        format(loss.mean().item(),
                acc.mean().item(),
                acc.std().item()))

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    pos_edge_index, neg_edge_index = data.train_pos_edge_index, data.train_neg_edge_index

    out = model(data, pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    # adj = to_dense_adj(data.edge_index).squeeze(0)
    # pdb.set_trace()
    # loss = torch.nn.MSELoss()
    # cold_loss = loss( ((s2+s3)/2).squeeze(), adj[data.mask_num])
    # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    # loss += cold_loss
    loss = F.binary_cross_entropy_with_logits(out, link_labels)
    loss.backward()
    optimizer.step()

    return loss
'''
def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits, s1, s2, s3 = model(data)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        adj = to_dense_adj(data.edge_index).squeeze()
        loss = torch.nn.MSELoss()
        cold_loss = loss(((s2+s3)/2).squeeze(), adj[data.mask_num])
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        loss+=cold_loss
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()


        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc

    edge_num = (adj[data.mask_num] == 1).sum()
    score_sum = s2+s3
    # print(int(edge_num))
    _, indices = torch.topk(score_sum, int(edge_num)*10)
    indices = torch.sort(indices)[0]
    cold_pred = torch.zeros(s1.size(), dtype=adj[data.mask_num].dtype, device=adj[data.mask_num].device)
    # pdb.set_trace()
    cold_pred.scatter_(0, indices, 1)
    target = (adj[data.mask_num] != 0).nonzero()
    target = torch.sort(target)[0].squeeze()
    # print(indices)
    # print(target)
    cold_acc = indices.sort()[0].eq(target.sort()[0]).sum().item() / edge_num
    # print(cold_acc)
    return outs, cold_acc
'''
def evaluate(model, data):
    model.eval()
    perfs = []
    for prefix in ["test"]:
        pos_edge_index, neg_edge_index = [
            index for _, index in data("{}_pos_edge_index".format(prefix),
                                       "{}_neg_edge_index".format(prefix))
        ]
        # pdb.set_trace()
        link_probs = torch.sigmoid(model(data, pos_edge_index, neg_edge_index.to(device)))
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        link_probs = link_probs.detach().cpu().numpy()
        link_labels = link_labels.detach().cpu().numpy()
        
    return roc_auc_score(link_labels, link_probs)

def get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                              neg_edge_index.size(1)).float().to(device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels
