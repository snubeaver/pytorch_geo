from __future__ import division

import time
import pdb
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch import tensor
from torch.optim import Adam
from train_test_split_edges import train_test_split_edges
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

    val_losses, accs, durations = [], [], []
    for _ in range(runs):
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

        # best_val_loss = float('inf')
        # test_acc = 0
        # val_loss_history = []
        # cold_accs=[]
        for epoch in range(1, epochs + 1):
            data = dataset[0]
            data = data.to(device)
            data = train_test_split_edges(data)
            # mask = torch.ones(data.x.size(), dtype=data.x.dtype, device=data.x.device)
            # mask[data.cold_mask_node] = torch.zeros(data.x.size(1), dtype=data.x.dtype, device=data.x.device )
            
            # data.masked_node = data.x[data.cold_mask_node]
            # data.x = torch.mul(data.x, mask)

            train_loss=train(model, optimizer, data)
            val_perf, tmp_test_perf = evaluate(model, data)
            if val_perf > best_val_perf:
                best_val_perf = val_perf
                test_perf = tmp_test_perf
            log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, train_loss, best_val_perf, test_perf))
            
            # eval_info, cold_acc = evaluate(model, data)
            # cold_accs.append(cold_acc.item())
            # eval_info['epoch'] = epoch
            # if logger is not None:
            #     logger(eval_info)

            # if eval_info['val_loss'] < best_val_loss:
            #     best_val_loss = eval_info['val_loss']
            #     test_acc = eval_info['test_acc']

            # val_loss_history.append(eval_info['val_loss'])
            # if early_stopping > 0 and epoch > epochs // 2:
            #     tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
            #     if eval_info['val_loss'] > tmp.mean().item():
            #         break

        if torch.cuda.is_available():
            torch.cuda.synchronize()


        # val_losses.append(best_val_loss)
        # accs.append(test_acc)
        # durations.append(t_end - t_start)
        # print(cold_accs)

    # loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)

    # print('Val Loss: {:.4f},  Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
    #       format(loss.mean().item(),
    #              acc.mean().item(),
    #              acc.std().item(),
    #              duration.mean().item()))


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
    for prefix in ["val", "test"]:
        pos_edge_index, neg_edge_index = [
            index for _, index in data("{}_pos_edge_index".format(prefix),
                                       "{}_neg_edge_index".format(prefix))
        ]
        # pdb.set_trace()
        link_probs = torch.sigmoid(model(data, pos_edge_index, neg_edge_index.to(device)))
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        link_probs = link_probs.detach().cpu().numpy()
        link_labels = link_labels.detach().cpu().numpy()
        perfs.append(roc_auc_score(link_labels, link_probs))
    return perfs

def get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                              neg_edge_index.size(1)).float().to(device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels
