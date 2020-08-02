from __future__ import division
import random
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
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from train_eval import run_cs, run_


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def run(dataset, model, runs, epochs, lr, weight_decay, early_stopping,
        permute_masks=None, logger=None):

    batch_size = 30
    test_losses, accs, p0, p1, r0, r1, f0, f1 = [], [], [], [], [], [], [], []

    for k in range(runs):
        
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        best_val_perf = test_perf = 0
        
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # t_start = time.perf_counter()

        
        
        data = dataset[0]
        # gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
        #             normalization_out='col',
        #             diffusion_kwargs=dict(method='ppr', alpha=0.05),
        #             sparsification_kwargs=dict(method='topk', k=128,
        #                                         dim=0), exact=True)
        # data = gdc(data)
        data = data.to(device)
        num_nodes = data.num_nodes
        pivot= int(num_nodes*0.1)
        cold_mask_node = range(k*pivot, (k+1)*pivot)
        data.test_mask = cold_mask_node
        train_node = range(num_nodes)
        train_node = [e for e in train_node if e not in cold_mask_node]
        data = test_edges(data, cold_mask_node)
        data.train_mask = random.sample(train_node,140)
        epoch_num = int((num_nodes-pivot)/batch_size)
        print("{}-fold Result".format(k))




        for i in range(5):
            for epoch in range(0, epoch_num):
                data.masked_nodes = train_node[epoch*batch_size:(epoch+1)*batch_size]
                data = train_edges(data, data.masked_nodes)
                train_loss =train(model, optimizer, data)
            # log = 'Epoch: {:03d}, Loss: {:.4f}, Test: {:.4f}'
            # print(log.format(epoch, train_loss, test_auc))
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        test_auc, test_recall, test_loss, c_m, edge_index_cs = evaluate(model, data)
        test_losses.append(test_loss)
        accs.append(test_auc)
        p0.append(test_recall[0][0])
        p1.append(test_recall[0][1])
        r0.append(test_recall[1][0])
        r1.append(test_recall[1][1])
        f0.append(test_recall[2][0])
        f1.append(test_recall[2][1])
        print("noedge: {}  edge: {}".format(data.total_edge_index[0].size(), edge_index_cs[0].size()))
        run_(data, dataset, data.total_edge_index, train_node )
        run_cs(data, dataset,edge_index_cs, train_node)
        # loss, acc = tensor(train_losses), tensor(accs)
        print(c_m)
        print("Train loss: {:.4f},  Test AUC Score: {:.3f},\n precision: {:.4f}, {:.4f},\n recall: {:.4f}, {:.4f},\n fscore: {:.4f}, {:.4f}".format(train_loss, test_auc, test_recall[0][0], test_recall[0][1], test_recall[1][0], test_recall[1][1], test_recall[2][0], test_recall[2][1]))
    acc_, acc, p0, p1, r0, r1, f0, f1 = tensor(test_losses), tensor(accs), tensor(p0), tensor(p1), tensor(r0), tensor(r1), tensor(f0), tensor(f1)
    print("\n===============\n")
    print("Final Result")
    print('Loss: {:.4f},  Test AUC: {:.3f} ± {:.3f}\n precision: {:.4f}, {:.4f},\n recall: {:.4f}, {:.4f},\n fscore: {:.4f}, {:.4f}'.
        format(acc_.mean().item(),
                acc.mean().item(),
                acc.std().item(),
                p0.mean(), p1.mean(),
                r0.mean(), r1.mean(),
                f0.mean(), f1.mean()
    ))

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    pos_edge_index, neg_edge_index = data.train_pos_edge_index, data.train_neg_edge_index

    out, score_loss= model(data, pos_edge_index, neg_edge_index, data.train_edge_index)

    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(out, link_labels)

    loss += score_loss.sum()
    loss.backward()
    optimizer.step()

    return loss


def evaluate(model, data):
    model.eval()
    perfs = []
    for prefix in ["test"]:
        pos_edge_index, neg_edge_index = [
            index for _, index in data("{}_pos_edge_index".format(prefix),
                                       "{}_neg_edge_index".format(prefix))
        ]
        
        all_edge_index = torch.cat((pos_edge_index, neg_edge_index),1)
        out,_ = model(data, pos_edge_index, neg_edge_index.to(device), data.total_edge_index)
        _, indices = torch.topk(out, 100)


        


        link_probs = torch.sigmoid(out)
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        loss = F.binary_cross_entropy_with_logits(out, link_labels)
        
        add_r = all_edge_index[0][indices]
        add_c = all_edge_index[1][indices]
        r, c = data.total_edge_index
        r = torch.cat((r, add_r, add_c))
        c = torch.cat((c, add_c, add_r))
        
        edge_index_cs = torch.stack([r,c], dim=0)
        # pdb.set_trace()
        link_probs = link_probs.detach().cpu().numpy()
        link_labels = link_labels.detach().cpu().numpy()
        
    return roc_auc_score(link_labels, link_probs), precision_recall_fscore_support(link_labels, link_probs>0.5), loss, confusion_matrix(link_labels, link_probs>0.5), edge_index_cs


def get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                              neg_edge_index.size(1)).float().to(device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels