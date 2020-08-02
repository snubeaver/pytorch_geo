from __future__ import division
import random
import time
import pdb
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch import tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from train_edges import train_edges
from test_edges import test_edges
from negative_sampling import negative_sampling
from torch_geometric.utils import (remove_self_loops, add_self_loops)
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from train_eval import run_cs, run_
from torch.utils.tensorboard import SummaryWriter

import time
from gae import GAE, InnerProductDecoder
writer = SummaryWriter('runs/{}'.format(time.time()))

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask
district =0

def run(dataset, model,model_ssl, runs, epochs, lr, weight_decay, early_stopping,
        permute_masks=None, logger=None):
    
    batch_size = 30
    for k in range(runs):
        district =k
        model.to(device).reset_parameters()
        model_ssl.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
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
        print("{}-fold Result".format(k))

        # one possibile solution: fix the train_mask_nodes to 140 nodes
        run_(data, dataset, data.total_edge_index, train_node)

        '''
        for i in range(20):
            for epoch in range(0, epoch_num):
                data.masked_nodes = train_node[epoch*batch_size:(epoch+1)*batch_size]
                data = train_edges(data, data.masked_nodes)
                with torch.autograd.set_detect_anomaly(True):
                    train_loss =train(model, optimizer, data)           
            # log = 'Epoch: {:03d}, Loss: {:.4f}, Test: {:.4f}'
            # print(log.format(epoch, train_loss, test_auc))
        '''
        data.masked_nodes  = data.train_mask
        data = train_edges(data, data.masked_nodes)
        scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)
        for epoch in range(2000):
            with torch.autograd.set_detect_anomaly(True):
                train_loss =train_Z(model, optimizer, data,epoch)   
        for epoch in range(5000):
            with torch.autograd.set_detect_anomaly(True):
                train_loss =train(model, optimizer,data,epoch)    
                scheduler.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        loss, acc = evaluate(model, data)
        print('Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(loss,acc))

def train_Z(model, optimizer, data, epoch):
    model.train()
    optimizer.zero_grad()
    pos_edge_index, neg_edge_index = data.train_pos_edge_index, data.train_neg_edge_index

    out, z, n= model(data, pos_edge_index, neg_edge_index, data.train_edge_index)
    # pos_pred = decoder(z, pos_edge_index, sigmoid=True)
    # neg_pred = decoder(z, neg_edge_index, sigmoid=True)
    # total_pred = torch.cat([pos_pred, neg_pred], dim=-1)
    # r, c = total_edge_index[0][total_pred>0.5], total_edge_index[1][total_pred>0.5]
    # new_index = torch.stack((torch.cat([r,c], dim= -1),(torch.cat([c,r], dim= -1))), dim=0 )
    # added_index = torch.cat([edge_index, new_index], dim=-1)
        
    # link_loss = recon_loss(z, pos_edge_index, neg_edge_index)
    link_loss = pre_loss(z, data.train_edge_index)
    link_loss.backward()
    optimizer.step()
    return link_loss


def train(model, optimizer,data, epoch):
    pos_edge_index, neg_edge_index = data.train_pos_edge_index, data.train_neg_edge_index

    logits, z, new_edge= model(data, pos_edge_index, neg_edge_index, data.train_edge_index)
    # z= z_out[data.masked_nodes]

    # link_loss = pre_loss(z, data.train_edge_index)    
    y_pred = index_to_mask(tensor(data.train_mask, device=logits.device), size=data.num_nodes)
    loss = F.nll_loss(logits[y_pred], data.y[y_pred])
    total_loss = loss #+ link_loss
    
    total_loss.backward()
    optimizer.step()
    writer.add_scalar('training loss', loss.item(), epoch)
    y_mask = index_to_mask(tensor(data.test_mask, device=logits.device), size=data.num_nodes)
    pred = logits[y_mask].max(1)[1]
    acc = pred.eq(data.y[y_mask]).sum().item() / y_mask.sum().item()
    writer.add_scalar('acc', acc, epoch)
    # print(new_edge)
    return total_loss

def evaluate(model, data):
    model.eval()
    pos_edge_index, neg_edge_index = data.test_pos_edge_index, data.test_neg_edge_index
    with torch.no_grad():
        # logits, z, n =model(data, pos_edge_index, neg_edge_index.to(device), data.total_edge_index)
        logits, z, n =model(data, pos_edge_index, neg_edge_index.to(device), data.test_edge_index)
    y_mask = index_to_mask(tensor(data.test_mask, device=logits.device), size=data.num_nodes)
    loss = F.nll_loss(logits[y_mask], data.y[y_mask]).item()
    pred = logits[y_mask].max(1)[1]
    acc = pred.eq(data.y[y_mask]).sum().item() / y_mask.sum().item()

    return loss, acc

def get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                              neg_edge_index.size(1)).float().to(device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

EPS = 1e-15
def pre_loss(z, pos_edge_index):
    pos_loss = -torch.log(
        decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
    pos_edge_index, _ = remove_self_loops(pos_edge_index)
    pos_edge_index, _ = add_self_loops(pos_edge_index)
    neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
    neg_loss = -torch.log(1 -
                            decoder(z, neg_edge_index, sigmoid=True) +
                            EPS).mean()

    return pos_loss + neg_loss


def recon_loss( z, pos_edge_index, neg_edge_index):

    pos_loss = -torch.log(
        decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

    neg_loss = -torch.log(1 -
                            decoder(z, neg_edge_index, sigmoid=True) +
                            EPS).mean()

    return pos_loss + neg_loss

def decoder( z, edge_index, sigmoid=True):
    value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    return torch.sigmoid(value) if sigmoid else value