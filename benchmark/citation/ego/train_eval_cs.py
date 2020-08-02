from __future__ import division
import random
import time
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch import tensor
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from train_edges import train_edges
from test_edges import test_edges
from negative_sampling import negative_sampling
from torch_geometric.utils import (remove_self_loops, add_self_loops)
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from train_eval import run_cs, run_
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, f1_score
import time
from gae import GAE, InnerProductDecoder
writer = SummaryWriter('runs/{}'.format(time.time()))


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask
district =0

def run(dataset, model, runs, epochs, lr, weight_decay, early_stopping,
        permute_masks=None, logger=None):
    
    batch_size = 30
    losses, accs , losses_wo, accs_wo= [], [], [], []

    for k in range(runs):
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        best_val_perf = test_perf = 0
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
        cold_mask_node = list(range(k*pivot, (k+1)*pivot))
        unknown = data.unknown
        for thing in unknown[0]:
            if thing in cold_mask_node:
                cold_mask_node.remove(thing) 
        data.test_masked_nodes = torch.tensor(cold_mask_node)
        train_node = range(num_nodes)
        train_node = [e for e in train_node if e not in cold_mask_node or unknown[0]]
        data.train_masked_nodes = torch.tensor(random.sample(train_node,140))
        epoch_num = int((num_nodes-pivot)/batch_size)
        print("{}-fold Result".format(k))

        loss_wo, acc_wo = run_(data, dataset, data.edge_index, train_node, writer)


        data = test_edges(data, cold_mask_node)    
        data = train_edges(data, data.train_masked_nodes)
        # 
        loss_wo, acc_wo = run_(data, dataset, data.train_edge_index, train_node, writer)
        losses_wo.append(loss_wo)
        accs_wo.append(acc_wo) 
        scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
        criterion = nn.BCEWithLogitsLoss(torch.ones(data.num_class).cuda())

        for epoch in range(2000):
            with torch.autograd.set_detect_anomaly(True):
                train_loss =train_Z(model, optimizer, data,epoch)   
        for epoch in range(20000):
            with torch.autograd.set_detect_anomaly(True):
                train_loss =train(model, optimizer,data,epoch, criterion)    
                scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        loss, acc = evaluate(model, data, criterion)
        losses.append(loss)
        accs.append(acc)
        print('Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(loss,acc))
    losses, accs, losses_wo, accs_wo = tensor(losses), tensor(accs), tensor(losses_wo), tensor(accs_wo)
    print('w/o Mean Val Loss: {:.4f}, Mean Test Accuracy: {:.3f} ± {:.3f}'.
        format(losses_wo.mean().item(),
                accs_wo.mean().item(),
                accs_wo.std().item()
                ))
    print('Mean Val Loss: {:.4f}, Mean Test Accuracy: {:.3f} ± {:.3f}'.
        format(losses.mean().item(),
                accs.mean().item(),
                accs.std().item()
                ))


def train_Z(model, optimizer, data, epoch):
    model.train()
    optimizer.zero_grad()
    pos_edge_index, neg_edge_index = data.train_pos_edge_index, data.train_neg_edge_index

    out, z, n,_= model(data, pos_edge_index, neg_edge_index, data.train_edge_index,  data.train_masked_nodes)
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


def train(model, optimizer,data, epoch, criterion):
    pos_edge_index, neg_edge_index = data.train_pos_edge_index, data.train_neg_edge_index

    logits, z, out,r= model(data, pos_edge_index, neg_edge_index, data.train_edge_index, data.train_masked_nodes)
    # z= z_out[data.masked_nodes]

    link_loss = pre_loss(z, data.train_edge_index)    
    y_pred = index_to_mask(data.train_masked_nodes.clone().detach(), size=data.num_nodes)
    # pdb.set_trace()
    loss = criterion(out[y_pred], data.y[y_pred])
    total_loss = loss + 0.1*link_loss
    
    total_loss.backward()
    optimizer.step()
    writer.add_scalar('cold/training loss', loss.item(), epoch)
    y_mask = index_to_mask(data.test_masked_nodes.clone().detach(), size=data.num_nodes)
    # pred = logits[y_mask].max(1)[1]
    test_loss = criterion(out[y_mask], data.y[y_mask]).item()
    writer.add_scalar('cold/val loss', test_loss, epoch)
    writer.add_scalar('cold/edge_num', r, epoch)
    # acc = pred.eq(data.y[y_mask]).sum().item() / y_mask.sum().item()
    pred = torch.sigmoid(out[y_mask]).data > 0.5
    pred = pred.detach().cpu().numpy()
    target = data.y[y_mask].detach().cpu().numpy()
    acc = f1_score(target, pred, average='micro')
    writer.add_scalar('cold/acc', acc, epoch)
    return total_loss


def evaluate(model, data, criterion):
    model.eval()
    pos_edge_index, neg_edge_index = data.test_pos_edge_index, data.test_neg_edge_index
    with torch.no_grad():
        logits, z, out,r=model(data, pos_edge_index, neg_edge_index.to(device), data.total_edge_index, data.test_masked_nodes)
    y_mask = index_to_mask(data.test_masked_nodes.clone().detach(), size=data.num_nodes)
    # loss = F.nll_loss(out[y_mask], data.y[y_mask]).item()
    # pred = out[y_mask].max(1)[1]
    # acc = pred.eq(data.y[y_mask]).sum().item() / y_mask.sum().item()

    loss = criterion(out[y_mask], data.y[y_mask]).item()
    # pred = logits[y_mask].max(1)[1]
    pred = torch.sigmoid(out[y_mask]).data > 0.5
    # acc = pred.eq(data.y[y_mask]).sum().item() / y_mask.sum().item()


    pred = pred.detach().cpu().numpy()
    target = data.y[y_mask].detach().cpu().numpy()
    acc = f1_score(target, pred, average='micro')
    pdb.set_trace()
    # print((pred.eq(data.y[y_mask]).sum().item()))

    # acc = (pred.eq(data.y[y_mask]).sum().item()) / (int(pred.size(0))*data.num_class.item())

    return loss, acc

def get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                              neg_edge_index.size(1)).float().to(device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

EPS = 1e-15
def recon_loss( z, pos_edge_index, neg_edge_index):
    decoder = InnerProductDecoder()

    pos_loss = -torch.log(
        decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

    neg_loss = -torch.log(1 -
                            decoder(z, neg_edge_index, sigmoid=True) +
                            EPS).mean()

    return pos_loss + neg_loss


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

def decoder( z, edge_index, sigmoid=True):
    value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    return torch.sigmoid(value) if sigmoid else value