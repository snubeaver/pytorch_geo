from __future__ import division
import pdb, os, time, pickle, random , torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch import tensor
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from train_edges import train_edges
from test_edges import test_edges
from val_edges import val_edges
from negative_sampling import negative_sampling
from torch_geometric.utils import (remove_self_loops, add_self_loops)
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from train_eval import run_cs, run_
from torch.utils.tensorboard import SummaryWriter

import time
from gae import GAE, InnerProductDecoder

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask
district =0
tt = time.time()
def run(dataset, model, runs, epochs, lr, weight_decay, early_stopping,
        permute_masks=None, logger=None):
    
    batch_size = 30
    losses, accs , losses_wo, accs_wo= [], [], [], []


    lr = 0.01
    perm = torch.randperm(dataset[0].num_nodes)

    # runs=1
    for k in range(runs):
        aucs, aucs_ssl =[], []
        best_val_perf = test_perf = 0

        writer = SummaryWriter('runs/{}_{}'.format(k, tt))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        data = dataset[0]
        data = data.to(device)
        num_nodes = data.num_nodes

        if os.path.isfile('{}_{}.pkl'.format(str(dataset)[:-2], k)):
            data = pickle.load(open('{}_{}.pkl'.format(str(dataset)[:-2], k), 'rb'))
            
        else:
            pivot= int(num_nodes*0.1)

            cold_mask_node = perm[list(range(k*pivot, (k+1)*pivot))]
            data.test_masked_nodes =cold_mask_node
            train_node = range(num_nodes)
            train_node = [e for e in train_node if e not in cold_mask_node] #or unknown]
            data = test_edges(data, cold_mask_node)
            val_mask_node = random.sample(train_node, int(pivot*0.5))
            data.val_masked_nodes = torch.tensor(val_mask_node)
            data = val_edges(data, val_mask_node)
            train_node = [e for e in train_node if e not in val_mask_node] #or unknown]
            data.train_nodes = train_node
            data.train_masked_nodes = torch.tensor(random.sample(train_node,int(num_nodes*0.1)))
            data = train_edges(data, data.train_masked_nodes)

            with open('{}_{}.pkl'.format(str(dataset)[:-2], k), 'wb') as f:
                pickle.dump(data, f)
        print("{}-fold Result".format(k))
        train_node=data.train_nodes
        

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.7)
        loss_track =[float('inf')]*100

        # pdb.set_trace()
        for epoch in range(3000):
            with torch.autograd.set_detect_anomaly(True):
                train_loss, val_loss =train_Z(model, optimizer, data,epoch, writer) 
                # if(max(loss_track)<val_loss):
                #     break
                loss_track.pop(0)
                loss_track.append(val_loss)
                scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize() 
        auc,_ = link_auc(model, data)
        aucs.append(auc)
        print('Epoch: {}  AUC: {:.4f}'.format(epoch, auc))
        # pdb.set_trace()


        model.to(device).reset_parameters()
        optimizer_2 = Adam(model.parameters(), lr=0.0001, weight_decay=weight_decay)
        loss_track =[float('inf')]*100
        scheduler_2 = StepLR(optimizer_2, step_size=500, gamma=0.6)
        for epoch in range(7000):
            with torch.autograd.set_detect_anomaly(True):
                train_loss, val_loss =train(model, optimizer_2,data,epoch, writer)
                scheduler_2.step()
                # if(max(loss_track)<val_loss):
                #     break
                loss_track.pop(0)
                loss_track.append(val_loss)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        auc_ssl,_ = link_auc(model, data)
        print('Epoch: {}  AUC: {:.4f}'.format(epoch, auc_ssl))
        
        aucs_ssl.append(auc_ssl)
    aucs, aucs_ssl = tensor(aucs), tensor(aucs_ssl)
    print('AUC: {:.4f}'.
        format(aucs.mean().item()))
    print('AUC: {:.4f}'.format(aucs_ssl.mean().item()))


def train_Z(model, optimizer, data, epoch, writer):
    model.train()
    optimizer.zero_grad()
    pos_edge_index, neg_edge_index = data.train_pos_edge_index, data.train_neg_edge_index

    out, z= model(data, pos_edge_index, neg_edge_index, data.train_edge_index)


    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    link_loss = F.binary_cross_entropy_with_logits(z, link_labels)
    writer.add_scalar('link/loss', link_loss, epoch)

    link_loss.backward()
    optimizer.step()

    auc_val, val_loss = link_auc_val(model, data)
    writer.add_scalar('link/val_auc', auc_val, epoch)
    writer.add_scalar('link/val_loss', val_loss, epoch)
    auc, test_loss = link_auc(model, data)
    writer.add_scalar('link/test_auc', auc, epoch)

    return link_loss, val_loss
def Average(lst): 
    return sum(lst) / len(lst) 

def train(model, optimizer,data, epoch, writer):
    pos_edge_index, neg_edge_index = data.train_pos_edge_index, data.train_neg_edge_index
    logits, z= model(data, pos_edge_index, neg_edge_index, data.train_edge_index)
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    link_loss = F.binary_cross_entropy_with_logits(z, link_labels)
    writer.add_scalar('ssl/link_loss', link_loss, epoch)

    y_pred = index_to_mask(data.train_masked_nodes.clone().detach(), size=data.num_nodes)
    loss = F.nll_loss(logits[y_pred], data.y[y_pred])
    writer.add_scalar('ssl/loss', loss, epoch)
    lamda=0.05
    total_loss = lamda*loss + link_loss
    
    total_loss.backward()
    optimizer.step()
    y_mask = index_to_mask(data.test_masked_nodes.clone().detach(), size=data.num_nodes)
    test_loss = F.nll_loss(logits[y_mask], data.y[y_mask]).item()
    pred = logits[y_mask].max(1)[1]
    acc = pred.eq(data.y[y_mask]).sum().item() / y_mask.sum().item()

    auc_val, val_loss = link_auc_val(model, data)
    writer.add_scalar('ssl/val_auc', auc_val, epoch)
    writer.add_scalar('ssl/val_loss', val_loss, epoch)
    auc, test_loss = link_auc(model, data)
    writer.add_scalar('ssl/test_auc', auc, epoch)

    return total_loss, val_loss


def evaluate(model, data):
    model.eval()
    pos_edge_index, neg_edge_index = data.test_pos_edge_index, data.test_neg_edge_index
    with torch.no_grad():
        out,z =model(data, pos_edge_index, neg_edge_index.to(device), data.test_edge_index)
    y_mask = index_to_mask(data.test_masked_nodes.clone().detach(), size=data.num_nodes)
    loss = F.nll_loss(out[y_mask], data.y[y_mask]).item()
    pred = out[y_mask].max(1)[1]
    acc = pred.eq(data.y[y_mask]).sum().item() / y_mask.sum().item()

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


def pre_loss(z, pos_edge_index,neg_edge_index):
    pos_loss = -torch.log(
        decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
    neg_loss = -torch.log(1 -
                            decoder(z, neg_edge_index, sigmoid=True) +
                            EPS).mean()

    return pos_loss + neg_loss

def decoder( z, edge_index, sigmoid=True):
    value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    return torch.sigmoid(value) if sigmoid else value





def link_auc(model, data):
    model.eval()
    logits, z= model(data, data.test_pos_edge_index, data.test_neg_edge_index, data.test_edge_index)
    link_labels = get_link_labels(data.test_pos_edge_index, data.test_neg_edge_index)
    test_loss = F.binary_cross_entropy_with_logits(z, link_labels)
    link_probs = torch.sigmoid(z)
    link_probs = link_probs.detach().cpu().numpy()
    link_labels = link_labels.detach().cpu().numpy()
    return roc_auc_score(link_labels, link_probs), test_loss


def link_auc_val(model, data):
    model.eval()
    logits, z= model(data, data.val_pos_edge_index, data.val_neg_edge_index, data.total_edge_index)
    link_labels = get_link_labels(data.val_pos_edge_index, data.val_neg_edge_index)
    val_loss = F.binary_cross_entropy_with_logits(z, link_labels)
    link_probs = torch.sigmoid(z)
    link_probs = link_probs.detach().cpu().numpy()
    link_labels = link_labels.detach().cpu().numpy()
    return roc_auc_score(link_labels, link_probs), val_loss