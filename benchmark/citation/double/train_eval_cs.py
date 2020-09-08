from __future__ import division
import random
import time, os
import pdb
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch import tensor
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from train_edges import train_edges
from val_edges import val_edges
from test_edges import test_edges
from negative_sampling import negative_sampling
from torch_geometric.utils import (remove_self_loops, add_self_loops)
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from train_eval import run_cs, run_
from torch.utils.tensorboard import SummaryWriter
import pickle
import time
from gae import GAE, InnerProductDecoder

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask
district =0
# 사전학습 하면 true
pre = True
# 본학습때 사전학습 freeze
freeze = True
lamda = 0.01

# if(pre==False):
#     freeze=False

def run(dataset, model, runs, epochs, lr, weight_decay, early_stopping,
        permute_masks=None, logger=None):
    batch_size = 30
    losses, accs , losses_wo, accs_wo= [], [], [], []
    name = "Pre = {} Fix Lamda = {}  Freeze = {}".format(pre, lamda, freeze)
    print(name)
    writer = SummaryWriter('runs/{}_{}_{}'.format(str(dataset)[:-2], name, time.time()))

    perm = torch.randperm(dataset[0].num_nodes)


    for k in range(runs):
        model.to(device).reset_parameters()
        best_val_perf = test_perf = 0
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
        loss_wo, acc_wo = run_(data, dataset, data.train_edge_index, train_node,writer)
        losses_wo.append(loss_wo)
        accs_wo.append(acc_wo) 

        for param in model.conv1.parameters():
            param.requires_grad = True
        for param in model.conv2.parameters():
            param.requires_grad = True
        optimizer_1 = Adam(model.parameters(), lr=lr, weight_decay=0.0005)

        # Pre train 

        if(pre==True):
            for epoch in range(2000):
                with torch.autograd.set_detect_anomaly(True):
                    train_loss =train_Z(model, optimizer_1, data,epoch, writer)   

        #  Freeze Gate
        if(freeze==True):
            for param in model.conv1.parameters():
                param.requires_grad = False
            for param in model.conv2.parameters():
                param.requires_grad = False
        optimizer_2 = Adam([param for param in model.parameters() if param.requires_grad==True], lr=lr, weight_decay=0.0005)
        scheduler = StepLR(optimizer_2, step_size=500, gamma=0.7)

        # SSL Train

        for epoch in range(6000):
            with torch.autograd.set_detect_anomaly(True):
                train_loss =train(model, optimizer_2,data,epoch, writer)    
                scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        loss, acc = evaluate(model, data)
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


def train_Z(model, optimizer, data, epoch, writer):
    model.train()
    optimizer.zero_grad()
    pos_edge_index, neg_edge_index = data.train_pos_edge_index, data.train_neg_edge_index

    out, z, n= model(data, pos_edge_index, neg_edge_index, data.train_edge_index,  data.train_masked_nodes)

    total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    x_j = torch.index_select(z, 0, total_edge_index[0])
    x_i = torch.index_select(z, 0, total_edge_index[1])
    z= torch.einsum("ef,ef->e", x_i, x_j)
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    link_loss = F.binary_cross_entropy_with_logits(z, link_labels)
    writer.add_scalar('link/loss', link_loss, epoch)


    y_pred = index_to_mask(data.train_masked_nodes.clone().detach(), size=data.num_nodes)
    loss = F.nll_loss(out[y_pred], data.y[y_pred])
    # link_loss+=lamda*loss
    link_loss.backward()
    optimizer.step()
    auc = link_auc(model, data)
    writer.add_scalar('link/link_auc', auc, epoch)

    return link_loss


def train(model, optimizer,data, epoch, writer):
    pos_edge_index, neg_edge_index = data.train_pos_edge_index, data.train_neg_edge_index

    logits, z, new_edge= model(data, pos_edge_index, neg_edge_index, data.train_edge_index, data.train_masked_nodes)
    # z= z_out[data.masked_nodes]

    # Link loss
    total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    x_j = torch.index_select(z, 0, total_edge_index[0])
    x_i = torch.index_select(z, 0, total_edge_index[1])
    z= torch.einsum("ef,ef->e", x_i, x_j)
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    link_loss = F.binary_cross_entropy_with_logits(z, link_labels)
    writer.add_scalar('cold/link loss', link_loss.item(), epoch)

    # Classfication loss
    y_pred = index_to_mask(data.train_masked_nodes.clone().detach(), size=data.num_nodes)
    loss = F.nll_loss(logits[y_pred], data.y[y_pred])
    total_loss = loss + lamda*link_loss

    # writer.add_scalar('cold/weight', model.conv2.weight.sum(), epoch)

    total_loss.backward()
    optimizer.step()
    writer.add_scalar('cold/training loss', loss.item(), epoch)
    y_mask = index_to_mask(data.val_masked_nodes.clone().detach(), size=data.num_nodes)
    # pred = logits[y_mask].max(1)[1]
    val_loss = F.nll_loss(logits[y_mask], data.y[y_mask]).item()
    writer.add_scalar('cold/val loss', val_loss, epoch)
    writer.add_scalar('cold/edge_num', new_edge, epoch)
    pred = logits[y_mask].max(1)[1]
    acc = pred.eq(data.y[y_mask]).sum().item() / y_mask.sum().item()
    writer.add_scalar('cold/acc', acc, epoch)
    auc = link_auc(model, data)
    writer.add_scalar('cold/ssl_link_auc', auc, epoch)

    return total_loss


def evaluate(model, data):
    model.eval()
    pos_edge_index, neg_edge_index = data.test_pos_edge_index, data.test_neg_edge_index
    with torch.no_grad():
        out,z , r =model(data, pos_edge_index, neg_edge_index.to(device), data.test_edge_index, data.test_masked_nodes)
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

def decoder( z, edge_index, sigmoid=True):
    value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    return torch.sigmoid(value) if sigmoid else value


def link_auc(model, data):
    model.eval()
    pos_edge_index, neg_edge_index = data.test_pos_edge_index, data.test_neg_edge_index
    z= model(data, data.test_pos_edge_index, data.test_neg_edge_index, data.test_edge_index, data.test_masked_nodes)[1]
    total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    x_j = torch.index_select(z, 0, total_edge_index[0])
    x_i = torch.index_select(z, 0, total_edge_index[1])
    score = torch.einsum("ef,ef->e", x_i, x_j)
    link_probs = torch.sigmoid(score)
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    link_probs = link_probs.detach().cpu().numpy()
    link_labels = link_labels.detach().cpu().numpy()
    return roc_auc_score(link_labels, link_probs)