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
from torch.utils.tensorboard import SummaryWriter


from gae import GAE, InnerProductDecoder
writer = SummaryWriter('runs/140_train_nodes')

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
        cold_mask_node = range(k*pivot, (k+1)*pivot)
        data.test_masked_nodes = cold_mask_node
        train_node = range(num_nodes)
        train_node = [e for e in train_node if e not in cold_mask_node]
        data = test_edges(data, cold_mask_node)
        data.train_masked_nodes = random.sample(train_node,140)
        epoch_num = int((num_nodes-pivot)/batch_size)
        print("{}-fold Result".format(k))
        
        data = train_edges(data, data.train_masked_nodes)

        loss_wo, acc_wo = run_(data, dataset, data.train_edge_index, train_node)
        losses_wo.append(loss_wo)
        accs_wo.append(acc_wo) 
        for epoch in range(1000):
            with torch.autograd.set_detect_anomaly(True):
                train_loss =train(model, optimizer, data,epoch)    

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

def train_Z(model, optimizer, data, epoch):
    model.train()
    optimizer.zero_grad()
    pos_edge_index, neg_edge_index = data.train_pos_edge_index, data.train_neg_edge_index

    out= model(data, pos_edge_index, neg_edge_index, data.train_edge_index)

    link_loss = recon_loss(out, pos_edge_index, neg_edge_index)

    link_loss.backward()
    optimizer.step()
    return link_loss
def train(model, optimizer, data, epoch):
    model.train()
    optimizer.zero_grad()
    pos_edge_index, neg_edge_index = data.train_pos_edge_index, data.train_neg_edge_index

    out= model(data, pos_edge_index, neg_edge_index, data.train_edge_index, data.train_masked_nodes)

    link_loss = recon_loss(out, pos_edge_index, neg_edge_index)

    logits = F.log_softmax(out, dim=1)
    y_pred = index_to_mask(data.train_mask.clone().detach(), size=data.num_nodes)
    loss = F.nll_loss(logits[y_pred], data.y[y_pred])
    # print(loss)
    total_loss = loss + 0.1*link_loss


    total_loss.backward()
    optimizer.step()
    writer.add_scalar('training loss', loss.item(), epoch)
    y_mask = index_to_mask(tensor(data.test_masked_nodes, device=logits.device), size=data.num_nodes)
    pred = logits[y_mask].max(1)[1]
    acc = pred.eq(data.y[y_mask]).sum().item() / y_mask.sum().item()
    writer.add_scalar('acc', acc, epoch)
    return total_loss

# def train(model, optimizer, data, epoch):
#     model.train()
#     optimizer.zero_grad()
#     pos_edge_index, neg_edge_index = data.train_pos_edge_index, data.train_neg_edge_index

#     out,z,r= model(data, pos_edge_index, neg_edge_index, data.train_edge_index, data.train_masked_nodes)


#     logits = F.log_softmax(out, dim=1)
#     y_pred = index_to_mask(data.train_mask.clone().detach(), size=data.num_nodes)
#     loss = F.nll_loss(logits[y_pred], data.y[y_pred])

#     total_loss.backward()
#     optimizer.step()
#     writer.add_scalar('training loss', loss.item(), epoch)
#     y_mask = index_to_mask(tensor(data.test_masked_nodes, device=logits.device), size=data.num_nodes)
#     pred = logits[y_mask].max(1)[1]
#     acc = pred.eq(data.y[y_mask]).sum().item() / y_mask.sum().item()
#     writer.add_scalar('acc', acc, epoch)
#     return total_loss


def evaluate(model, data):
    model.eval()
    pos_edge_index, neg_edge_index = data.test_pos_edge_index, data.test_neg_edge_index
    with torch.no_grad():
        out =model(data, pos_edge_index, neg_edge_index.to(device), data.total_edge_index, data.test_masked_nodes)
    logits = F.log_softmax(out, dim=1)
    y_mask = index_to_mask(tensor(data.test_masked_nodes, device=logits.device), size=data.num_nodes)
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
def recon_loss( z, pos_edge_index, neg_edge_index):
    decoder = InnerProductDecoder()

    pos_loss = -torch.log(
        decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

    neg_loss = -torch.log(1 -
                            decoder(z, neg_edge_index, sigmoid=True) +
                            EPS).mean()

    return pos_loss + neg_loss