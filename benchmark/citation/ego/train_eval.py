from __future__ import division

import time
import random
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
import pdb
from train_edges import train_edges
from test_edges import test_edges
from ssl_ import Net
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask
def run_cs(data, dataset, edge_index_cs , train_node):
    model = Net(dataset)
    losses, accs = [], []
    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    for i in range(2000):
        train(model, optimizer, data, edge_index_cs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    loss, acc = evaluate(model, data, edge_index_cs)
    losses.append(loss)
    accs.append(acc)
    print('Val Loss: {:.4f}, Test Accuracy: {:.3f}'.
    format(loss,
            acc))
    losses, accs = tensor(losses), tensor(accs)
    print('::::::Cold Edge::::::\n Mean Val Loss: {:.4f}, Mean Test Accuracy: {:.3f} ± {:.3f}'.
        format(losses.mean().item(),
                accs.mean().item(),
                accs.std().item()
                ))

def run_(data, dataset,edge_index, train_node, runs=10, lr=0.01, weight_decay=0.005):

    model = Net(dataset)
    losses, accs = [], []
    self_loop= torch.stack((torch.LongTensor(range(data.num_nodes)), torch.LongTensor(range(data.num_nodes))), dim=0).cuda()
    # pdb.set_trace()
    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    for i in range(500):
        # train(model, optimizer, data, self_loop)
        train(model, optimizer, data, edge_index)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # loss, acc = evaluate(model, data, self_loop)
    loss, acc = evaluate(model, data, edge_index)
    print('w/o Val Loss: {:.4f}, Test Accuracy: {:.3f}'.
    format(loss,
            acc))
    return loss, acc


def train(model, optimizer, data, edge_index):
    model.train()
    optimizer.zero_grad()
    out = model(data, edge_index)
    
    y_pred = index_to_mask(data.train_masked_nodes.clone().detach(), size=data.num_nodes)
    # pdb.set_trace()
    loss = F.nll_loss(out[y_pred], data.y[y_pred])
    loss.backward()
    optimizer.step()


def evaluate(model, data, edge_index):
    model.eval()

    with torch.no_grad():
        logits = model(data, edge_index)

    y_mask = index_to_mask(data.test_masked_nodes.clone().detach(), size=data.num_nodes)
    loss = F.nll_loss(logits[y_mask], data.y[y_mask]).item()
    pred = logits[y_mask].max(1)[1]
    acc = pred.eq(data.y[y_mask]).sum().item() / y_mask.sum().item()

    return loss, acc
