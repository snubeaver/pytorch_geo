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
    for i in range(200):
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

def run_(data, dataset, train_node, runs=10, lr=0.01, weight_decay=0.005):

    model = Net(dataset)
    losses, accs = [], []
    
    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    for i in range(200):
        train(model, optimizer, data, data.edge_index)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    loss, acc = evaluate(model, data, data.edge_index)
    losses.append(loss)
    accs.append(acc)
    print('Val Loss: {:.4f}, Test Accuracy: {:.3f}'.
    format(loss,
            acc))
    losses, accs = tensor(losses), tensor(accs)
    print('::::::No Edge::::::\n Val Loss: {:.4f}, Mean Test Accuracy: {:.3f} ± {:.3f}'.
        format(losses.mean().item(),
                accs.mean().item(),
                accs.std().item()
                ))


def train(model, optimizer, data, edge_index):
    model.train()
    optimizer.zero_grad()
    out = model(data, edge_index)
    y_pred = index_to_mask(tensor(data.train_mask, device=out.device), size=data.num_nodes)
    loss = F.nll_loss(out[y_pred], data.y[y_pred])
    loss.backward()
    optimizer.step()


def evaluate(model, data, edge_index):
    model.eval()

    with torch.no_grad():
        logits = model(data, edge_index)

    y_mask = index_to_mask(tensor(data.test_mask, device=logits.device), size=data.num_nodes)
    loss = F.nll_loss(logits[y_mask], data.y[y_mask]).item()
    pred = logits[y_mask].max(1)[1]
    acc = pred.eq(data.y[y_mask]).sum().item() / y_mask.sum().item()

    return loss, acc
