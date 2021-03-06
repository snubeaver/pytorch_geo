#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
from torch.nn import Linear
from datasets import get_planetoid_dataset
import pdb
from torch_geometric.nn import GCNConv


hidden=128
class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden)
        # self.conv2 = GCNConv(hidden, int(dataset[0].num_class))
        self.conv2 = GCNConv(dataset.num_features, int(dataset[0].num_class))
        self.lin = Linear(int(dataset[0].num_class),int(dataset[0].num_class))
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, data, edge_index):
        x= data.x
        # x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return  self.lin(F.relu(x))

