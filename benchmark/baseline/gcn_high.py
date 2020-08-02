import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from gcn_conv_high import GCNConv_, GCNConv_random

from datasets import get_planetoid_dataset
from train_eval import random_planetoid_splits, run


class GCNHIGH(torch.nn.Module):
    def __init__(self, dataset, hidden, alpha):
        super(GCNHIGH, self).__init__()
        self.alpha = alpha*0.05
        self.conv1 = GCNConv_(dataset.num_features, hidden, self.alpha)
        self.conv2 = GCNConv_(hidden, dataset.num_classes, self.alpha)
        
        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()


    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)



