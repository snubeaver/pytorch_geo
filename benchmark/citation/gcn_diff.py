import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from gcn_conv_high import GCNConv_, GCNConv_random
import os
class GCNDiff(torch.nn.Module):
    def __init__(self, data, dataset, hidden):
        super(GCNDiff, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
                             normalize=False)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                             normalize=False)
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

