import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax

import pdb
from inits import reset


class GATScore(torch.nn.Module):

    def __init__(self, gate_nn, nn=None):
        super(GATScore, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x, masked_node, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = x.size(0)
        feature_num = x.size(-1)
        masked_node = masked_node.unsqueeze(1)
        tiled_mask = masked_node.repeat(1,size,1)
        tiled_x = x.repeat(masked_node.size(0),1,1)
        tiled_mask = tiled_mask.view(-1, feature_num)
        tiled_x = tiled_x.view(-1, feature_num)


        x = torch.cat((x,tiled_mask), -1)
        gate = self.gate_nn(x).view(-1, 1)

        # gate = self.gate_nn(x)
        # query = self.gate_nn(masked_node)
        # x = self.nn(x) if self.nn is not None else x
        # assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        # gate = softmax(gate, batch, size)

        # output = F.softmax(output, 1)

        # gate = softmax(gate)
        return F.relu(output).to(x.device)

        # pdist = torch.nn.PairwiseDistance(p=2)
        # output = pdist(tiled_mask, tiled_x)

        # output = output.view(-1, size)



    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)
