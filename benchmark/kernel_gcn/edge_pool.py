import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (GCNConv, EdgePooling, global_mean_pool,
                                JumpingKnowledge)

class GNN_Block(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GNN_Block, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels + hidden_channels, hidden_channels)


    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        out = self.lin(torch.cat((x1, x2), -1))

        return out


class EdgePool(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio):
        super(EdgePool, self).__init__()
        self.conv1 = GNN_Block(dataset.num_features, hidden)
        self.pool1 = EdgePooling(hidden)
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GNN_Block(hidden, hidden)
            for i in range(num_layers - 1)
        ])
        self.pools.extend(
            [EdgePooling(hidden) for i in range((num_layers) -1)])
        self.embed_final = GNN_Block(hidden, hidden)
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear((num_layers+1)*hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        x, edge_index,batch, _ = self.pool1(x, edge_index,
                                                    batch=batch)
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            xs += [global_mean_pool(x, batch)]
            pool = self.pools[i]
            x, edge_index,batch, _ = pool(x, edge_index,
                                                    batch=batch)
        x = self.embed_final(x, edge_index)
        xs +=[global_mean_pool(x, batch)]
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
