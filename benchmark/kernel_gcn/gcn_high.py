import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, JumpingKnowledge
from gcn_conv_high import GCNConv_

class GCN_HIGH(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(GCN_HIGH, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.convs_random_1=GCNConv_random(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.convs_random = GCNConv_random(hidden, hidden)
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)
        self.tog = False

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        conv_1 = self.conv1(x, edge_index)
        random_1 = self.convs_random_1(x, edge_index)
        x = F.relu(0.95*conv_1+0.05*random_1)
        for conv in self.convs:
            conv_ = conv(x, edge_index)
            random_ = self.convs_random(x, edge_index)
            x = F.relu(0.95*conv_+0.05*random_)

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GCNWithJK_HIGH(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, mode='cat'):
        super(GCNWithJK_HIGH, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.convs_random_1=GCNConv_random(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.convs_random = GCNConv_random(hidden, hidden)
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)
        self.tog = False

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        conv_1 = self.conv1(x, edge_index)
        random_1 = self.convs_random_1(x, edge_index)
        x = F.relu(0.95*conv_1+0.05*random_1)
        xs = [x]
        for conv in self.convs:
            conv_ = conv(x, edge_index)
            random_ = self.convs_random(x, edge_index)
            x = F.relu(0.95*conv_+0.05*random_)
            xs += [x]
        x = self.jump(xs)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
