from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import DenseGCNConv,DenseSAGEConv, dense_diff_pool, JumpingKnowledge


class Block(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, mode='cat'):
        super(Block, self).__init__()

        self.conv1 = DenseGCNConv(in_channels, out_channels)
        # self.conv2 = DenseGCNConv(hidden_channels, hidden_channels)
        # self.lin = Linear(hidden_channels + hidden_channels, out_channels)


    def reset_parameters(self):
        self.conv1.reset_parameters()
        # self.conv2.reset_parameters()
        # self.lin.reset_parameters()

    def forward(self, x, adj, mask=None, add_loop=True):
        x1 = self.conv1(x, adj, mask, add_loop)
        # x1 = F.relu(self.conv1(x, adj, mask, add_loop))
        # x2 = F.relu(self.conv2(x1, adj, mask, add_loop))
        # out = self.lin(torch.cat((x1, x2), -1))
        return x1


class DiffPool(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.25, lambda_=0.0):
        super(DiffPool, self).__init__()

        num_nodes = ceil(ratio * dataset[0].num_nodes)
        self.lambda_ = lambda_
        self.embed_block1 = Block(dataset.num_features, hidden, hidden)
        self.pool_block1 = Block(dataset.num_features, hidden, num_nodes)

        self.embed_blocks = torch.nn.ModuleList()
        self.pool_blocks = torch.nn.ModuleList()
        for i in range(num_layers-1):
            num_nodes = ceil(ratio * num_nodes)
            self.embed_blocks.append(Block(hidden, hidden, hidden))
            self.pool_blocks.append(Block(hidden, hidden, num_nodes))

        self.embed_final = Block(hidden, hidden, hidden)
        ###self.jump = JumpingKnowledge(mode='cat')
        ###self.lin1 = Linear((len(self.embed_blocks) + 1) * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.embed_block1.reset_parameters()
        self.pool_block1.reset_parameters()
        for embed_block, pool_block in zip(self.embed_blocks,
                                           self.pool_blocks):
            embed_block.reset_parameters()
            pool_block.reset_parameters()
        self.embed_final.reset_parameters()
        ###self.jump.reset_parameters()
        ###self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, adj, mask = data.x, data.adj, data.mask
        link_losses = 0.
        ent_losses = 0.
        s = self.pool_block1(x, adj, mask, add_loop=True)
        x = F.relu(self.embed_block1(x, adj, mask, add_loop=True))
        ###xs = [x.mean(dim=1)]
        x, adj, link_loss, ent_loss = dense_diff_pool(x, adj, s, mask)
        link_losses+=link_loss
        ent_losses+=ent_loss
        for i, (embed_block, pool_block) in enumerate(
                zip(self.embed_blocks, self.pool_blocks)):
            s = pool_block(x, adj)
            x = F.relu(embed_block(x, adj))
            ###xs.append(x.mean(dim=1))
            if i < len(self.embed_blocks) - 1:
                x, adj, link_loss, ent_loss = dense_diff_pool(x, adj, s)
                link_losses+=link_loss
                ent_losses+=ent_loss

        x = F.relu(self.embed_final(x, adj)).mean(dim=1)
        ###x = self.jump(xs)
        ###x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1), self.lambda_*(link_losses+ent_losses)

    def __repr__(self):
        return self.__class__.__name__
