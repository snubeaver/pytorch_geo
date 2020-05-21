from math import ceil
from torch.nn import Linear
import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from torch_geometric.nn import DenseGCNConv, DenseSAGEConv, JumpingKnowledge
from model_graph import dense_diff_pool, get_Spectral_loss, dense_ssgpool, dense_ssgpool_gumbel
#from torch_geometric.nn import GCNConv


def l2norm(X, eps=1e-10):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm+eps)
    return X

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


class SSGPool(nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.1, lambda_=0.1):
        super(SSGPool, self).__init__()

        num_nodes = ceil(ratio * dataset[0].num_nodes)
        self.embed_block1 = Block(dataset.num_features, hidden, hidden) #DenseGCNConv(word_dim, embed_size)
        self.pool_block1 = Block(dataset.num_features, hidden, num_nodes) #DenseGCNConv(embed_size, 20) # 
        self.lambda_=lambda_
        self.embed_blocks = torch.nn.ModuleList()
        self.pool_blocks = torch.nn.ModuleList()
        for i in range(num_layers-1):
            num_nodes = ceil(ratio * num_nodes)
            self.embed_blocks.append(Block(hidden, hidden, hidden))
            self.pool_blocks.append(Block(hidden, hidden, num_nodes))

        self.embed_final = Block(hidden, hidden, hidden) #kwon added
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
        ###self.jump = JumpingKnowledge(mode='cat')
        ###self.lin1.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, data):
        #spec_losses = 0.
        x, adj, mask = data.x, data.adj, data.mask
        old_adj = adj
        B, N, _ = adj.size()
        s = self.pool_block1(x, adj, mask, add_loop=True)
        x = F.relu(self.embed_block1(x, adj, mask, add_loop=True))
        ###xs = [x.mean(dim=1)]
        #xs = [torch.sum(x, 1) / lengths.unsqueeze(-1).to(x.dtype)]
        #x, adj, link_loss, entr_loss, spec_loss = dense_diff_pool(x, adj, s, mask)
        #x, adj, link_loss, entr_loss, spec_loss = dense_diff_pool_lap(x, adj, s, mask)

        x, adj, Lapl, L_next, s, s_inv = dense_ssgpool(x, adj, s, mask)
        s_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
        s_inv_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
        s_final = torch.bmm(s_final, s)
        s_inv_final = torch.bmm(s_inv, s_inv_final)
        
        # multi layer implement
        for i, (embed_block, pool_block) in enumerate(
                zip(self.embed_blocks, self.pool_blocks)):
            s = pool_block(x, adj)
            x = F.relu(embed_block(x, adj))
            #xs.append(x.mean(dim=1))
            if i < len(self.embed_blocks) - 1:
                x, adj, _, L_next, s, s_inv = dense_ssgpool(x, adj, s)
                s_final = torch.bmm(s_final, s)
                s_inv_final = torch.bmm(s_inv, s_inv_final)

        x = F.relu(self.embed_final(x, adj)).mean(dim=1)
        # spectral loss
        #s_inv_final = s_final.transpose(1, 2) / ((s_final * s_final).sum(dim=1).unsqueeze(-1) + 1e-10)
        if self.lambda_ != 0.0:
            spec_loss = get_Spectral_loss(Lapl, L_next, s_inv_final.transpose(1, 2), 1, mask)
            spec_losses = spec_loss.mean()

            # coarsen loss
            identity = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
            old_adj = old_adj + identity# for new link loss
            coarsen_loss = old_adj - torch.matmul(s_final, s_final.transpose(1,2))
            mask_ = mask.view(B, N, 1).to(x.dtype)
            coarsen_loss = coarsen_loss * mask_
            coarsen_loss = coarsen_loss * mask_.transpose(1, 2)
            coarsen_loss = torch.sqrt((coarsen_loss * coarsen_loss).sum(dim=(1, 2)))
            coarsen_losses = coarsen_loss.mean()
        else:
            spec_losses = 0.0
            coarsen_losses = 0.0

        #spec_losses = coarsen_losses

        ###x = self.jump(xs)
        ###x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), self.lambda_*(spec_losses+coarsen_losses)


class SSGPool_gumbel(nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.25, lambda_=0.0):
        super(SSGPool_gumbel, self).__init__()

        num_nodes = ceil(ratio * dataset[0].num_nodes)
        self.embed_block1 = Block(dataset.num_features, hidden, hidden)  # DenseGCNConv(word_dim, embed_size)
        self.pool_block1 = Block(dataset.num_features, hidden, num_nodes)  # DenseGCNConv(embed_size, 20) #
        self.lambda_ = lambda_
        self.embed_blocks = torch.nn.ModuleList()
        self.pool_blocks = torch.nn.ModuleList()
        for i in range(num_layers - 1):
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
        ###self.jump = JumpingKnowledge(mode='cat')
        ###self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        # spec_losses = 0.
        x, adj, mask = data.x, data.adj, data.mask
        old_adj = adj
        B, N, _ = adj.size()
        s = self.pool_block1(x, adj, mask, add_loop=True)
        x = F.relu(self.embed_block1(x, adj, mask, add_loop=True))
        ###xs = [x.mean(dim=1)]
        # xs = [torch.sum(x, 1) / lengths.unsqueeze(-1).to(x.dtype)]
        # x, adj, link_loss, entr_loss, spec_loss = dense_diff_pool(x, adj, s, mask)
        # x, adj, link_loss, entr_loss, spec_loss = dense_diff_pool_lap(x, adj, s, mask)

        x, adj, Lapl, L_next, s, s_inv = dense_ssgpool_gumbel(x, adj, s, mask)
        s_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
        s_inv_final = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
        s_final = torch.bmm(s_final, s)
        s_inv_final = torch.bmm(s_inv, s_inv_final)

        # multi layer implement
        for i, (embed_block, pool_block) in enumerate(
                zip(self.embed_blocks, self.pool_blocks)):
            s = pool_block(x, adj)
            x = F.relu(embed_block(x, adj))
            ###xs.append(x.mean(dim=1))
            if i < len(self.embed_blocks) - 1:
                x, adj, _, L_next, s, s_inv = dense_ssgpool_gumbel(x, adj, s)
                s_final = torch.bmm(s_final, s)
                s_inv_final = torch.bmm(s_inv, s_inv_final)

        x = F.relu(self.embed_final(x, adj)).mean(dim=1)
        # spectral loss
        # s_inv_final = s_final.transpose(1, 2) / ((s_final * s_final).sum(dim=1).unsqueeze(-1) + 1e-10)
        if self.lambda_ != 0.0:
            spec_loss = get_Spectral_loss(Lapl, L_next, s_inv_final.transpose(1, 2), 1, mask)
            spec_losses = spec_loss.mean()

            # coarsen loss
            ###identity = torch.eye(N).unsqueeze(0).expand(B, N, N).cuda()
            ###old_adj = old_adj + identity  # for new link loss
            coarsen_loss = old_adj - torch.matmul(s_final, s_final.transpose(1, 2))
            mask_ = mask.view(B, N, 1).to(x.dtype)
            coarsen_loss = coarsen_loss * mask_
            coarsen_loss = coarsen_loss * mask_.transpose(1, 2)
            coarsen_loss = torch.sqrt((coarsen_loss * coarsen_loss).sum(dim=(1, 2)))
            coarsen_losses = coarsen_loss.mean()
        else:
            spec_losses = 0.0
            coarsen_losses = 0.0

        # spec_losses = coarsen_losses

        ###x = self.jump(xs)
        ###x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), self.lambda_ * (spec_losses + coarsen_losses)